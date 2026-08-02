"""Supervised multi-label transformer fine-tuning over sampled document chunks."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import multilabel_metrics
from .pipeline import build_transformer_segments
from .thresholds import apply_thresholds, optimize_global_threshold, optimize_per_label_thresholds

LOGGER = logging.getLogger("ir_subject_classification.finetuning")


def _uniform_chunks(token_ids: list[int], length: int, maximum: int) -> list[list[int]]:
    chunks = [token_ids[start : start + length] for start in range(0, len(token_ids), length)] or [[]]
    if len(chunks) <= maximum:
        return chunks
    return [chunks[index] for index in np.linspace(0, len(chunks) - 1, maximum, dtype=int)]


def _add_special_tokens(tokenizer, token_ids: list[int]) -> list[int]:
    """Add single-sequence special tokens across Transformers tokenizer APIs."""
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(builder):
        return list(builder(token_ids))

    # Transformers 5 removed the public builder from some slow tokenizers.
    # The fine-tuned encoders used here are BERT-family models, whose single
    # sequence convention is [CLS] content [SEP].
    prefix = [int(tokenizer.cls_token_id)] if tokenizer.cls_token_id is not None else []
    suffix = [int(tokenizer.sep_token_id)] if tokenizer.sep_token_id is not None else []
    if not prefix and not suffix:
        raise AttributeError(
            f"{tokenizer.__class__.__name__} cannot add special tokens: "
            "no builder, cls_token_id, or sep_token_id is available"
        )
    return [*prefix, *token_ids, *suffix]


def aggregate_document_logits(logits: np.ndarray, document_ids: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros((count, logits.shape[1]), dtype=np.float32)
    frequencies = np.zeros(count, dtype=np.int64)
    for vector, document_id in zip(logits, document_ids):
        result[document_id] += vector
        frequencies[document_id] += 1
    return result / np.maximum(frequencies[:, None], 1)


def aggregate_weighted_document_logits(
    logits: np.ndarray, document_ids: np.ndarray, weights: np.ndarray, count: int
) -> np.ndarray:
    result = np.zeros((count, logits.shape[1]), dtype=np.float32)
    totals = np.zeros(count, dtype=np.float32)
    for vector, document_id, weight in zip(logits, document_ids, weights):
        result[document_id] += vector * weight
        totals[document_id] += weight
    return result / np.maximum(totals[:, None], 1e-12)


@dataclass
class EncodedChunks:
    input_ids: list[list[int]]
    document_ids: np.ndarray
    labels: np.ndarray
    document_weights: np.ndarray


def _encode_documents(
    tokenizer,
    documents: list[list[tuple[str, float]]],
    labels: np.ndarray,
    max_chunks: int,
    max_length: int,
) -> EncodedChunks:
    model_limit = int(getattr(tokenizer, "model_max_length", max_length))
    if model_limit <= 0 or model_limit > 1_000_000:
        model_limit = max_length
    sequence_length = min(max_length, model_limit)
    content_length = sequence_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_length < 1:
        raise ValueError("finetuning.max_length is too small for this tokenizer")
    input_ids: list[list[int]] = []
    document_ids: list[int] = []
    chunk_labels: list[np.ndarray] = []
    weights: list[float] = []
    for document_id, (units, label) in enumerate(zip(documents, labels)):
        for text, unit_weight in units:
            ids = tokenizer.encode(text, add_special_tokens=False)
            chunks = _uniform_chunks(ids, content_length, max_chunks)
            for chunk in chunks:
                input_ids.append(_add_special_tokens(tokenizer, chunk))
                document_ids.append(document_id)
                chunk_labels.append(label.astype(np.float32))
                weights.append(float(unit_weight) / len(chunks))
    return EncodedChunks(input_ids, np.asarray(document_ids), np.asarray(chunk_labels), np.asarray(weights))


def run_finetuning(dataset: pd.DataFrame, labels: list[str], config: dict, output_dir: str | Path) -> Path:
    """Fine-tune configured encoders and test only the validation-selected winner."""
    try:
        import torch
        from sklearn.preprocessing import MultiLabelBinarizer
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    except ImportError as exc:
        raise ImportError("Fine-tuning requires torch and transformers") from exc

    settings = config.get("finetuning", {})
    output = Path(output_dir) / "finetuning"
    output.mkdir(parents=True, exist_ok=True)
    split_fingerprint = str(config.get("_split_fingerprint", ""))
    context_path = output / "context.json"
    if any(output.iterdir()):
        stored_context = (
            json.loads(context_path.read_text(encoding="utf-8"))
            if context_path.exists()
            else {}
        )
        if stored_context.get("split_fingerprint") != split_fingerprint:
            raise RuntimeError(
                "Fine-tuning artifacts do not match the frozen dataset split. "
                "Start a new run_id; incompatible model checkpoints will not be reused."
            )
    else:
        context_path.write_text(
            json.dumps({"split_fingerprint": split_fingerprint}, indent=2) + "\n",
            encoding="utf-8",
        )
    if (output / "_VALIDATION_SUCCESS").exists():
        LOGGER.info("Reusing completed fine-tuning validation: %s", output)
        return output
    mlb = MultiLabelBinarizer(classes=labels)
    y_all = mlb.fit_transform(dataset["labels"])
    feature_set = settings.get("feature_set", "abstract+keywords+fulltext")
    documents = build_transformer_segments(
        dataset, feature_set, config.get("features", {}).get("field_weights")
    )
    split_indices = {
        split: np.flatnonzero(dataset["split"].eq(split))
        for split in ("train", "calibration", "validation", "test")
    }
    if not len(split_indices["calibration"]):
        raise ValueError("Fine-tuning requires a dedicated calibration split")
    models = settings.get(
        "models",
        {
            "sbert_finetuned": "sentence-transformers/distiluse-base-multilingual-cased-v1",
            "labse_finetuned": "sentence-transformers/LaBSE",
        },
    )
    validation_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    class ChunkDataset(Dataset):
        def __init__(self, encoded: EncodedChunks): self.encoded = encoded
        def __len__(self): return len(self.encoded.input_ids)
        def __getitem__(self, index):
            return {
                "input_ids": self.encoded.input_ids[index],
                "labels": self.encoded.labels[index],
                "weight": self.encoded.document_weights[index],
                "document_id": self.encoded.document_ids[index],
            }

    for name, model_name in models.items():
        model_dir = output / name
        model_dir.mkdir(exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoded = {
            split: _encode_documents(
                tokenizer,
                [documents[index] for index in indices],
                y_all[indices],
                int(settings.get("max_chunks_per_text_unit", settings.get("max_chunks_per_document", 8))),
                int(settings.get("max_length", 256)),
            )
            for split, indices in split_indices.items()
            if split in {"train", "calibration", "validation"}
        }
        collator_base = DataCollatorWithPadding(tokenizer, return_tensors="pt")

        def collate(rows):
            batch = collator_base([{"input_ids": row["input_ids"]} for row in rows])
            batch["labels"] = torch.tensor(np.asarray([row["labels"] for row in rows]), dtype=torch.float32)
            batch["weight"] = torch.tensor([row["weight"] for row in rows], dtype=torch.float32)
            batch["document_id"] = torch.tensor([row["document_id"] for row in rows], dtype=torch.long)
            return batch

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(labels), problem_type="multi_label_classification", ignore_mismatched_sizes=True
        ).to(device)
        if bool(settings.get("gradient_checkpointing", True)):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        positives = y_all[split_indices["train"]].sum(axis=0)
        negatives = len(split_indices["train"]) - positives
        loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(negatives / np.maximum(positives, 1), dtype=torch.float32, device=device),
            reduction="none",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("learning_rate", 2e-5)))
        checkpoint = model_dir / "checkpoint.pt"
        start_epoch = 0
        resume_step = 0
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            if state.get("split_fingerprint") != split_fingerprint:
                raise RuntimeError(
                    f"Fine-tuning checkpoint for {name} belongs to a different dataset split"
                )
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"])
            resume_step = int(state.get("step", 0))
            if state.get("torch_rng") is not None:
                torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        accumulation = int(settings.get("gradient_accumulation_steps", 4))
        checkpoint_steps = int(settings.get("checkpoint_steps", 500))
        for epoch in range(start_epoch, int(settings.get("epochs", 3))):
            generator = torch.Generator().manual_seed(int(config["experiment"].get("seed", 42)) + epoch)
            train_loader = DataLoader(
                ChunkDataset(encoded["train"]), batch_size=int(settings.get("batch_size", 16)), shuffle=True,
                generator=generator, collate_fn=collate, num_workers=int(settings.get("num_workers", 2)),
                pin_memory=device.type == "cuda",
            )
            model.train(); optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            processed_steps = 0
            for step, batch in enumerate(train_loader, 1):
                if epoch == start_epoch and step <= resume_step:
                    continue
                labels_batch = batch.pop("labels").to(device)
                weights = batch.pop("weight").to(device)
                batch.pop("document_id")
                inputs = {key: value.to(device) for key, value in batch.items()}
                use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(**inputs).logits
                    loss = (loss_function(logits, labels_batch).mean(dim=1) * weights).mean() / accumulation
                loss.backward()
                running_loss += float(loss.detach().cpu()) * accumulation
                processed_steps += 1
                if step % accumulation == 0:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                if checkpoint_steps and step % checkpoint_steps == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "step": step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                            "split_fingerprint": split_fingerprint,
                        },
                        checkpoint,
                    )
                    LOGGER.info(
                        "Fine-tuning model=%s epoch=%d/%d step=%d/%d mean_loss=%.6f checkpoint=%s",
                        name,
                        epoch + 1,
                        int(settings.get("epochs", 3)),
                        step,
                        len(train_loader),
                        running_loss / max(processed_steps, 1),
                        checkpoint,
                    )
            if len(train_loader) % accumulation:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": 0,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "split_fingerprint": split_fingerprint,
                },
                checkpoint,
            )
            LOGGER.info(
                "Fine-tuning epoch complete model=%s epoch=%d/%d mean_loss=%.6f",
                name,
                epoch + 1,
                int(settings.get("epochs", 3)),
                running_loss / max(processed_steps, 1),
            )
            resume_step = 0

        def predict(split: str) -> np.ndarray:
            loader = DataLoader(ChunkDataset(encoded[split]), batch_size=int(settings.get("eval_batch_size", 64)), collate_fn=collate)
            vectors=[]; document_ids=[]; model.eval()
            with torch.no_grad():
                for batch in loader:
                    batch.pop("labels"); batch.pop("weight")
                    document_ids.extend(batch.pop("document_id").numpy().tolist())
                    inputs={key:value.to(device) for key,value in batch.items()}
                    vectors.append(model(**inputs).logits.float().cpu().numpy())
            return aggregate_weighted_document_logits(
                np.concatenate(vectors),
                np.asarray(document_ids),
                encoded[split].document_weights,
                len(split_indices[split]),
            )

        calibration_scores = predict("calibration")
        validation_scores = predict("validation")
        global_threshold, _ = optimize_global_threshold(y_all[split_indices["calibration"]], calibration_scores, np.linspace(-6, 6, 49))
        thresholds = optimize_per_label_thresholds(
            y_all[split_indices["calibration"]], calibration_scores, global_threshold,
            int(config.get("thresholds", {}).get("minimum_label_support", 20)), np.linspace(-6, 6, 49),
        )
        metrics = multilabel_metrics(
            y_all[split_indices["validation"]], apply_thresholds(validation_scores, thresholds), validation_scores
        )
        validation_rows.append({"model": name, "model_name": model_name, "feature_set": feature_set, **metrics})
        np.save(model_dir / "thresholds.npy", thresholds)
        tokenizer.save_pretrained(model_dir / "tokenizer")
        model.save_pretrained(model_dir / "model")

    validation = pd.DataFrame(validation_rows).sort_values("f1_macro", ascending=False)
    validation.to_csv(output / "results_validation.csv", index=False)
    (output / "_VALIDATION_SUCCESS").write_text(
        "Fine-tuning validation completed; isolated test remains untouched.\n",
        encoding="utf-8",
    )
    LOGGER.info("Fine-tuning validation completed for %d encoders", len(validation))
    return output


def evaluate_finetuned_model(
    dataset: pd.DataFrame,
    labels: list[str],
    config: dict,
    output_dir: str | Path,
    model_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the single globally validation-selected fine-tuned encoder."""
    import torch
    from sklearn.preprocessing import MultiLabelBinarizer
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    settings = config["finetuning"]
    output = Path(output_dir) / "finetuning"
    model_dir = output / model_key
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir / "model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    mlb = MultiLabelBinarizer(classes=labels)
    y_all = mlb.fit_transform(dataset["labels"])
    test_idx = np.flatnonzero(dataset["split"].eq("test"))
    documents = build_transformer_segments(
        dataset, settings.get("feature_set", "abstract+keywords+fulltext"),
        config.get("features", {}).get("field_weights"),
    )
    encoded = _encode_documents(
        tokenizer,
        [documents[index] for index in test_idx],
        y_all[test_idx],
        int(settings.get("max_chunks_per_text_unit", settings.get("max_chunks_per_document", 8))),
        int(settings.get("max_length", 256)),
    )

    class TestChunks(Dataset):
        def __len__(self): return len(encoded.input_ids)
        def __getitem__(self, index):
            return {"input_ids": encoded.input_ids[index], "document_id": encoded.document_ids[index]}

    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    def collate(rows):
        batch = collator([{"input_ids": row["input_ids"]} for row in rows])
        batch["document_id"] = torch.tensor([row["document_id"] for row in rows], dtype=torch.long)
        return batch

    vectors = []
    document_ids = []
    loader = DataLoader(TestChunks(), batch_size=int(settings.get("eval_batch_size", 64)), collate_fn=collate)
    with torch.no_grad():
        for batch in loader:
            document_ids.extend(batch.pop("document_id").numpy().tolist())
            vectors.append(model(**{key: value.to(device) for key, value in batch.items()}).logits.float().cpu().numpy())
    scores = aggregate_weighted_document_logits(
        np.concatenate(vectors), np.asarray(document_ids), encoded.document_weights, len(test_idx)
    )
    thresholds = np.load(model_dir / "thresholds.npy")
    prediction = apply_thresholds(scores, thresholds)
    model_output = output / model_key
    np.save(model_output / "test_scores.npy", scores)
    (model_output / "_TEST_SUCCESS").write_text(
        "Pre-specified family representative evaluated successfully.\n", encoding="utf-8"
    )
    return scores, prediction, thresholds
