"""Supervised multi-label transformer fine-tuning over sampled document chunks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import per_label_evaluation
from .metrics import multilabel_metrics
from .pipeline import build_transformer_text
from .thresholds import apply_thresholds, optimize_global_threshold, optimize_per_label_thresholds

LOGGER = logging.getLogger("ir_subject_classification.finetuning")


def _uniform_chunks(token_ids: list[int], length: int, maximum: int) -> list[list[int]]:
    chunks = [token_ids[start : start + length] for start in range(0, len(token_ids), length)] or [[]]
    if len(chunks) <= maximum:
        return chunks
    return [chunks[index] for index in np.linspace(0, len(chunks) - 1, maximum, dtype=int)]


def aggregate_document_logits(logits: np.ndarray, document_ids: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros((count, logits.shape[1]), dtype=np.float32)
    frequencies = np.zeros(count, dtype=np.int64)
    for vector, document_id in zip(logits, document_ids):
        result[document_id] += vector
        frequencies[document_id] += 1
    return result / np.maximum(frequencies[:, None], 1)


@dataclass
class EncodedChunks:
    input_ids: list[list[int]]
    document_ids: np.ndarray
    labels: np.ndarray
    document_weights: np.ndarray


def _encode_documents(
    tokenizer, texts: list[str], labels: np.ndarray, max_chunks: int, max_length: int
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
    for document_id, (text, label) in enumerate(zip(texts, labels)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = _uniform_chunks(ids, content_length, max_chunks)
        for chunk in chunks:
            input_ids.append(tokenizer.build_inputs_with_special_tokens(chunk))
            document_ids.append(document_id)
            chunk_labels.append(label.astype(np.float32))
            weights.append(1.0 / len(chunks))
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
    if (output / "_SUCCESS").exists():
        LOGGER.info("Reusing completed fine-tuning run: %s", output)
        return output
    mlb = MultiLabelBinarizer(classes=labels)
    y_all = mlb.fit_transform(dataset["labels"])
    feature_set = settings.get("feature_set", "abstract+keywords+fulltext")
    texts = build_transformer_text(dataset, feature_set)
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
                [texts[index] for index in indices],
                y_all[indices],
                int(settings.get("max_chunks_per_document", 8)),
                int(settings.get("max_length", 256)),
            )
            for split, indices in split_indices.items()
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
                        },
                        checkpoint,
                    )
            if len(train_loader) % accumulation:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.save(
                {"epoch": epoch + 1, "step": 0, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                checkpoint,
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
            return aggregate_document_logits(np.concatenate(vectors), np.asarray(document_ids), len(split_indices[split]))

        calibration_scores = predict("calibration")
        validation_scores = predict("validation")
        # Computing and caching logits is label-blind. Metrics remain unavailable
        # until the validation winner has been frozen below.
        np.save(model_dir / "test_logits.npy", predict("test"))
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
    best = validation.iloc[0]
    # The isolated test is evaluated exactly once, for the validation-selected encoder.
    best_dir = output / str(best.model)
    (output / "selected_model.json").write_text(json.dumps(best.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    test_scores = np.load(best_dir / "test_logits.npy")
    thresholds = np.load(best_dir / "thresholds.npy")
    test_prediction = apply_thresholds(test_scores, thresholds)
    test_metrics = multilabel_metrics(
        y_all[split_indices["test"]], test_prediction, test_scores
    )
    pd.DataFrame([{"model": best.model, "model_name": best.model_name, **test_metrics}]).to_csv(
        output / "results_test.csv", index=False
    )
    per_label_evaluation(
        y_all[split_indices["test"]],
        test_prediction,
        test_scores,
        labels,
        y_all[split_indices["train"]].sum(axis=0),
        y_all[split_indices["validation"]].sum(axis=0),
        thresholds,
    ).to_csv(output / "per_label_test.csv", index=False)
    (output / "_SUCCESS").write_text("Fine-tuning completed successfully.\n", encoding="utf-8")
    LOGGER.info(
        "Fine-tuning completed; selected model=%s test_f1_macro=%.6f",
        best.model,
        test_metrics["f1_macro"],
    )
    return output
