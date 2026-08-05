"""Fixed long-document SentenceTransformer embeddings and cache.

The historical ``sbert`` identifier is the DistilUSE checkpoint. This module
uses each complete Sentence-Transformers pipeline without parameter updates.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def chunk_token_ids(token_ids: list[int], max_length: int, overlap: int = 0) -> list[list[int]]:
    if max_length <= 0 or overlap < 0 or overlap >= max_length:
        raise ValueError("Require max_length > 0 and 0 <= overlap < max_length")
    step = max_length - overlap
    return [token_ids[start : start + max_length] for start in range(0, len(token_ids), step)] or [[]]


def pool_embeddings(embeddings: np.ndarray, lengths: list[int], mode: str) -> np.ndarray:
    if mode == "chunked_mean":
        return embeddings.mean(axis=0)
    if mode == "chunked_length_weighted_mean":
        weights = np.asarray(lengths, dtype=float)
        if weights.sum() == 0:
            return embeddings.mean(axis=0)
        return np.average(embeddings, axis=0, weights=weights)
    if mode == "chunked_max":
        return embeddings.max(axis=0)
    raise ValueError(f"Unknown pooling mode: {mode}")


class DocumentEmbedder:
    def __init__(
        self,
        model_name: str,
        mode: str = "legacy_truncated",
        overlap: int = 32,
        batch_size: int = 32,
        document_batch_size: int = 32,
        cache_dir: str | Path | None = None,
        device: str | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install the 'dense' extra to use embeddings") from exc
        self.model_name = model_name
        self.mode = mode
        self.overlap = overlap
        self.batch_size = batch_size
        self.document_batch_size = document_batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = SentenceTransformer(model_name, device=device)
        self.last_statistics: dict[str, int | float | str] = {}

    def _cache_path(self, texts: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        hasher = hashlib.sha256()
        for value in (
            self.model_name,
            self.mode,
            str(self.overlap),
            str(self.document_batch_size),
        ):
            hasher.update(value.encode("utf-8"))
            hasher.update(b"\0")
        for text in texts:
            hasher.update(text.encode("utf-8"))
            hasher.update(b"\0")
        digest = hasher.hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{digest}.npy"

    def _document_cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        hasher = hashlib.sha256()
        for value in (self.model_name, self.mode, str(self.overlap), text):
            hasher.update(value.encode("utf-8"))
            hasher.update(b"\0")
        directory = self.cache_dir / "documents"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{hasher.hexdigest()}.npy"

    def _batch_cache_path(self, texts: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        hasher = hashlib.sha256()
        for value in (self.model_name, self.mode, str(self.overlap), *texts):
            hasher.update(value.encode("utf-8"))
            hasher.update(b"\0")
        directory = self.cache_dir / "batches"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{hasher.hexdigest()}.npy"

    @staticmethod
    def _save_array_atomic(path: Path, value: np.ndarray) -> None:
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, value)
        os.replace(temporary, path)

    def encode(self, texts: list[str]) -> np.ndarray:
        cache_path = self._cache_path(texts)
        if cache_path and cache_path.exists():
            result = np.load(cache_path)
            self.last_statistics = {
                "mode": self.mode,
                "documents": len(texts),
                "cache_hit": 1,
            }
            return result
        batches: list[np.ndarray] = []
        cached_documents = 0
        total_tokens = 0
        total_chunks = 0
        tokenizer = self.model.tokenizer if self.mode != "legacy_truncated" else None
        max_length = int(self.model.max_seq_length) if tokenizer is not None else 0
        content_max_length = (
            max(1, max_length - int(tokenizer.num_special_tokens_to_add(pair=False)))
            if tokenizer is not None
            else 0
        )
        for batch_start in range(0, len(texts), self.document_batch_size):
            batch_texts = texts[batch_start : batch_start + self.document_batch_size]
            batch_cache = self._batch_cache_path(batch_texts)
            if batch_cache and batch_cache.exists():
                batches.append(np.load(batch_cache))
                cached_documents += len(batch_texts)
                continue
            if self.mode == "legacy_truncated":
                batch_vectors = self.model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )
            else:
                document_chunks: list[list[list[int]]] = []
                chunk_texts: list[str] = []
                for text in batch_texts:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    total_tokens += len(ids)
                    chunks = chunk_token_ids(
                        ids, max_length=content_max_length, overlap=self.overlap
                    )
                    total_chunks += len(chunks)
                    document_chunks.append(chunks)
                    chunk_texts.extend(
                        tokenizer.decode(chunk, skip_special_tokens=True)
                        for chunk in chunks
                    )
                chunk_vectors = self.model.encode(
                    chunk_texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                pooled = []
                offset = 0
                for chunks in document_chunks:
                    count = len(chunks)
                    pooled.append(
                        pool_embeddings(
                            chunk_vectors[offset : offset + count],
                            [len(chunk) for chunk in chunks],
                            self.mode,
                        )
                    )
                    offset += count
                batch_vectors = np.asarray(pooled)
            if batch_cache:
                self._save_array_atomic(batch_cache, np.asarray(batch_vectors))
            batches.append(np.asarray(batch_vectors))
        result = np.concatenate(batches, axis=0) if batches else np.empty((0, 0))
        if self.mode == "legacy_truncated":
            self.last_statistics = {
                "mode": self.mode,
                "documents": len(texts),
                "documents_loaded_from_cache": cached_documents,
                "cache_hit": 0,
                "cache_layout": "document_batches",
            }
        else:
            self.last_statistics = {
                "mode": self.mode,
                "documents": len(texts),
                "source_tokens": total_tokens,
                "chunks": total_chunks,
                "model_max_tokens": max_length,
                "max_content_tokens_per_chunk": content_max_length,
                "overlap_tokens": self.overlap,
                "documents_loaded_from_cache": cached_documents,
                "cache_hit": 0,
                "cache_layout": "document_batches",
            }
        if cache_path:
            self._save_array_atomic(cache_path, result)
        return result

    def encode_segmented(
        self, documents: list[list[tuple[str, float]]]
    ) -> np.ndarray:
        """Pool chunks within units, then weighted units within each handle."""
        flattened: list[str] = []
        layout: list[tuple[int, int, list[float]]] = []
        for units in documents:
            start = len(flattened)
            valid = [(text, float(weight)) for text, weight in units if str(text).strip()]
            flattened.extend(text for text, _ in valid)
            layout.append((start, len(valid), [weight for _, weight in valid]))
        vectors = self.encode(flattened)
        unit_statistics = dict(self.last_statistics)
        if not len(flattened):
            return np.empty((len(documents), 0))
        pooled = []
        for start, count, weights in layout:
            if not count:
                pooled.append(np.zeros(vectors.shape[1], dtype=vectors.dtype))
                continue
            values = np.asarray(weights, dtype=float)
            values = values / values.sum() if values.sum() else np.full(count, 1 / count)
            pooled.append(np.average(vectors[start : start + count], axis=0, weights=values))
        self.last_statistics = {
            **unit_statistics,
            "documents": len(documents),
            "text_units": len(flattened),
            "aggregation": "chunks_to_text_unit_to_handle",
        }
        return np.asarray(pooled)

