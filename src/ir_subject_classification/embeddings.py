"""Long-document SentenceTransformer encoding and cache."""

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
        cached_vectors: list[np.ndarray | None] = []
        missing_indices: list[int] = []
        for index, text in enumerate(texts):
            document_cache = self._document_cache_path(text)
            if document_cache and document_cache.exists():
                cached_vectors.append(np.load(document_cache))
            else:
                cached_vectors.append(None)
                missing_indices.append(index)
        if self.mode == "legacy_truncated":
            for batch_start in range(0, len(missing_indices), self.document_batch_size):
                indices = missing_indices[batch_start : batch_start + self.document_batch_size]
                vectors = self.model.encode(
                    [texts[index] for index in indices],
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )
                for index, vector in zip(indices, vectors):
                    cached_vectors[index] = vector
                    document_cache = self._document_cache_path(texts[index])
                    if document_cache:
                        self._save_array_atomic(document_cache, vector)
            result = np.asarray(cached_vectors)
            self.last_statistics = {
                "mode": self.mode,
                "documents": len(texts),
                "documents_loaded_from_cache": len(texts) - len(missing_indices),
                "cache_hit": 0,
            }
        else:
            tokenizer = self.model.tokenizer
            max_length = int(self.model.max_seq_length)
            content_max_length = max(
                1, max_length - int(tokenizer.num_special_tokens_to_add(pair=False))
            )
            pooled = cached_vectors
            total_tokens = 0
            total_chunks = 0
            for batch_start in range(0, len(missing_indices), self.document_batch_size):
                indices = missing_indices[batch_start : batch_start + self.document_batch_size]
                document_chunks: list[list[list[int]]] = []
                chunk_texts: list[str] = []
                for index in indices:
                    text = texts[index]
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    total_tokens += len(ids)
                    chunks = chunk_token_ids(
                        ids, max_length=content_max_length, overlap=self.overlap
                    )
                    total_chunks += len(chunks)
                    document_chunks.append(chunks)
                    chunk_texts.extend(tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks)
                vectors = self.model.encode(
                    chunk_texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
                )
                offset = 0
                for index, chunks in zip(indices, document_chunks):
                    count = len(chunks)
                    vector = pool_embeddings(
                        vectors[offset : offset + count],
                        [len(chunk) for chunk in chunks],
                        self.mode,
                    )
                    pooled[index] = vector
                    document_cache = self._document_cache_path(texts[index])
                    if document_cache:
                        self._save_array_atomic(document_cache, vector)
                    offset += count
            result = np.asarray(pooled)
            self.last_statistics = {
                "mode": self.mode,
                "documents": len(texts),
                "source_tokens": total_tokens,
                "chunks": total_chunks,
                "model_max_tokens": max_length,
                "max_content_tokens_per_chunk": content_max_length,
                "overlap_tokens": self.overlap,
                "documents_loaded_from_cache": len(texts) - len(missing_indices),
                "cache_hit": 0,
            }
        if cache_path:
            self._save_array_atomic(cache_path, result)
        return result

