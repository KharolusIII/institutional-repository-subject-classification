"""Long-document SentenceTransformer encoding and cache."""

from __future__ import annotations

import hashlib
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
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = SentenceTransformer(model_name, device=device)

    def _cache_path(self, texts: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            (self.model_name + self.mode + str(self.overlap) + "\0".join(texts)).encode("utf-8")
        ).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{digest}.npy"

    def encode(self, texts: list[str]) -> np.ndarray:
        cache_path = self._cache_path(texts)
        if cache_path and cache_path.exists():
            return np.load(cache_path)
        if self.mode == "legacy_truncated":
            result = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=True)
        else:
            tokenizer = self.model.tokenizer
            max_length = int(self.model.max_seq_length)
            pooled = []
            for text in texts:
                ids = tokenizer.encode(text, add_special_tokens=False)
                chunks = chunk_token_ids(ids, max_length=max_length, overlap=self.overlap)
                chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
                vectors = self.model.encode(
                    chunk_texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
                )
                pooled.append(pool_embeddings(vectors, [len(chunk) for chunk in chunks], self.mode))
            result = np.asarray(pooled)
        if cache_path:
            np.save(cache_path, result)
        return result

