"""Field-level preprocessing for sparse and transformer representations."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


@lru_cache(maxsize=None)
def language_stopwords(language: str) -> frozenset[str]:
    """Return Stopwords ISO terms for an ISO 639-1 language code."""
    if not language or language == "und":
        return frozenset()
    try:
        import stopwordsiso
    except ImportError as exc:
        raise ImportError(
            "Install stopwordsiso to use preprocessing mode=language_stopwords"
        ) from exc
    if language not in stopwordsiso.langs():
        return frozenset()
    return frozenset(word.casefold() for word in stopwordsiso.stopwords(language))


def normalize_text(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\x00", " ")
    return re.sub(r"\s+", " ", value).strip()


def remove_language_stopwords(text: str, language: str) -> str:
    stopwords = language_stopwords(language)
    if not stopwords:
        return text
    tokens = re.findall(r"(?u)\b\w+\b", text)
    return " ".join(token for token in tokens if token.casefold() not in stopwords)


def preprocess_sparse(text: object, language: str, mode: str) -> str:
    if mode == "raw":
        return str(text or "")
    normalized = normalize_text(text)
    if mode == "normalized":
        return normalized
    if mode == "language_stopwords":
        return remove_language_stopwords(normalized, language)
    raise ValueError(f"Unknown sparse preprocessing mode: {mode}")


def preprocess_transformer(text: object) -> str:
    """Technical cleanup only: never removes stopwords or performs stemming."""
    return normalize_text(text)

