"""Field-level preprocessing for sparse and transformer representations."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "es": {"el", "la", "los", "las", "de", "del", "y", "en", "para", "por", "un", "una"},
    "en": {"the", "a", "an", "and", "of", "in", "for", "to", "with", "on"},
    "pt": {"o", "a", "os", "as", "de", "da", "do", "e", "em", "para", "com"},
    "fr": {"le", "la", "les", "de", "des", "et", "en", "pour", "un", "une"},
}


def normalize_text(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\x00", " ")
    return re.sub(r"\s+", " ", value).strip()


def remove_language_stopwords(text: str, language: str) -> str:
    if language == "und" or language not in STOPWORDS:
        return text
    stopwords = STOPWORDS[language]
    return " ".join(token for token in text.split() if token.lower() not in stopwords)


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

