"""Pluggable language identification with auditable scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LanguagePrediction:
    language: str
    score: float | None = None


class LanguageDetector(Protocol):
    def detect(self, text: str) -> LanguagePrediction: ...


class HeuristicLanguageDetector:
    """Offline deterministic fallback intended for tests and smoke runs."""

    markers = {
        "es": {" el ", " la ", " los ", " las ", " de ", " para ", " investigación ", " educación "},
        "en": {" the ", " and ", " of ", " for ", " research ", " with ", " study "},
        "pt": {" os ", " as ", " uma ", " para ", " pesquisa ", " educação ", " com "},
        "fr": {" le ", " la ", " les ", " des ", " une ", " recherche ", " avec "},
    }

    def __init__(self, min_chars: int = 12) -> None:
        self.min_chars = min_chars

    def detect(self, text: str) -> LanguagePrediction:
        normalized = f" {' '.join(str(text or '').lower().split())} "
        if len(normalized.strip()) < self.min_chars:
            return LanguagePrediction("und", None)
        scores = {lang: sum(marker in normalized for marker in markers) for lang, markers in self.markers.items()}
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        if scores[best] == 0:
            return LanguagePrediction("und", None)
        return LanguagePrediction(best, scores[best] / total if total else None)


class LinguaLanguageDetector:
    def __init__(self) -> None:
        try:
            from lingua import LanguageDetectorBuilder
        except ImportError as exc:
            raise ImportError("Install the 'language' extra to use Lingua") from exc
        self._detector = LanguageDetectorBuilder.from_all_languages().build()

    def detect(self, text: str) -> LanguagePrediction:
        if not str(text or "").strip():
            return LanguagePrediction("und", None)
        language = self._detector.detect_language_of(text)
        if language is None:
            return LanguagePrediction("und", None)
        iso = language.iso_code_639_1
        score = self._detector.compute_language_confidence(text, language)
        return LanguagePrediction(iso.name.lower(), float(score))


class LangidLanguageDetector:
    def __init__(self) -> None:
        try:
            import langid
        except ImportError as exc:
            raise ImportError("Install the 'language' extra to use langid") from exc
        self._langid = langid

    def detect(self, text: str) -> LanguagePrediction:
        if not str(text or "").strip():
            return LanguagePrediction("und", None)
        language, score = self._langid.classify(text)
        return LanguagePrediction(language or "und", float(score))


class LangdetectLanguageDetector:
    def __init__(self) -> None:
        try:
            from langdetect import detect_langs
        except ImportError as exc:
            raise ImportError("Install the 'language' extra to use langdetect") from exc
        self._detect_langs = detect_langs

    def detect(self, text: str) -> LanguagePrediction:
        if not str(text or "").strip():
            return LanguagePrediction("und", None)
        try:
            predictions = self._detect_langs(text)
        except Exception:
            return LanguagePrediction("und", None)
        if not predictions:
            return LanguagePrediction("und", None)
        return LanguagePrediction(predictions[0].lang, float(predictions[0].prob))


def create_language_detector(backend: str = "heuristic", **kwargs: object) -> LanguageDetector:
    if backend == "heuristic":
        return HeuristicLanguageDetector(**kwargs)
    if backend == "lingua":
        return LinguaLanguageDetector()
    if backend == "langid":
        return LangidLanguageDetector()
    if backend == "langdetect":
        return LangdetectLanguageDetector()
    raise ValueError(f"Unsupported language backend: {backend}")
