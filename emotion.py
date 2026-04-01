"""Sentence-level emotion detection using a Hugging Face transformer model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List

from transformers import pipeline

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Normalize model output labels to the core set used by the voice mapper.
LABEL_ALIASES = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral",
    "disgust": "anger",
    "love": "joy",
}


@dataclass(frozen=True)
class SentenceEmotion:
    sentence: str
    emotion: str
    confidence: float
    raw_emotion: str
    all_scores: Dict[str, float]


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    return LABEL_ALIASES.get(normalized, "neutral")


@lru_cache(maxsize=1)
def get_emotion_classifier():
    """
    Load and cache the Hugging Face emotion classifier.

    The first run downloads model weights from Hugging Face and can take time.
    """
    try:
        return pipeline(
            "text-classification",
            model=MODEL_NAME,
            return_all_scores=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Hugging Face emotion model. "
            "Check internet access for first-time download and local cache permissions."
        ) from exc


def _aggregate_scores(score_entries: List[Dict[str, float]]) -> Dict[str, float]:
    """Convert raw model scores to normalized emotion-score mapping."""
    aggregated: Dict[str, float] = {}
    for entry in score_entries:
        label = _normalize_label(str(entry.get("label", "neutral")))
        score = float(entry.get("score", 0.0))
        aggregated[label] = max(score, aggregated.get(label, 0.0))
    return aggregated


def _extract_score_entries(raw_result: object) -> List[Dict[str, float]]:
    """
    Normalize different transformers output shapes to a flat list of score dicts.

    Depending on transformers version, outputs can be:
    - [{"label": "...", "score": ...}, ...]
    - [[{"label": "...", "score": ...}, ...]]
    - {"label": "...", "score": ...}
    """
    if isinstance(raw_result, dict):
        return [raw_result]

    if isinstance(raw_result, list):
        if not raw_result:
            return []

        first = raw_result[0]
        if isinstance(first, dict):
            return [entry for entry in raw_result if isinstance(entry, dict)]

        if isinstance(first, list):
            return [entry for entry in first if isinstance(entry, dict)]

    return []


def analyze_sentence(sentence: str) -> SentenceEmotion:
    """Analyze one sentence and return top emotion + confidence."""
    text = sentence.strip()
    if not text:
        return SentenceEmotion(
            sentence="",
            emotion="neutral",
            confidence=0.0,
            raw_emotion="neutral",
            all_scores={"neutral": 0.0},
        )

    classifier = get_emotion_classifier()
    raw_result = classifier(text, truncation=True, top_k=None)
    raw_scores = _extract_score_entries(raw_result)
    if not raw_scores:
        raise RuntimeError("Emotion model returned no scores for sentence analysis.")

    top_entry = max(raw_scores, key=lambda item: float(item.get("score", 0.0)))
    raw_label = str(top_entry.get("label", "neutral")).strip().lower()
    emotion = _normalize_label(raw_label)
    confidence = float(top_entry.get("score", 0.0))

    return SentenceEmotion(
        sentence=text,
        emotion=emotion,
        confidence=confidence,
        raw_emotion=raw_label,
        all_scores=_aggregate_scores(raw_scores),
    )


def analyze_sentences(sentences: Iterable[str]) -> List[SentenceEmotion]:
    """Analyze all sentences independently."""
    return [analyze_sentence(sentence) for sentence in sentences if sentence.strip()]
