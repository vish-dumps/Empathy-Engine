"""Utility helpers for text preprocessing and sentence segmentation."""

from __future__ import annotations

from typing import List

import nltk
from nltk.tokenize import sent_tokenize


def _ensure_sentence_tokenizer() -> None:
    """Ensure NLTK sentence tokenizer resources are available."""
    resources = [
        ("tokenizers/punkt", "punkt", True),
        # Newer NLTK versions may also rely on punkt_tab.
        ("tokenizers/punkt_tab", "punkt_tab", False),
    ]

    for resource_path, package_name, required in resources:
        try:
            nltk.data.find(resource_path)
            continue
        except LookupError:
            pass

        downloaded = nltk.download(package_name, quiet=True)
        if not downloaded and required:
            raise RuntimeError(
                f"Unable to download required NLTK package '{package_name}'."
            )

        try:
            nltk.data.find(resource_path)
        except LookupError:
            if required:
                raise RuntimeError(
                    f"Required NLTK resource '{resource_path}' is missing."
                )


def split_sentences(text: str) -> List[str]:
    """Split text into cleaned sentences using NLTK sent_tokenize."""
    cleaned = text.strip()
    if not cleaned:
        return []

    _ensure_sentence_tokenizer()
    sentences = [sentence.strip() for sentence in sent_tokenize(cleaned) if sentence.strip()]

    if not sentences:
        return [cleaned]

    return sentences
