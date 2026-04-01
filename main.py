"""CLI entry point for the Empathy Engine project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

from emotion import analyze_sentences
from tts_engine import SentenceTTSResult, render_emotional_speech
from utils import split_sentences


def _read_multiline_input() -> str:
    """Read paragraph input from CLI until an empty line or EOF."""
    print("Enter your paragraph. Submit an empty line to finish:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if not line.strip():
            if lines:
                break
            continue

        lines.append(line.strip())

    return " ".join(lines).strip()


def _resolve_input_text(cli_text: Optional[str]) -> str:
    if cli_text and cli_text.strip():
        return cli_text.strip()
    return _read_multiline_input()


def _print_sentence_feedback(result: SentenceTTSResult) -> None:
    print(f'Sentence: "{result.sentence}"')
    print(f"Emotion: {result.emotion}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Speed: {result.speed:.2f}x")
    print(f"Pause After: {result.pause_after:.2f}s")
    print(f"Input File: {result.raw_file_generated}")
    print(f"Output File: {result.file_generated}")
    print(f"Pitch Shift (n_steps): {result.pitch_shift:+.2f}")
    print(f"Volume Gain: {result.volume_gain:.2f}x")
    print(f"Duration Before: {result.duration_before:.3f}s")
    print(f"Duration After: {result.duration_after:.3f}s")
    print(f"Waveform Delta: {result.waveform_delta:.6f}")
    print(f"Pitch Applied: {result.pitch_applied}")
    print(f"Volume Applied: {result.volume_applied}")
    print("-" * 50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Empathy Engine: sentence-wise emotional text-to-speech."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input paragraph. If omitted, text is read interactively.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Path for merged output WAV file.",
    )
    parser.add_argument(
        "--save-sentences",
        action="store_true",
        help="Also save separate WAV files for each sentence.",
    )
    parser.add_argument(
        "--sentence-dir",
        type=str,
        default="sentence_audio",
        help="Directory for per-sentence WAV files when --save-sentences is used.",
    )
    parser.add_argument(
        "--speaker-wav",
        type=str,
        default=None,
        help="Optional speaker reference WAV for consistent voice cloning.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    text = _resolve_input_text(args.text)
    if not text:
        print("Error: input text is empty.", file=sys.stderr)
        return 1

    sentences = split_sentences(text)
    if not sentences:
        print("Error: no sentences were detected in input.", file=sys.stderr)
        return 1

    sentence_emotions = analyze_sentences(sentences)
    if not sentence_emotions:
        print("Error: sentence analysis failed.", file=sys.stderr)
        return 1

    print(f"Detected {len(sentence_emotions)} sentence(s). Starting speech synthesis...")

    try:
        render_emotional_speech(
            sentence_emotions=sentence_emotions,
            output_file=args.output,
            save_sentence_files=args.save_sentences,
            sentence_output_dir=args.sentence_dir,
            speaker_wav=args.speaker_wav,
            on_sentence_processed=_print_sentence_feedback,
        )
    except Exception as exc:
        print(f"Error during synthesis: {exc}", file=sys.stderr)
        return 1

    print(f"Merged audio saved to: {Path(args.output).resolve()}")
    if args.save_sentences:
        print(f"Sentence WAV files saved in: {Path(args.sentence_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
