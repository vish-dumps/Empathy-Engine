"""Voice synthesis with sentence-wise transformer emotion modulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Callable, List, Optional, Sequence, Tuple
import wave

import pyttsx3

from emotion import SentenceEmotion

RATE_MIN = 120
RATE_MAX = 230
VOLUME_MIN = 0.5
VOLUME_MAX = 1.0


@dataclass(frozen=True)
class SentenceVoiceResult:
    sentence: str
    emotion: str
    confidence: float
    rate: int
    volume: float
    voice_used: str


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def modulation_from_emotion(
    emotion: str,
    confidence: float,
    base_rate: int = 170,
    base_volume: float = 0.9,
) -> Tuple[int, float]:
    """
    Map emotion + confidence to rate and volume.

    Rules:
    - joy: rate = base + score*100; volume = min(1.0, base + 0.2)
    - sadness/fear: rate = base - score*80; volume = max(0.5, base - 0.3)
    - anger: rate = base + score*60; volume = 1.0
    - surprise: rate = base + score*120
    - neutral: base values
    """
    label = emotion.strip().lower()
    score = _clamp(confidence, 0.0, 1.0)

    rate = float(base_rate)
    volume = float(base_volume)

    if label == "joy":
        rate = base_rate + (score * 100)
        volume = min(1.0, base_volume + 0.2)
    elif label in {"sadness", "fear"}:
        rate = base_rate - (score * 80)
        volume = max(0.5, base_volume - 0.3)
    elif label == "anger":
        rate = base_rate + (score * 60)
        volume = 1.0
    elif label == "surprise":
        rate = base_rate + (score * 120)

    return (
        int(_clamp(round(rate), RATE_MIN, RATE_MAX)),
        float(_clamp(volume, VOLUME_MIN, VOLUME_MAX)),
    )


def _select_voice_for_emotion(
    voices: Sequence[object],
    emotion: str,
    fallback_voice_id: str,
) -> Tuple[str, str]:
    """Select voice ID and human-readable voice name for the emotion."""
    if not voices:
        return fallback_voice_id, "default"

    label = emotion.strip().lower()
    if label in {"joy", "surprise"}:
        index = 0
    elif label in {"sadness", "fear"}:
        index = 1 if len(voices) > 1 else 0
    elif label == "anger":
        if len(voices) > 2:
            index = 2
        elif len(voices) > 1:
            index = 1
        else:
            index = 0
    else:
        index = 0

    selected = voices[index]
    voice_id = str(getattr(selected, "id", fallback_voice_id))
    voice_name = str(getattr(selected, "name", voice_id))
    return voice_id, voice_name


def _apply_transition_smoothing(
    target_rate: int,
    target_volume: float,
    previous_rate: Optional[int],
    previous_volume: Optional[float],
    smoothing_factor: float,
) -> Tuple[int, float]:
    if previous_rate is None or previous_volume is None or smoothing_factor <= 0:
        return target_rate, target_volume

    factor = _clamp(smoothing_factor, 0.0, 1.0)
    smoothed_rate = previous_rate + ((target_rate - previous_rate) * factor)
    smoothed_volume = previous_volume + ((target_volume - previous_volume) * factor)

    return (
        int(_clamp(round(smoothed_rate), RATE_MIN, RATE_MAX)),
        float(_clamp(smoothed_volume, VOLUME_MIN, VOLUME_MAX)),
    )


def _wait_for_audio_file(path: Path, timeout_seconds: float = 2.0) -> None:
    """Wait briefly for pyttsx3 to flush audio to disk."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 44:
            return
        time.sleep(0.05)

    raise RuntimeError(f"Failed to render audio file: {path}")


def _merge_wav_files(
    sentence_files: Sequence[Path],
    output_file: Path,
    pauses: Sequence[float],
) -> None:
    if not sentence_files:
        raise ValueError("No sentence audio files were provided for merging.")

    with wave.open(str(sentence_files[0]), "rb") as first_reader:
        params = first_reader.getparams()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_file), "wb") as writer:
        writer.setparams(params)

        for index, sentence_file in enumerate(sentence_files):
            with wave.open(str(sentence_file), "rb") as reader:
                if (
                    reader.getnchannels() != params.nchannels
                    or reader.getsampwidth() != params.sampwidth
                    or reader.getframerate() != params.framerate
                    or reader.getcomptype() != params.comptype
                ):
                    raise RuntimeError(
                        "Inconsistent WAV format across sentence files; cannot merge."
                    )

                writer.writeframes(reader.readframes(reader.getnframes()))

            if index < len(sentence_files) - 1:
                pause_seconds = pauses[index] if index < len(pauses) else 0.0
                if pause_seconds > 0:
                    silence_frames = int(params.framerate * pause_seconds)
                    silence_bytes = (
                        b"\x00"
                        * silence_frames
                        * params.nchannels
                        * params.sampwidth
                    )
                    writer.writeframes(silence_bytes)


def _initialize_tts_engine() -> Tuple[pyttsx3.Engine, Optional[Callable[[], None]]]:
    """
    Initialize pyttsx3 safely.

    On Windows, explicitly initialize COM to avoid failures in threaded runtimes
    (for example, Streamlit request threads).
    """
    cleanup: Optional[Callable[[], None]] = None
    if sys.platform.startswith("win"):
        try:
            import pythoncom  # type: ignore

            pythoncom.CoInitialize()
            cleanup = pythoncom.CoUninitialize
        except Exception:
            cleanup = None

    engine = pyttsx3.init()
    return engine, cleanup


def render_emotional_speech(
    sentence_emotions: Sequence[SentenceEmotion],
    output_file: str = "output.wav",
    base_rate: int = 170,
    base_volume: float = 0.9,
    pause_range: Tuple[float, float] = (0.5, 0.8),
    save_sentence_files: bool = False,
    sentence_output_dir: str = "sentence_audio",
    transition_smoothing: float = 0.0,
    play_during_generation: bool = True,
    on_sentence_processed: Optional[Callable[[SentenceVoiceResult], None]] = None,
) -> List[SentenceVoiceResult]:
    """
    Speak each sentence with dynamic voice modulation and save one merged WAV file.

    Returns sentence-wise applied settings for terminal/GUI reporting.
    """
    if not sentence_emotions:
        return []

    if pause_range[0] < 0 or pause_range[1] < 0 or pause_range[0] > pause_range[1]:
        raise ValueError("pause_range must be a valid non-negative (min, max) tuple.")

    total_sentences = len(sentence_emotions)
    output_path = Path(output_file)
    smoothing = _clamp(transition_smoothing, 0.0, 1.0)

    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    if save_sentence_files:
        sentence_dir = Path(sentence_output_dir)
        sentence_dir.mkdir(parents=True, exist_ok=True)
        working_dir = sentence_dir
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="empathy_engine_")
        working_dir = Path(temp_dir.name)

    engine, cleanup = _initialize_tts_engine()
    voices = engine.getProperty("voices") or []
    default_voice_id = str(engine.getProperty("voice") or "")

    sentence_files: List[Path] = []
    pauses: List[float] = []
    results: List[SentenceVoiceResult] = []
    previous_rate: Optional[int] = None
    previous_volume: Optional[float] = None

    try:
        for index, sentence_emotion in enumerate(sentence_emotions, start=1):
            target_rate, target_volume = modulation_from_emotion(
                emotion=sentence_emotion.emotion,
                confidence=sentence_emotion.confidence,
                base_rate=base_rate,
                base_volume=base_volume,
            )
            applied_rate, applied_volume = _apply_transition_smoothing(
                target_rate=target_rate,
                target_volume=target_volume,
                previous_rate=previous_rate,
                previous_volume=previous_volume,
                smoothing_factor=smoothing,
            )

            voice_id, voice_name = _select_voice_for_emotion(
                voices=voices,
                emotion=sentence_emotion.emotion,
                fallback_voice_id=default_voice_id,
            )

            engine.setProperty("voice", voice_id)
            engine.setProperty("rate", applied_rate)
            engine.setProperty("volume", applied_volume)

            sentence_audio_path = working_dir / f"sentence_{index:03d}.wav"
            if play_during_generation:
                engine.say(sentence_emotion.sentence)
            engine.save_to_file(sentence_emotion.sentence, str(sentence_audio_path))
            engine.runAndWait()
            engine.stop()
            _wait_for_audio_file(sentence_audio_path)

            result = SentenceVoiceResult(
                sentence=sentence_emotion.sentence,
                emotion=sentence_emotion.emotion,
                confidence=sentence_emotion.confidence,
                rate=applied_rate,
                volume=applied_volume,
                voice_used=voice_name,
            )
            results.append(result)
            sentence_files.append(sentence_audio_path)

            if on_sentence_processed is not None:
                on_sentence_processed(result)

            if index < total_sentences:
                pause_seconds = random.uniform(pause_range[0], pause_range[1])
                pauses.append(pause_seconds)
                time.sleep(pause_seconds)

            previous_rate = applied_rate
            previous_volume = applied_volume
    finally:
        engine.stop()
        if cleanup is not None:
            cleanup()

    _merge_wav_files(sentence_files, output_path, pauses)

    if temp_dir is not None:
        temp_dir.cleanup()

    return results
