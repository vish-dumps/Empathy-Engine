"""Voice synthesis with sentence-wise modulation and pitch post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Callable, List, Optional, Sequence, Tuple

import pyttsx3

from audio_processing import change_pitch, merge_wav_files
from emotion import SentenceEmotion

RATE_MIN = 120
RATE_MAX = 230
VOLUME_MIN = 0.5
VOLUME_MAX = 1.0

PITCH_STEPS_BY_EMOTION = {
    "joy": 2,
    "surprise": 3,
    "sadness": -2,
    "fear": -1,
    "anger": 1,
    "neutral": 0,
}


@dataclass(frozen=True)
class SentenceVoiceResult:
    sentence: str
    emotion: str
    confidence: float
    rate: int
    volume: float
    pitch_shift: int


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


def pitch_shift_from_emotion(emotion: str) -> int:
    """Get semitone pitch shift for detected emotion."""
    return int(PITCH_STEPS_BY_EMOTION.get(emotion.strip().lower(), 0))


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


def _wait_for_audio_file(path: Path, timeout_seconds: float = 3.0) -> None:
    """Wait briefly for pyttsx3/audio writers to flush to disk."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 44:
            return
        time.sleep(0.05)

    raise RuntimeError(f"Failed to render audio file: {path}")


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


def _set_single_global_voice(engine: pyttsx3.Engine, voice_index: int = 0) -> None:
    """Set one consistent voice globally for the whole narration."""
    voices = engine.getProperty("voices") or []
    if not voices:
        return

    index = int(_clamp(float(voice_index), 0, len(voices) - 1))
    voice_id = str(getattr(voices[index], "id", ""))
    if voice_id:
        engine.setProperty("voice", voice_id)


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
    voice_index: int = 0,
    on_sentence_processed: Optional[Callable[[SentenceVoiceResult], None]] = None,
) -> List[SentenceVoiceResult]:
    """
    Speak each sentence with dynamic modulation, pitch-shift sentence audio,
    and merge all processed files into one final WAV.
    """
    if not sentence_emotions:
        return []

    if pause_range[0] < 0 or pause_range[1] < 0 or pause_range[0] > pause_range[1]:
        raise ValueError("pause_range must be a valid non-negative (min, max) tuple.")

    output_path = Path(output_file)
    smoothing = _clamp(transition_smoothing, 0.0, 1.0)

    temp_root = Path("temp")
    temp_root.mkdir(parents=True, exist_ok=True)
    run_temp_dir = Path(tempfile.mkdtemp(prefix="empathy_engine_", dir=str(temp_root)))
    base_dir = run_temp_dir / "base"
    processed_dir = run_temp_dir / "processed"
    base_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    export_dir: Optional[Path] = None
    if save_sentence_files:
        export_dir = Path(sentence_output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

    engine, cleanup = _initialize_tts_engine()
    _set_single_global_voice(engine, voice_index=voice_index)

    processed_files: List[Path] = []
    pauses: List[float] = []
    results: List[SentenceVoiceResult] = []
    previous_rate: Optional[int] = None
    previous_volume: Optional[float] = None

    try:
        try:
            total_sentences = len(sentence_emotions)
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
                pitch_shift = pitch_shift_from_emotion(sentence_emotion.emotion)

                engine.setProperty("rate", applied_rate)
                engine.setProperty("volume", applied_volume)

                base_sentence_path = base_dir / f"sentence_{index:03d}.wav"
                processed_sentence_path = processed_dir / f"processed_sentence_{index:03d}.wav"

                if play_during_generation:
                    engine.say(sentence_emotion.sentence)
                engine.save_to_file(sentence_emotion.sentence, str(base_sentence_path))
                engine.runAndWait()
                engine.stop()
                _wait_for_audio_file(base_sentence_path)

                change_pitch(
                    input_file=str(base_sentence_path),
                    output_file=str(processed_sentence_path),
                    n_steps=pitch_shift,
                )
                _wait_for_audio_file(processed_sentence_path)

                if export_dir is not None:
                    shutil.copy2(base_sentence_path, export_dir / base_sentence_path.name)
                    shutil.copy2(
                        processed_sentence_path,
                        export_dir / processed_sentence_path.name,
                    )

                processed_files.append(processed_sentence_path)

                result = SentenceVoiceResult(
                    sentence=sentence_emotion.sentence,
                    emotion=sentence_emotion.emotion,
                    confidence=sentence_emotion.confidence,
                    rate=applied_rate,
                    volume=applied_volume,
                    pitch_shift=pitch_shift,
                )
                results.append(result)

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

        merge_wav_files(processed_files=processed_files, output_file=output_path, pauses=pauses)
        return results
    finally:
        shutil.rmtree(run_temp_dir, ignore_errors=True)
        try:
            temp_root.rmdir()
        except OSError:
            pass
