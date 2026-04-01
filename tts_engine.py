"""Neural TTS engine powered by Coqui TTS with sentence-level pitch processing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, List, Optional, Sequence
import warnings

import librosa
import numpy as np
from pydub import AudioSegment
from TTS.api import TTS
import soundfile as sf
import torch

from emotion import SentenceEmotion

warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffmpeg or avconv.*",
    category=RuntimeWarning,
)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MIN_PITCH_DURATION_SECONDS = 0.25

PAUSE_BY_EMOTION = {
    "joy": 0.3,
    "surprise": 0.3,
    "sadness": 0.7,
    "fear": 0.7,
    "anger": 0.4,
    "neutral": 0.5,
}

SPEED_BY_EMOTION = {
    "joy": 1.03,
    "surprise": 1.05,
    "sadness": 0.97,
    "fear": 0.98,
    "anger": 1.02,
    "neutral": 1.00,
}

TIME_STRETCH_BY_EMOTION = {
    "joy": 1.01,
    "surprise": 1.02,
    "sadness": 0.99,
    "fear": 0.99,
    "anger": 1.01,
    "neutral": 1.00,
}

VOLUME_BY_EMOTION = {
    "joy": (1.03, 1.10),
    "surprise": (1.05, 1.14),
    "sadness": (0.86, 0.95),
    "fear": (0.90, 0.98),
    "anger": (1.06, 1.16),
    "neutral": (1.00, 1.00),
}

PITCH_MIN_STEPS = -3.0
PITCH_MAX_STEPS = 3.0
MAX_PITCH_STEP_CHANGE = 1.0
PITCH_BLEND_RATIO = 0.80


@dataclass(frozen=True)
class PitchProcessResult:
    output_file: str
    pitch_shift: float
    volume_gain: float
    duration_before: float
    duration_after: float
    waveform_delta: float
    pitch_applied: bool
    volume_applied: bool


@dataclass(frozen=True)
class SentenceTTSResult:
    sentence: str
    emotion: str
    confidence: float
    file_generated: str
    raw_file_generated: str = ""
    speed: float = 1.0
    pause_after: float = 0.0
    pitch_shift: float = 0.0
    volume_gain: float = 1.0
    duration_before: float = 0.0
    duration_after: float = 0.0
    waveform_delta: float = 0.0
    pitch_applied: bool = False
    volume_applied: bool = False


def pause_for_emotion(emotion: str) -> float:
    """Get pause duration between sentences based on emotion."""
    return float(PAUSE_BY_EMOTION.get(emotion.strip().lower(), 0.5))


def speed_for_emotion(emotion: str) -> float:
    """Get speaking speed modifier for sentence synthesis."""
    return float(SPEED_BY_EMOTION.get(emotion.strip().lower(), 1.0))


def volume_for_emotion(emotion: str, confidence: float) -> float:
    """Get amplitude gain based on emotion and confidence."""
    label = emotion.strip().lower()
    score = max(0.0, min(1.0, float(confidence)))
    low, high = VOLUME_BY_EMOTION.get(label, (1.0, 1.0))
    return float(low + ((high - low) * score))


def pitch_shift_for_emotion(emotion: str, confidence: float) -> float:
    """
    Get cohesive semitone pitch shifts with confidence scaling.

    Mapping:
    - joy: +1.0..+2.0
    - surprise: +1.5..+2.5
    - sadness: -1.5..-2.5
    - fear: -0.8..-1.6
    - anger: +0.8..+1.6
    - neutral: 0.0 (no pitch shift)
    """
    label = emotion.strip().lower()
    score = max(0.0, min(1.0, float(confidence)))

    if label == "joy":
        return 1.0 + (1.0 * score)
    if label == "surprise":
        return 1.5 + (1.0 * score)
    if label == "sadness":
        return -(1.5 + (1.0 * score))
    if label == "fear":
        return -(0.8 + (0.8 * score))
    if label == "anger":
        return 0.8 + (0.8 * score)
    return 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def smooth_pitch_shift(target_shift: float, previous_shift: Optional[float], emotion: str) -> float:
    """
    Keep adjacent sentence pitch modulation cohesive.

    Neutral is fixed to 0.0 by design.
    """
    if emotion.strip().lower() == "neutral":
        return 0.0

    target = _clamp(target_shift, PITCH_MIN_STEPS, PITCH_MAX_STEPS)
    if previous_shift is None:
        return target

    blended = (PITCH_BLEND_RATIO * target) + ((1.0 - PITCH_BLEND_RATIO) * previous_shift)
    delta = blended - previous_shift
    delta = _clamp(delta, -MAX_PITCH_STEP_CHANGE, MAX_PITCH_STEP_CHANGE)
    smoothed = previous_shift + delta
    return _clamp(smoothed, PITCH_MIN_STEPS, PITCH_MAX_STEPS)


@lru_cache(maxsize=1)
def get_tts_model() -> TTS:
    """
    Load and cache Coqui XTTS model.

    First run downloads model weights and can take significant time.
    """
    # Required for non-interactive model downloads when CPML acceptance is needed.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    try:
        tts = TTS(model_name=MODEL_NAME, progress_bar=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Coqui TTS model. "
            "Ensure internet access for first-time model download."
        ) from exc

    if torch.cuda.is_available() and hasattr(tts, "to"):
        try:
            tts = tts.to("cuda")
        except Exception:
            pass

    return tts


def _resolve_default_language(tts: TTS, preferred: str = "en") -> Optional[str]:
    if not getattr(tts, "is_multi_lingual", False):
        return None

    languages = getattr(tts, "languages", None)
    if isinstance(languages, (list, tuple)) and languages:
        if preferred in languages:
            return preferred
        return str(languages[0])
    return preferred


def _resolve_default_speaker(tts: TTS) -> Optional[str]:
    if not getattr(tts, "is_multi_speaker", False):
        return None

    speakers = getattr(tts, "speakers", None)
    if isinstance(speakers, (list, tuple)) and speakers:
        return str(speakers[0])
    return None


def _validate_audio_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 44:
        raise RuntimeError(f"Generated audio file is missing or invalid: {path}")


def _tts_to_file_safe(
    tts: TTS,
    *,
    text: str,
    file_path: Path,
    speed: float,
    language: Optional[str],
    speaker: Optional[str],
    speaker_wav: Optional[str],
) -> None:
    signature = inspect.signature(tts.tts_to_file)
    supported = set(signature.parameters.keys())

    kwargs = {"text": text, "file_path": str(file_path)}
    if "speed" in supported:
        kwargs["speed"] = speed
    if language and "language" in supported:
        kwargs["language"] = language
    if speaker_wav and "speaker_wav" in supported:
        kwargs["speaker_wav"] = speaker_wav
    elif speaker and "speaker" in supported:
        kwargs["speaker"] = speaker

    try:
        tts.tts_to_file(**kwargs)
    except TypeError:
        minimal_kwargs = {"text": text, "file_path": str(file_path)}
        if language and "language" in supported:
            minimal_kwargs["language"] = language
        if speaker_wav and "speaker_wav" in supported:
            minimal_kwargs["speaker_wav"] = speaker_wav
        elif speaker and "speaker" in supported:
            minimal_kwargs["speaker"] = speaker
        tts.tts_to_file(**minimal_kwargs)


def change_pitch(
    input_file: str,
    output_file: str,
    n_steps: float,
    emotion: str,
    volume_gain: float,
) -> PitchProcessResult:
    """
    Apply pitch shifting + volume gain using librosa and save with soundfile.

    Pipeline:
    - load raw sentence wav
    - apply pitch shift
    - optional subtle time-stretch to improve perceptibility
    - write processed wav
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _validate_audio_file(input_path)

    try:
        y, sr = librosa.load(str(input_path), sr=None)
    except Exception as exc:
        raise RuntimeError(f"Unable to load input audio for pitch processing: {input_path}") from exc

    duration_before = float(len(y) / sr) if sr else 0.0
    if y.size == 0:
        raise RuntimeError(f"Input waveform is empty: {input_path}")

    pitch_applied = True
    volume_applied = True
    y_shifted = y
    if duration_before < MIN_PITCH_DURATION_SECONDS or abs(n_steps) < 1e-3:
        pitch_applied = False
    else:
        try:
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        except Exception as exc:
            raise RuntimeError(f"Pitch shift failed for file: {input_path}") from exc

        stretch_rate = TIME_STRETCH_BY_EMOTION.get(emotion.strip().lower(), 1.0)
        if abs(stretch_rate - 1.0) > 1e-3 and y_shifted.size > 512:
            try:
                y_shifted = librosa.effects.time_stretch(y_shifted, rate=stretch_rate)
            except Exception:
                pass

    safe_gain = float(_clamp(volume_gain, 0.6, 1.4))
    if abs(safe_gain - 1.0) < 1e-3:
        volume_applied = False
    else:
        y_shifted = np.clip(y_shifted * safe_gain, -1.0, 1.0)

    duration_after = float(len(y_shifted) / sr) if sr else 0.0
    compare_len = min(len(y), len(y_shifted))
    if compare_len > 0:
        waveform_delta = float(np.mean(np.abs(y[:compare_len] - y_shifted[:compare_len])))
    else:
        waveform_delta = 0.0

    try:
        sf.write(str(output_path), y_shifted, sr)
    except Exception as exc:
        raise RuntimeError(f"Unable to write processed audio file: {output_path}") from exc

    _validate_audio_file(output_path)

    return PitchProcessResult(
        output_file=str(output_path.resolve()),
        pitch_shift=float(n_steps),
        volume_gain=safe_gain,
        duration_before=duration_before,
        duration_after=duration_after,
        waveform_delta=waveform_delta,
        pitch_applied=pitch_applied,
        volume_applied=volume_applied,
    )


def _merge_processed_audio(
    processed_files: Sequence[Path],
    pauses: Sequence[float],
    output_file: Path,
) -> None:
    if not processed_files:
        raise ValueError("No processed sentence files were provided for merging.")

    first_segment = AudioSegment.from_wav(str(processed_files[0]))
    target_rate = first_segment.frame_rate
    target_channels = first_segment.channels
    target_width = first_segment.sample_width

    combined = AudioSegment.silent(duration=0)
    for index, sentence_file in enumerate(processed_files):
        _validate_audio_file(sentence_file)
        segment = AudioSegment.from_wav(str(sentence_file))
        segment = (
            segment.set_frame_rate(target_rate)
            .set_channels(target_channels)
            .set_sample_width(target_width)
        )
        combined += segment

        if index < len(processed_files) - 1:
            pause_seconds = pauses[index] if index < len(pauses) else 0.5
            combined += AudioSegment.silent(duration=int(round(pause_seconds * 1000)))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_file), format="wav")
    _validate_audio_file(output_file)


def render_emotional_speech(
    sentence_emotions: Sequence[SentenceEmotion],
    output_file: str = "output.wav",
    save_sentence_files: bool = False,
    sentence_output_dir: str = "sentence_audio",
    speaker_wav: Optional[str] = None,
    on_sentence_processed: Optional[Callable[[SentenceTTSResult], None]] = None,
) -> List[SentenceTTSResult]:
    """Generate raw sentence audio, pitch-shift it, then merge processed files only."""
    if not sentence_emotions:
        return []

    output_path = Path(output_file)
    temp_root = Path("temp")
    temp_root.mkdir(parents=True, exist_ok=True)
    run_temp_dir = Path(tempfile.mkdtemp(prefix="empathy_engine_", dir=str(temp_root)))
    raw_dir = run_temp_dir / "raw"
    processed_dir = run_temp_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    export_dir: Optional[Path] = None
    if save_sentence_files:
        export_dir = Path(sentence_output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

    if speaker_wav:
        speaker_wav_path = Path(speaker_wav)
        if not speaker_wav_path.exists():
            raise RuntimeError(f"speaker_wav file was not found: {speaker_wav}")
        speaker_wav = str(speaker_wav_path.resolve())

    tts = get_tts_model()
    language = _resolve_default_language(tts)
    speaker = _resolve_default_speaker(tts)

    processed_files: List[Path] = []
    pauses: List[float] = []
    results: List[SentenceTTSResult] = []
    previous_pitch_shift: Optional[float] = None

    try:
        total_sentences = len(sentence_emotions)
        for index, sentence_emotion in enumerate(sentence_emotions, start=1):
            raw_path = raw_dir / f"sentence_{index:03d}.wav"
            processed_path = processed_dir / f"processed_sentence_{index:03d}.wav"
            speed = speed_for_emotion(sentence_emotion.emotion)
            target_volume_gain = volume_for_emotion(
                emotion=sentence_emotion.emotion,
                confidence=sentence_emotion.confidence,
            )

            _tts_to_file_safe(
                tts=tts,
                text=sentence_emotion.sentence,
                file_path=raw_path,
                speed=speed,
                language=language,
                speaker=speaker,
                speaker_wav=speaker_wav,
            )
            _validate_audio_file(raw_path)

            target_pitch_steps = pitch_shift_for_emotion(
                emotion=sentence_emotion.emotion,
                confidence=sentence_emotion.confidence,
            )
            pitch_steps = smooth_pitch_shift(
                target_shift=target_pitch_steps,
                previous_shift=previous_pitch_shift,
                emotion=sentence_emotion.emotion,
            )
            pitch_result = change_pitch(
                input_file=str(raw_path),
                output_file=str(processed_path),
                n_steps=pitch_steps,
                emotion=sentence_emotion.emotion,
                volume_gain=target_volume_gain,
            )
            previous_pitch_shift = pitch_steps

            if export_dir is not None:
                export_raw = export_dir / raw_path.name
                export_processed = export_dir / processed_path.name
                shutil.copy2(raw_path, export_raw)
                shutil.copy2(processed_path, export_processed)
                raw_generated = str(export_raw.resolve())
                processed_generated = str(export_processed.resolve())
            else:
                raw_generated = str(raw_path.resolve())
                processed_generated = str(processed_path.resolve())

            processed_files.append(processed_path)
            pause_after = (
                pause_for_emotion(sentence_emotion.emotion)
                if index < total_sentences
                else 0.0
            )
            result = SentenceTTSResult(
                sentence=sentence_emotion.sentence,
                emotion=sentence_emotion.emotion,
                confidence=sentence_emotion.confidence,
                file_generated=processed_generated,
                raw_file_generated=raw_generated,
                speed=speed,
                pause_after=pause_after,
                pitch_shift=pitch_result.pitch_shift,
                volume_gain=pitch_result.volume_gain,
                duration_before=pitch_result.duration_before,
                duration_after=pitch_result.duration_after,
                waveform_delta=pitch_result.waveform_delta,
                pitch_applied=pitch_result.pitch_applied,
                volume_applied=pitch_result.volume_applied,
            )
            results.append(result)

            if on_sentence_processed is not None:
                on_sentence_processed(result)

            if index < total_sentences:
                pauses.append(pause_after)

        _merge_processed_audio(
            processed_files=processed_files,
            pauses=pauses,
            output_file=output_path,
        )
        return results
    finally:
        shutil.rmtree(run_temp_dir, ignore_errors=True)
        try:
            temp_root.rmdir()
        except OSError:
            pass
