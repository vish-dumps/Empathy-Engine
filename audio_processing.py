"""Audio post-processing helpers for pitch shifting and file merging."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Sequence
import warnings

import librosa
warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffmpeg or avconv*",
    category=RuntimeWarning,
)
from pydub import AudioSegment
import soundfile as sf


def change_pitch(input_file: str, output_file: str, n_steps: float) -> bool:
    """
    Apply semitone pitch shift to a WAV file.

    Returns True when pitch processing succeeds and False when a safe fallback
    copy of input audio is used.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or input_path.stat().st_size <= 44:
        raise RuntimeError(f"Input audio is missing or invalid: {input_path}")

    try:
        waveform, sample_rate = librosa.load(str(input_path), sr=None, mono=True)
    except Exception:
        shutil.copy2(input_path, output_path)
        return False

    if waveform.size == 0:
        shutil.copy2(input_path, output_path)
        return False

    try:
        if abs(n_steps) < 1e-9 or waveform.size < 512:
            shifted = waveform
        else:
            shifted = librosa.effects.pitch_shift(
                y=waveform,
                sr=sample_rate,
                n_steps=float(n_steps),
            )
    except Exception:
        shifted = waveform

    try:
        sf.write(str(output_path), shifted, sample_rate)
    except Exception:
        shutil.copy2(input_path, output_path)
        return False

    if output_path.stat().st_size <= 44:
        shutil.copy2(input_path, output_path)
        return False

    return True


def merge_wav_files(
    processed_files: Sequence[Path],
    output_file: Path,
    pauses: Sequence[float],
) -> None:
    """Merge processed sentence WAV files into one final WAV output."""
    if not processed_files:
        raise ValueError("No processed files were provided for merging.")

    combined = AudioSegment.silent(duration=0)
    for index, audio_path in enumerate(processed_files):
        if not audio_path.exists() or audio_path.stat().st_size <= 44:
            raise RuntimeError(f"Processed file is missing or invalid: {audio_path}")

        try:
            segment = AudioSegment.from_wav(str(audio_path))
        except Exception as exc:
            raise RuntimeError(f"Unable to read processed WAV file: {audio_path}") from exc

        combined += segment

        if index < len(processed_files) - 1:
            pause_seconds = pauses[index] if index < len(pauses) else 0.0
            if pause_seconds > 0:
                combined += AudioSegment.silent(duration=int(round(pause_seconds * 1000)))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_file), format="wav")

    if not output_file.exists() or output_file.stat().st_size <= 44:
        raise RuntimeError(f"Final merged output appears invalid: {output_file}")
