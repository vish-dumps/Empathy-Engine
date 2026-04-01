"""Streamlit frontend for interactive Empathy Engine usage."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List
import zipfile

import streamlit as st

from emotion import analyze_sentences
from utils import split_sentences
from voice import SentenceVoiceResult, render_emotional_speech


def _zip_sentence_audio(sentence_dir: Path) -> bytes:
    """Zip all sentence WAV files and return archive bytes."""
    archive_path = sentence_dir.parent / "sentence_audio.zip"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for wav_file in sorted(sentence_dir.glob("*.wav")):
            archive.write(wav_file, arcname=wav_file.name)
    return archive_path.read_bytes()


def _emotion_counts(results: List[SentenceVoiceResult]) -> Dict[str, int]:
    counts = {
        "joy": 0,
        "sadness": 0,
        "anger": 0,
        "fear": 0,
        "surprise": 0,
        "neutral": 0,
    }
    for result in results:
        key = result.emotion.lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _result_rows(results: List[SentenceVoiceResult]) -> List[dict]:
    rows = []
    for index, result in enumerate(results, start=1):
        row = asdict(result)
        rows.append(
            {
                "Sentence #": index,
                "Sentence": row["sentence"],
                "Emotion": str(row["emotion"]).title(),
                "Confidence": f"{row['confidence']:.2f}",
                "Rate": row["rate"],
                "Volume": f"{row['volume']:.2f}",
                "Voice": row["voice_used"],
            }
        )
    return rows


def _clear_previous_result_state() -> None:
    for key in ("audio_bytes", "rows", "sentence_zip_bytes", "sentence_count"):
        st.session_state.pop(key, None)


def _render_results() -> None:
    audio_bytes = st.session_state.get("audio_bytes")
    rows = st.session_state.get("rows")
    sentence_zip_bytes = st.session_state.get("sentence_zip_bytes")

    if not audio_bytes or not rows:
        return

    st.subheader("Generated Audio")
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        "Download output.wav",
        data=audio_bytes,
        file_name="output.wav",
        mime="audio/wav",
    )
    if sentence_zip_bytes:
        st.download_button(
            "Download sentence_audio.zip",
            data=sentence_zip_bytes,
            file_name="sentence_audio.zip",
            mime="application/zip",
        )

    results_for_counts = []
    for row in rows:
        results_for_counts.append(
            SentenceVoiceResult(
                sentence=row["Sentence"],
                emotion=str(row["Emotion"]).lower(),
                confidence=float(row["Confidence"]),
                rate=int(row["Rate"]),
                volume=float(row["Volume"]),
                voice_used=str(row["Voice"]),
            )
        )

    counts = _emotion_counts(results_for_counts)

    c1, c2, c3 = st.columns(3)
    c1.metric("Joy", counts.get("joy", 0))
    c2.metric("Sadness", counts.get("sadness", 0))
    c3.metric("Neutral", counts.get("neutral", 0))

    c4, c5, c6 = st.columns(3)
    c4.metric("Anger", counts.get("anger", 0))
    c5.metric("Fear", counts.get("fear", 0))
    c6.metric("Surprise", counts.get("surprise", 0))

    st.subheader("Sentence-Level Analysis")
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Empathy Engine", page_icon="voice", layout="wide")
    st.title("Empathy Engine")
    st.caption(
        "Convert paragraph text into emotionally expressive speech with sentence-level modulation."
    )

    input_text = st.text_area(
        "Input paragraph",
        height=220,
        placeholder="Type or paste a paragraph with multiple sentences...",
    )

    with st.expander("Voice Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        base_rate = col1.slider("Base Rate", min_value=120, max_value=220, value=170, step=1)
        base_volume = col2.slider(
            "Base Volume", min_value=0.5, max_value=1.0, value=0.9, step=0.01
        )
        smoothing = col3.slider(
            "Transition Smoothing", min_value=0.0, max_value=1.0, value=0.0, step=0.05
        )

        p1, p2, p3 = st.columns(3)
        min_pause = p1.slider("Min Pause (s)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
        max_pause = p2.slider("Max Pause (s)", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
        live_playback = p3.checkbox(
            "Live Speaker Playback",
            value=False,
            help="If enabled, audio also plays on your system speaker while generating.",
        )

        save_sentence_files = st.checkbox(
            "Save separate audio for each sentence",
            value=False,
        )

    generate_clicked = st.button("Generate Emotional Speech", type="primary")

    if generate_clicked:
        _clear_previous_result_state()

        if not input_text.strip():
            st.error("Input text is empty.")
            _render_results()
            return
        if min_pause > max_pause:
            st.error("Min pause must be less than or equal to max pause.")
            _render_results()
            return

        sentences = split_sentences(input_text)
        if not sentences:
            st.error("No sentences detected in input.")
            _render_results()
            return

        sentence_emotions = analyze_sentences(sentences)
        if not sentence_emotions:
            st.error("Sentence analysis failed.")
            _render_results()
            return

        with st.spinner("Analyzing sentiment and generating speech..."):
            run_dir = Path(tempfile.mkdtemp(prefix="empathy_engine_ui_"))
            output_file = run_dir / "output.wav"
            sentence_dir = run_dir / "sentence_audio"

            try:
                results = render_emotional_speech(
                    sentence_emotions=sentence_emotions,
                    output_file=str(output_file),
                    base_rate=base_rate,
                    base_volume=base_volume,
                    pause_range=(min_pause, max_pause),
                    save_sentence_files=save_sentence_files,
                    sentence_output_dir=str(sentence_dir),
                    transition_smoothing=smoothing,
                    play_during_generation=live_playback,
                )

                st.session_state["audio_bytes"] = output_file.read_bytes()
                st.session_state["rows"] = _result_rows(results)
                st.session_state["sentence_count"] = len(results)

                if save_sentence_files:
                    st.session_state["sentence_zip_bytes"] = _zip_sentence_audio(sentence_dir)
            except Exception as exc:
                st.error(f"Speech generation failed: {exc}")
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)

        if st.session_state.get("sentence_count", 0):
            st.success(f"Generated audio for {st.session_state['sentence_count']} sentence(s).")

    _render_results()


if __name__ == "__main__":
    main()
