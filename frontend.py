"""Streamlit frontend for Empathy Engine using Coqui TTS."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List

import streamlit as st

from emotion import analyze_sentences
from tts_engine import SentenceTTSResult, render_emotional_speech
from utils import split_sentences


def _result_rows(results: List[SentenceTTSResult]) -> List[dict]:
    rows = []
    for index, result in enumerate(results, start=1):
        row = asdict(result)
        rows.append(
            {
                "Sentence #": index,
                "Sentence": row["sentence"],
                "Emotion": str(row["emotion"]).title(),
                "Confidance": f"{row['confidence']:.2f}",
                "Speed (x)": f"{float(row['speed']):.2f}",
                "Pitch Shift": f"{float(row['pitch_shift']):+.2f}",
                "Volume Gain (x)": f"{float(row['volume_gain']):.2f}",
            }
        )
    return rows


def _clear_previous_result_state() -> None:
    for key in ("audio_bytes", "rows", "sentence_count"):
        st.session_state.pop(key, None)


def _render_results() -> None:
    audio_bytes = st.session_state.get("audio_bytes")
    rows = st.session_state.get("rows")

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

    counts = {
        "joy": 0,
        "sadness": 0,
        "anger": 0,
        "fear": 0,
        "surprise": 0,
        "neutral": 0,
    }
    for row in rows:
        key = str(row["Emotion"]).lower()
        counts[key] = counts.get(key, 0) + 1
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
    st.set_page_config(page_title="Empathy Engine", page_icon="speech_balloon", layout="wide")
    st.title("Empathy Engine")
    st.caption("Sentence-level emotion detection with neural speech synthesis (Coqui TTS).")

    input_text = st.text_area(
        "Input paragraph",
        height=220,
        placeholder="Type or paste a paragraph with multiple sentences...",
    )

    generate_clicked = st.button("Generate Emotional Speech", type="primary")

    if generate_clicked:
        _clear_previous_result_state()

        if not input_text.strip():
            st.error("Input text is empty.")
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

        with st.spinner("Analyzing sentiment and generating neural speech..."):
            run_dir = Path(tempfile.mkdtemp(prefix="empathy_engine_ui_"))
            output_file = run_dir / "output.wav"

            try:
                results = render_emotional_speech(
                    sentence_emotions=sentence_emotions,
                    output_file=str(output_file),
                )

                st.session_state["audio_bytes"] = output_file.read_bytes()
                st.session_state["rows"] = _result_rows(results)
                st.session_state["sentence_count"] = len(results)
            except Exception as exc:
                st.error(f"Speech generation failed: {exc}")
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)

        if st.session_state.get("sentence_count", 0):
            st.success(f"Generated audio for {st.session_state['sentence_count']} sentence(s).")

    _render_results()


if __name__ == "__main__":
    main()
