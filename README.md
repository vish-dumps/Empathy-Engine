<<<<<<< HEAD
# Empathy-Engine
Giving AI a human emotion
=======
# Empathy Engine

Empathy Engine converts paragraph input into emotionally expressive speech,
processing one sentence at a time and adapting voice delivery dynamically.

We use a transformer-based emotion classification model to capture nuanced emotional context beyond simple polarity scores.

## Why Hugging Face Instead of TextBlob

The project now uses `j-hartmann/emotion-english-distilroberta-base` from
Hugging Face via `transformers.pipeline(...)`.

Advantages over polarity-only analysis:

- Detects discrete emotions (`joy`, `sadness`, `anger`, `fear`, `surprise`,
  `neutral`) instead of only positive/negative polarity.
- Produces confidence scores per sentence for stronger intensity control.
- Handles mixed emotional context more accurately in multi-sentence paragraphs.

## Features

- CLI input support:
  - `python main.py`
  - `python main.py --text "your paragraph"`
- Sentence segmentation with NLTK `sent_tokenize`
- Transformer-based sentence-level emotion detection
- Confidence-aware voice modulation with clamped safety bounds
- Emotion-specific voice selection (with fallback for limited voice lists)
- Sequential sentence speaking with natural pause (`0.5s` to `0.8s`)
- Optional frontend UI (`streamlit run frontend.py`)
- Single merged WAV output (`output.wav`) plus optional per-sentence files

## Project Structure

```text
project/
|-- main.py
|-- emotion.py
|-- voice.py
|-- utils.py
|-- requirements.txt
|-- README.md
```

Note: `frontend.py` is included as an optional UI layer and reuses the same
core modules.

## Installation

```bash
pip install -r requirements.txt
```

## Run

CLI interactive mode:

```bash
python main.py
```

CLI argument mode:

```bash
python main.py --text "I got promoted. I'm still nervous about the new responsibilities."
```

Optional frontend:

```bash
streamlit run frontend.py
```

## Sentence-Level Emotional Modulation Pipeline

1. Split paragraph into sentences using NLTK.
2. Classify each sentence with transformer model scores (`return_all_scores=True`).
3. Pick top emotion label and confidence for each sentence.
4. Map emotion + confidence to rate/volume and voice choice.
5. Speak each sentence independently with `runAndWait()`, `stop()`, and pause.
6. Merge generated sentence WAV files into a single `output.wav`.

## Emotion to Voice Mapping

Base values:

- `rate = 170`
- `volume = 0.9`

Mapping logic:

- `joy`
  - `rate = base + score * 100`
  - `volume = min(1.0, base + 0.2)`
  - preferred voice: `voices[0]`
- `sadness` / `fear`
  - `rate = base - score * 80`
  - `volume = max(0.5, base - 0.3)`
  - preferred voice: `voices[1]` (if available)
- `anger`
  - `rate = base + score * 60`
  - `volume = 1.0`
  - preferred voice: stronger alternate voice (if available)
- `surprise`
  - `rate = base + score * 120`
  - `volume = base`
  - preferred voice: `voices[0]`
- `neutral`
  - default values
  - default voice

Clamping:

- rate: `120..230`
- volume: `0.5..1.0`

## Debug Output (Per Sentence)

During CLI execution, each sentence prints:

- sentence text
- detected emotion
- confidence score
- applied rate
- applied volume
- selected voice

## Model Download and Cache Behavior

- First run may take longer because the transformer model is downloaded.
- The pipeline is cached in-process to avoid repeated reloads.
- Hugging Face cache on disk is reused across runs by `transformers`.
>>>>>>> fbd8dc5 (v1)
