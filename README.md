# Empathy Engine

Empathy Engine converts paragraph input into emotionally expressive speech,
processing one sentence at a time with transformer-based emotion detection and
audio post-processing.

We use a transformer-based emotion classification model to capture nuanced emotional context beyond simple polarity scores.

## Why Hugging Face Instead of TextBlob

The project uses the Hugging Face model
`j-hartmann/emotion-english-distilroberta-base` to detect:

- `joy`
- `sadness`
- `anger`
- `fear`
- `surprise`
- `neutral`

Compared to polarity-only methods, this improves nuanced emotional detection and
provides confidence scores that drive stronger modulation.

## Updated Pipeline

Text -> Sentence Split -> Emotion Detection -> pyttsx3 (per sentence WAV) ->
Pitch Shift Processing -> Merge Audio -> Final Output

## Features

- Sentence-wise processing using NLTK `sent_tokenize`
- Transformer emotion detection per sentence (`emotion + confidence`)
- Dynamic rate and volume modulation
- Dynamic pitch shifting per sentence (librosa)
- Single consistent TTS voice across all sentences
- Smooth pauses between sentences
- Final merged `output.wav`
- Optional Streamlit frontend

## Emotion Mapping

Base:

- `rate = 170`
- `volume = 0.9`

Rate/volume mapping:

- joy:
  - `rate = base + score * 100`
  - `volume = min(1.0, base + 0.2)`
- sadness / fear:
  - `rate = base - score * 80`
  - `volume = max(0.5, base - 0.3)`
- anger:
  - `rate = base + score * 60`
  - `volume = 1.0`
- surprise:
  - `rate = base + score * 120`
- neutral:
  - default base values

Clamping:

- rate: `120..230`
- volume: `0.5..1.0`

Pitch mapping (semitones):

- joy -> `+2`
- surprise -> `+3`
- sadness -> `-2`
- fear -> `-1`
- anger -> `+1`
- neutral -> `0`

## Installation

```bash
pip install -r requirements.txt
```

Note: first run downloads the Hugging Face model and may take longer.

## Run

Interactive CLI:

```bash
python main.py
```

CLI with argument:

```bash
python main.py --text "I got promoted. I am nervous about the new role."
```

Frontend:

```bash
streamlit run frontend.py
```

## Debug Output

Per sentence, CLI prints:

- sentence
- detected emotion
- confidence
- rate
- volume
- pitch shift applied

## File Management

- Temporary sentence files are created in `temp/` during synthesis.
- Intermediate files are cleaned up after final merge.
- Optional `--save-sentences` exports base and processed sentence WAV files.
