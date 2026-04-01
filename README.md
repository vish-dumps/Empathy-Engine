# 🧠 Empathy Engine: Emotion-Aware Speech Synthesis

## 🚀 Project Description

The Empathy Engine is a system that converts text into emotionally expressive speech. Unlike traditional text-to-speech systems that sound flat and robotic, this project focuses on making the output feel more human by adapting the voice based on the emotion of the text.

The system processes input text at a sentence level, detects the emotion of each sentence using a transformer-based model, and then generates speech with natural variations in tone, pacing, and delivery using neural TTS (Coqui TTS).

---

## 🎯 Key Features

* Sentence-level emotion detection
* Transformer-based emotion classification (Hugging Face)
* Neural text-to-speech using Coqui TTS
* Dynamic emotional variation in speech output
* Smooth transitions between sentences
* Single consistent voice for natural flow

---

## 🏗️ Tech Stack

* **Python**
* **Hugging Face Transformers** (emotion detection)
* **Coqui TTS (xtts_v2)** (speech synthesis)
* **NLTK** (sentence tokenization)
* **Flask** (web interface)
* **Pydub** (audio merging)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/vish-dumps/Empathy-Engine.git
cd Empathy-Engine
```

---

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
venv\Scripts\activate.bat   # Windows
# source venv/bin/activate   # Mac/Linux
```

---

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

### 4. Download NLTK Data

```bash
python -m nltk.downloader punkt
```

---

### 5. Run the Application

#### Option A: CLI Mode

```bash
python main.py
```

#### Option B: With Text Input

```bash
python main.py --text "Your input text here"
```

#### Option C: Web Interface (if Flask implemented)

```bash
python -m streamlit run frontend.py
```

Then open:

```
http://localhost:8501
```

---

## ▶️ How It Works

### Pipeline:

```
Text Input
   ↓
Sentence Segmentation
   ↓
Emotion Detection (Transformer Model)
   ↓
Emotion Mapping
   ↓
Speech Generation (Coqui TTS)
   ↓
Audio Merging
   ↓
Final Output (output.wav)
```

---

## 🧠 Design Choices

### 1. Sentence-Level Processing

Instead of analyzing the entire paragraph at once, the text is split into sentences.
This allows the system to capture mixed emotions within the same input and generate more realistic speech.

---

### 2. Emotion Detection

We use a Hugging Face transformer model:

`j-hartmann/emotion-english-distilroberta-base`

This model provides fine-grained emotion labels:

* joy
* sadness
* fear
* anger
* surprise
* neutral

This was chosen over basic sentiment tools because it captures context much better.

---

### 3. Emotion → Voice Mapping

Initially, emotion was mapped manually using:

* speech rate
* volume
* pitch (via signal processing)

However, this approach was limited and often produced unnatural results.

The final system uses **Coqui TTS**, where emotional expression is handled implicitly by the neural model.

That said, emotion still influences:

* **Pause duration**

  * joy / surprise → shorter pauses
  * sadness / fear → longer pauses

* **Speech flow**

  * smoother transitions between emotional shifts

This combination helps maintain natural delivery without forcing artificial parameter changes.

---

### 4. Why Coqui TTS?

Traditional TTS (like pyttsx3):

* lacks pitch control
* sounds robotic
* requires manual tuning

Coqui TTS:

* generates natural prosody
* includes pitch variation automatically
* produces more human-like speech

This significantly improved the realism of the output.

---

### 5. Voice Consistency

The system uses a **single consistent voice** across all sentences.

Using multiple voices for different emotions initially seemed like a good idea, but it reduced realism.
A consistent voice with varying tone feels much more natural.

---

## 🎧 Output

* Final output is saved as:
  `output.wav`

* The audio contains:

  * sentence-level emotional variation
  * smooth transitions
  * natural speech flow

---

## ⚠️ Challenges & Solutions

| Challenge                    | Solution                                 |
| ---------------------------- | ---------------------------------------- |
| Poor emotion detection       | Switched to transformer model            |
| No pitch control in pyttsx3  | Replaced with Coqui TTS                  |
| Robotic speech output        | Used neural TTS                          |
| Abrupt emotional transitions | Added pauses + sentence-level processing |
| Voice inconsistency          | Used single voice                        |

---

## 🚀 Future Improvements

* Real-time streaming speech
* User-selectable voice styles
* Better control over emotion intensity
* Multilingual support

---

## 📌 Conclusion

This project explores how combining NLP and neural speech synthesis can make AI communication more natural and engaging.

By moving from rule-based voice modulation to neural TTS, the system achieves a much more realistic and emotionally expressive output.

---

## 🙌 Acknowledgements

* Hugging Face for transformer models
* Coqui TTS for open-source neural speech synthesis
* NLTK for text processing
