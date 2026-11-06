# FastSpeech 2: English and Telugu

This repository contains the complete implementation and deliverables for the FastSpeech 2 Text-to-Speech (TTS) assignment, supporting independent English (`ai4bharat/Svarah`) and Telugu (`SPRINGLab/IndicTTS_Telugu`) acoustic models with explicit phone-level duration, pitch, and energy predictors, coupled with a universal HiFi-GAN neural vocoder.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SandeshAwate6055/FastSpeech2-English-Telugu/blob/main/FastSpeech2_Colab.ipynb)

---

## 1. Project Folder Structure

The repository strictly adheres to the prescribed assignment folder structure:

```text
project/
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   ├── checkpoints/
│   │   ├── .gitkeep
│   │   └── README.md
│   ├── audio/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── hifigan/
│   ├── lexicon/
│   ├── model/
│   ├── prepare_data.py
│   ├── preprocessor/
│   ├── text/
│   ├── transformer/
│   └── utils/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   ├── checkpoints/
│   │   ├── .gitkeep
│   │   └── README.md
│   ├── audio/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── hifigan/
│   ├── lexicon/
│   ├── model/
│   ├── prepare_data.py
│   ├── preprocessor/
│   ├── text/
│   ├── transformer/
│   └── utils/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   │   ├── E1.wav
│   │   │   ├── E2a.wav
│   │   │   ├── E2b.wav
│   │   │   ├── E3.wav
│   │   │   └── E4.wav
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       │   ├── T1.wav
│       │   ├── T2a.wav
│       │   ├── T2b.wav
│       │   ├── T3.wav
│       │   └── T4.wav
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── FastSpeech2_Colab.ipynb
├── README.md
└── requirements.txt
```

---

## 2. Deliverables Summary

| Requirement | Description | Location | Status |
|---|---|---|---|
| **English Model** | Svarah dataset, ARPAbet MFA alignment, multi-speaker FastSpeech 2 | `English/` | **Complete** |
| **Telugu Model** | IndicTTS Telugu, phonemic Akshara decomposition, FastSpeech 2 | `Telugu/` | **Complete** |
| **Loss Curves** | Training and validation loss curves (mel, duration, pitch, energy) | `Results/<Lang>/{training,validation}_loss.png` | **Complete** |
| **Test Audios** | 10 synthesized 22,050 Hz mono WAVs for mandatory sentences | `Results/<Lang>/test_generated_audios/` | **Complete** |
| **Variance Plots** | 30 high-resolution duration, F0 (pitch), and energy prediction plots | `Results/<Lang>/variance_adaptor_plots/` | **Complete** |
| **Technical Report** | 20-page comprehensive report covering theory, training, and analysis | `Report/Technical_Report.pdf` | **Complete** |
| **Colab Pipeline** | Clean, runnable end-to-end pipeline notebook | `FastSpeech2_Colab.ipynb` | **Complete** |

---

## 3. Mandatory Test Sentences and Audio Results

### English (`Results/English/test_generated_audios/`)
- **E1:** *"The quick brown fox jumps over the lazy dog."* &rarr; [`E1.wav`](Results/English/test_generated_audios/E1.wav)
- **E2a:** *"The birch canoe slid on the smooth planks."* &rarr; [`E2a.wav`](Results/English/test_generated_audios/E2a.wav)
- **E2b:** *"Glue the sheet to the dark blue background."* &rarr; [`E2b.wav`](Results/English/test_generated_audios/E2b.wav)
- **E3:** *"She had a  deep, expressive  voice that carried  across the hall."* *(intentional double spaces preserved)* &rarr; [`E3.wav`](Results/English/test_generated_audios/E3.wav)
- **E4:** *"Artificial intelligence and speech synthesis are transforming modern communication systems."* &rarr; [`E4.wav`](Results/English/test_generated_audios/E4.wav)

### Telugu (`Results/Telugu/test_generated_audios/`)
- **T1:** *"త్వరగా కదిలే గోధుమ రంగు నక్క సోమరి కుక్కపై నుండి దూకుతుంది."* &rarr; [`T1.wav`](Results/Telugu/test_generated_audios/T1.wav)
- **T2a:** *"ఆమె అందమైన పాట పాడుతూ అందరినీ ఆకట్టుకుంది."* &rarr; [`T2a.wav`](Results/Telugu/test_generated_audios/T2a.wav)
- **T2b:** *"పిల్లలు మైదానంలో ఉత్సాహంగా ఫుట్‌బాల్ ఆడుతున్నారు."* &rarr; [`T2b.wav`](Results/Telugu/test_generated_audios/T2b.wav)
- **T3:** *"పుస్తకాలు  జ్ఞానాన్ని పెంచే  మంచి  స్నేహితులు."* *(intentional double spaces preserved)* &rarr; [`T3.wav`](Results/Telugu/test_generated_audios/T3.wav)
- **T4:** *"సాంకేతిక పరిజ్ఞానం మానవ జీవితాన్ని ఎంతో సులభతరం మరియు సౌకర్యవంతంగా మార్చింది."* &rarr; [`T4.wav`](Results/Telugu/test_generated_audios/T4.wav)

---

## 4. Model Checkpoint Downloads & Provenance

Checkpoints exceed GitHub's 100 MB file limit and are hosted on public Google Drive:

- **Shared Artifacts Folder:** [Google Drive Folder](https://drive.google.com/drive/folders/1WsR1NHhzpkLvk3V0g8X3Jy-xsXNlsYHf?usp=sharing)

| Model | Checkpoint Step | Speaker ID | File Size | SHA-256 Checksum | Direct Download |
|---|---|---|---|---|---|
| **English** | Step 8,000 *(best val)* | 14 (`svarah_spk_014`) | 420.5 MB | `b50f0558474eab89e95e8079df23cb3c364e339ce00fe6d88316beb4b86ff49a` | [Download 8000.pth.tar](https://drive.google.com/uc?export=download&id=1asQbbLU4y9qPAHd8Jvu2T1G-5VES6X7P) |
| **Telugu** | Step 13,000 *(best val)* | 0 (`te_female`) | 420.9 MB | `49902507629887721818ec180dac2b419609f6c7a9e601c6d767a11233d1baa1` | [Download 13000.pth.tar](https://drive.google.com/uc?export=download&id=1loNuZ9gRVLTobXHzzwZGsDe8KkB_FB-p) |

*Place downloaded files in `English/checkpoints/8000.pth.tar` and `Telugu/checkpoints/13000.pth.tar`.*

---

## 5. Quickstart & Synthesis

### Environment Setup
```bash
git clone https://github.com/SandeshAwate6055/FastSpeech2-English-Telugu.git
cd FastSpeech2-English-Telugu/project
pip install -r requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

### English Inference
```bash
cd English
# Download checkpoint if not already present:
# gdown 1asQbbLU4y9qPAHd8Jvu2T1G-5VES6X7P -O checkpoints/8000.pth.tar

python inference.py \
  --checkpoint_path checkpoints/8000.pth.tar \
  --speaker_id 14 \
  --text "The quick brown fox jumps over the lazy dog." \
  --output_path ../Results/English/test_generated_audios/E1.wav
```

### Telugu Inference
```bash
cd ../Telugu
# Download checkpoint if not already present:
# gdown 1loNuZ9gRVLTobXHzzwZGsDe8KkB_FB-p -O checkpoints/13000.pth.tar

python inference.py \
  --checkpoint_path checkpoints/13000.pth.tar \
  --speaker_id 0 \
  --text "త్వరగా కదిలే గోధుమ రంగు నక్క సోమరి కుక్కపై నుండి దూకుతుంది." \
  --output_path ../Results/Telugu/test_generated_audios/T1.wav
```

---

## 6. Training Pipeline

Both models follow deterministic data preparation with 5.0 hours training and 1.0 hour validation:
- **English:** Subsampled from `ai4bharat/Svarah` (3,359 train / 674 val utterances, 115 speakers, 0 overlap).
- **Telugu:** Subsampled from `SPRINGLab/IndicTTS_Telugu` (2,164 train / 421 val utterances, 2 speakers, 0 overlap).

To retrain from scratch:
```bash
# English
cd English
python prepare_data.py --raw_dir raw_data --train_hours 5.0 --val_hours 1.0 --seed 42
python preprocess.py --config config.yaml
python train.py --config config.yaml

# Telugu
cd ../Telugu
python prepare_data.py --raw_dir raw_data --train_hours 5.0 --val_hours 1.0 --seed 42
python preprocess.py --config config.yaml
python train.py --config config.yaml
```

---

## 7. Google Colab Pipeline

For an interactive, cloud-based execution with GPU support, open [`FastSpeech2_Colab.ipynb`](FastSpeech2_Colab.ipynb) directly in Google Colab:
- Setup and dependency installation
- Checkpoint restoration from Google Drive
- Full inference on test sentences with in-browser audio playback
- Plotting duration, pitch, and energy predictions
