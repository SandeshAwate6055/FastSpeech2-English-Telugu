# FastSpeech 2 Text-to-Speech Synthesis (English & Telugu)

Implementation and evaluation of **FastSpeech 2** (*Fast and High-Quality End-to-End Text to Speech*, [Ren et al., 2020](https://arxiv.org/abs/2006.04558)) adapted independently for **English** (`ai4bharat/Svarah`) and **Telugu** (`SPRINGLab/IndicTTS_Telugu`), based on [`ming024/FastSpeech2`](https://github.com/ming024/FastSpeech2) (reference commit `d4e79eb52e8b01d24703b2dfc0385544092958f3`).

---

## 1. Repository Structure

```text
project/
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   ├── prepare_data.py
│   └── checkpoints/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   ├── prepare_data.py
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── notebooks/
│   └── FastSpeech2_Colab.ipynb
├── CHANGES.md
├── README.md
└── requirements.txt
```

---

## 2. Environment Setup

### Local / Linux / Google Colab Setup
1. Install Python dependencies (Python 3.10 recommended):
   ```bash
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
   ```
2. Install **Montreal Forced Aligner (MFA)** (in a dedicated conda/mamba environment for forced alignment):
   ```bash
   conda create -n mfa -c conda-forge montreal-forced-aligner -y
   ```
3. Ensure pretrained **Universal HiFi-GAN** vocoder weights (`generator_universal.pth.tar`) from [`ming024/FastSpeech2/hifigan`](https://github.com/ming024/FastSpeech2/tree/master/hifigan) are placed in `English/hifigan/` and `Telugu/hifigan/`.

---

## 3. Dataset Preparation & Subsampling (~5h Train / ~1h Val)

Both datasets use deterministic (`seed=42`), **speaker-stratified, utterance-disjoint** splits (`1.0s–20.0s` duration range, SHA-1 audio waveform deduplication, and split-specific reserve candidates) so that every validation speaker has trained speaker embeddings and zero overlap exists between training and validation.

### English (`ai4bharat/Svarah`)
Requires a Hugging Face access token (`HF_TOKEN`) after accepting terms at [`ai4bharat/Svarah`](https://huggingface.co/datasets/ai4bharat/Svarah):
```bash
cd English
export HF_TOKEN="your_hf_token_here"
python prepare_data.py --train_hours 5.0 --val_hours 1.0
```
- Groups utterances into `115` demographic speaker groups (`multi_speaker: True`).
- Resamples audio to `22,050 Hz` mono 16-bit PCM `.wav` and writes normalized `.lab` transcripts.
- Generates a zero-OOV ARPAbet dictionary (`lexicon/english_svarah_mfa.dict`) combining `librispeech-lexicon.txt` and `g2p_en`.

### Telugu (`SPRINGLab/IndicTTS_Telugu`)
Uses both studio voices (`te_female` and `te_male`, `multi_speaker: True`) from [`SPRINGLab/IndicTTS_Telugu`](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu):
```bash
cd Telugu
python prepare_data.py --train_hours 5.0 --val_hours 1.0
```
- Resamples audio from `48,000 Hz` to `22,050 Hz` mono 16-bit PCM `.wav` and writes normalized UTF-8 `.lab` transcripts.
- Builds the phonemic Telugu lexicon (`lexicon/telugu-lexicon.txt`) modeling inherent vowel `'అ'`, dependent vowel signs (*Gunintalu* $\rightarrow$ canonical *Achulu*), *Virama/Halant* (`్` suppressing inherent vowel in consonant clusters and geminates), *Anusvara* (`ం`), and *Visarga* (`ః`).

---

## 4. Forced Alignment (MFA) & Feature Extraction (`preprocess.py`)

### English Forced Alignment & Preprocessing
```bash
# 1. Align with MFA (english_us_arpa acoustic model + zero-OOV ARPAbet dictionary)
mfa model download acoustic english_us_arpa
mfa align English/raw_data English/lexicon/english_svarah_mfa.dict english_us_arpa English/preprocessed_data/TextGrid --clean

# 2. Extract Mel-spectrograms, Pitch (F0), Energy, and Durations (train-only normalization)
cd English
python preprocess.py --config config.yaml
```

### Telugu Forced Alignment & Preprocessing
To avoid validation leakage, the Telugu MFA acoustic model is trained on the **training split only**, then used to align both splits:
```bash
# 1. Train MFA acoustic model on Telugu training split & align into TextGrid/
mfa train Telugu/raw_data Telugu/lexicon/telugu-lexicon.txt Telugu/te_mfa_acoustic.zip --output_directory Telugu/preprocessed_data/TextGrid --clean

# 2. Extract Mel-spectrograms, Pitch (F0), Energy, and Durations (train-only normalization)
cd Telugu
python preprocess.py --config config.yaml
```

---

## 5. Model Training (`train.py`)

Train each language model independently:

```bash
# Train English FastSpeech 2
cd English
python train.py --config config.yaml

# Resume from checkpoint if interrupted (e.g. step 25000)
python train.py --config config.yaml --restore_step 25000
```

```bash
# Train Telugu FastSpeech 2
cd Telugu
python train.py --config config.yaml
```
- Saves periodic checkpoints (`5000.pth.tar`, ..., `60000.pth.tar`) and best validation checkpoint (`best_val.pth.tar`) in `<Language>/checkpoints/`.
- Logs training and validation component losses to `<Language>/logs/train_losses.csv` and `val_losses.csv`.
- Automatically plots 6-panel loss curves (`training_loss.png` and `validation_loss.png`) in `Results/<Language>/`.

---

## 6. Inference & Variance Adaptor Plots (`inference.py`)

Generate speech samples (`.wav`) and Variance Adaptor plots (Duration, Pitch $F_0$, Energy) for all test sentences:

```bash
# English Test Sentences (E1, E2a, E2b, E3, E4)
cd English
python inference.py --config config.yaml --speaker_id 0

# Telugu Test Sentences (T1, T2a, T2b, T3, T4)
cd Telugu
python inference.py --config config.yaml --speaker_id 0
```

### Outputs Generated Automatically
- **Synthesized Audio (`.wav`):** `Results/<Language>/test_generated_audios/<ID>.wav`
- **Duration Plots:** `Results/<Language>/variance_adaptor_plots/duration/<ID>_duration.png`
- **Pitch ($F_0$) Plots:** `Results/<Language>/variance_adaptor_plots/pitch/<ID>_pitch.png`
- **Energy Plots:** `Results/<Language>/variance_adaptor_plots/energy/<ID>_energy.png`
- **Numerical Summary JSON:** `Results/<Language>/variance_predictions_summary.json`

---

## 7. Google Colab Pro Execution

Open [`notebooks/FastSpeech2_Colab.ipynb`](notebooks/FastSpeech2_Colab.ipynb) in Google Colab Pro:
1. Run **Sections 1–4** on a **CPU (High-RAM)** runtime to prepare data, run MFA forced alignment, extract features, and back up `preprocessed_data/` to Google Drive without spending GPU compute units.
2. Switch to an **L4 GPU** runtime and run **Sections 5–7** to verify HiFi-GAN mel reconstruction, train both models with automatic Google Drive checkpoint persistence, and generate all 10 test `.wav` files and 30 variance adaptor plots.

For a detailed list of all architectural and frontend modifications made to `ming024/FastSpeech2`, see [`CHANGES.md`](CHANGES.md).
