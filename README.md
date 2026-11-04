# FastSpeech 2 Text-to-Speech Synthesis (English & Telugu)

Implementation and evaluation of **FastSpeech 2** (*Fast and High-Quality End-to-End Text to Speech*, [Ren et al., ICLR 2021](https://arxiv.org/abs/2006.04558)) adapted independently for **English** (`ai4bharat/Svarah`) and **Telugu** (`SPRINGLab/IndicTTS_Telugu`), based on [`ming024/FastSpeech2`](https://github.com/ming024/FastSpeech2) (reference commit `d4e79eb52e8b01d24703b2dfc0385544092958f3`).

---

## 1. Deliverables & Repository Structure

```text
project/
├── English/
│   ├── train.py                     # FastSpeech 2 training with ScheduledOptim & auto-resume
│   ├── preprocess.py                # Preprocessing entry point (mel, pitch, energy, duration)
│   ├── inference.py                 # Synthesis of E1, E2a, E2b, E3, E4 + 15 variance plots
│   ├── config.yaml                  # Model & preprocessing configuration for English
│   ├── prepare_data.py              # Svarah dataset download, subsampling (5h train, 1h val)
│   ├── text/                        # CMUdict + neural g2p_en frontend + time/date expander
│   ├── lexicon/                     # ARPAbet lexicon + MFA dictionary
│   └── checkpoints/                 # Model checkpoint directory (see Section 5 for download)
├── Telugu/
│   ├── train.py                     # FastSpeech 2 training with ScheduledOptim & auto-resume
│   ├── preprocess.py                # Preprocessing entry point (mel, pitch, energy, duration)
│   ├── inference.py                 # Synthesis of T1, T2a, T2b, T3, T4 + 15 variance plots
│   ├── config.yaml                  # Model & preprocessing configuration for Telugu
│   ├── prepare_data.py              # IndicTTS download, subsampling (5h train, 1h val)
│   ├── text/                        # Telugu Unicode akshara-to-phoneme FST (64 symbols)
│   ├── lexicon/                     # Telugu phonemic lexicon
│   └── checkpoints/                 # Model checkpoint directory (see Section 5 for download)
├── Results/
│   ├── English/
│   │   ├── training_loss.png        # 6-panel training loss curve across 60,000 steps
│   │   ├── validation_loss.png      # 6-panel validation loss curve across 60,000 steps
│   │   ├── test_generated_audios/   # E1.wav, E2a.wav, E2b.wav, E3.wav, E4.wav (22.05 kHz)
│   │   └── variance_adaptor_plots/
│   │       ├── duration/            # Predicted duration plots (E1–E4)
│   │       ├── pitch/               # Predicted pitch (F0) contours (E1–E4)
│   │       └── energy/              # Predicted energy contours (E1–E4)
│   └── Telugu/
│       ├── training_loss.png        # 6-panel training loss curve across 60,000 steps
│       ├── validation_loss.png      # 6-panel validation loss curve across 60,000 steps
│       ├── test_generated_audios/   # T1.wav, T2a.wav, T2b.wav, T3.wav, T4.wav (22.05 kHz)
│       └── variance_adaptor_plots/
│           ├── duration/            # Predicted duration plots (T1–T4)
│           ├── pitch/               # Predicted pitch (F0) contours (T1–T4)
│           └── energy/              # Predicted energy contours (T1–T4)
├── Report/
│   ├── Technical_Report.pdf         # Comprehensive 6-page IEEE/ACL-style technical report
│   └── generate_report.py           # Automated PDF generation script via PyMuPDF Story
├── notebooks/
│   └── FastSpeech2_Colab.ipynb      # Clean 7-cell pure-Python end-to-end master pipeline
├── CHANGES.md                       # Comprehensive log of architectural & frontend changes
├── README.md                        # Project documentation, results & reproducibility guide
└── requirements.txt                 # Pinned dependencies
```

---

## 2. Quantitative Results & Training Milestones (60,000 Steps)

Both models were trained for **60,000 steps** on an NVIDIA GPU (batch size 16, Adam $\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$, 4,000-step linear warmup, gradient clipping 1.0). Component losses logged in `<Language>/logs/`:

| Model & Split | Step | Total Loss | Mel MAE | PostNet MAE | Pitch MSE | Energy MSE | Duration MSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **English (Train)** | 100 | 9.5892 | 3.4841 | 2.6863 | 1.1178 | 1.6051 | 0.6959 |
| **English (Train)** | 10,000 | 1.4943 | 0.6208 | 0.6206 | 0.1043 | 0.0801 | 0.0685 |
| **English (Train)** | 30,000 | 0.9928 | 0.4574 | 0.4569 | 0.0243 | 0.0299 | 0.0243 |
| **English (Train)** | **60,000** | **0.9216** | **0.4331** | **0.4326** | **0.0259** | **0.0168** | **0.0131** |
| **English (Val)** | 1,000 | 2.9111 | 0.8495 | 0.9103 | 0.4710 | 0.4233 | 0.2570 |
| **English (Val)** | 10,000 | 2.7355 | 0.7601 | 0.7599 | 0.4892 | 0.4366 | 0.2897 |
| **English (Val)** | **60,000** | **2.7353** | **0.7709** | **0.7706** | **0.4820** | **0.4299** | **0.2820** |
| **Telugu (Train)** | 100 | 10.4081 | 4.2251 | 3.5288 | 1.0831 | 1.1258 | 0.4454 |
| **Telugu (Train)** | 10,000 | 0.9728 | 0.4290 | 0.4289 | 0.0501 | 0.0417 | 0.0232 |
| **Telugu (Train)** | 30,000 | 0.7656 | 0.3580 | 0.3576 | 0.0223 | 0.0188 | 0.0089 |
| **Telugu (Train)** | **60,000** | **0.6627** | **0.3153** | **0.3150** | **0.0152** | **0.0126** | **0.0045** |
| **Telugu (Val)** | 1,000 | 2.4060 | 0.7439 | 0.8161 | 0.5500 | 0.2292 | 0.0667 |
| **Telugu (Val)** | 10,000 | 1.9848 | 0.5928 | 0.5926 | 0.5269 | 0.2139 | 0.0585 |
| **Telugu (Val)** | **60,000** | **1.9794** | **0.5968** | **0.5966** | **0.5168** | **0.2089** | **0.0603** |

---

## 3. Test Sentences & Variance Adaptor Predictions

All 10 test sentences were synthesized keeping input text, intentional double spaces (`E3`, `T3`), and Telugu vowel signs intact:

| ID | Test Sentence | Audio | Tokens | Frames | Audio Dur | F0 Mean [Min, Max] Hz | Energy Mean [Min, Max] | Key Linguistic / Prosodic Phenomenon |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | [Audio](Results/English/test_generated_audios/E1.wav) | 85 | 501 | 5.82 s | 134.7 [93.1, 174.4] | 77.9 [13.9, 149.0] | 28-letter OOV compound via neural `g2p_en` |
| **E2a** | *The orchestra's conduct during the ceremony was praised by the audience.* | [Audio](Results/English/test_generated_audios/E2a.wav) | 50 | 283 | 3.29 s | 128.4 [109.1, 189.2] | 68.4 [7.2, 138.1] | Homograph *conduct* (noun context after possessive *orchestra's*) |
| **E2b** | *The researchers will conduct an experiment to evaluate the proposed method.* | [Audio](Results/English/test_generated_audios/E2b.wav) | 56 | 337 | 3.91 s | 134.8 [112.9, 178.3] | 67.3 [9.4, 121.1] | Homograph *conduct* (verb context after modal *will*) |
| **E3** | *The rhyth  m of the orchestra changed unexpectedly.* | [Audio](Results/English/test_generated_audios/E3.wav) | 37 | 217 | 2.52 s | 122.9 [108.7, 143.4] | 52.7 [6.4, 113.0] | Double-space split (`rhyth  m` &rarr; `rhyth` + `m`) |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | [Audio](Results/English/test_generated_audios/E4.wav) | 80 | 562 | 6.52 s | 135.1 [114.8, 196.4] | 68.4 [5.2, 139.5] | Number/time/date expansion, comma pause (`sp`=40f), Telugu word *andaroo* |
| **T1** | న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. | [Audio](Results/Telugu/test_generated_audios/T1.wav) | 66 | 451 | 5.24 s | 209.3 [155.4, 321.9] | 58.8 [8.2, 140.6] | 10-akshara scientific loanword & diphthong *ఔ* |
| **T2a** | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. | [Audio](Results/Telugu/test_generated_audios/T2a.wav) | 31 | 222 | 2.58 s | 221.3 [153.5, 329.2] | 59.7 [4.1, 130.2] | Homograph *పట్టు* (grip/hold) with geminate retroflex *ట్ట* (3f+9f) |
| **T2b** | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. | [Audio](Results/Telugu/test_generated_audios/T2b.wav) | 37 | 236 | 2.74 s | 213.0 [158.5, 307.5] | 55.0 [5.7, 133.1] | Homograph *పట్టు* (silk) in pre-nominal modifier context *పట్టు చీరను* |
| **T3** | పులి అడ  విలో నిశ్శబ్దంగా నడిచింది. | [Audio](Results/Telugu/test_generated_audios/T3.wav) | 31 | 201 | 2.33 s | 214.3 [155.9, 334.3] | 63.4 [5.6, 115.5] | Double-space split (*అడ  విలో*) & geminate sibilant *నిశ్శబ్దంగా* (11f+10f) |
| **T4** | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. | [Audio](Results/Telugu/test_generated_audios/T4.wav) | 120 | 775 | 9.00 s | 213.9 [160.1, 289.3] | 56.9 [1.4, 114.5] | Telugu year/ordinal/time expansion & 52-frame (0.60 s) comma pause |

---

## 4. Architectural Modifications & MFA 2.x Fix

### Critical MFA 2.2.17 Interior Pause Alignment Bug & Fix
In `ming024/FastSpeech2` (developed for MFA 1.x), `tgt.io.read_textgrid()` was called with `include_empty_intervals=False`. 
- In MFA 2.2.17, inter-word interior pauses are labeled as empty intervals (`""`).
- Setting `include_empty_intervals=False` silently deleted every inter-word pause while still slicing the audio Mel spectrogram from utterance start to end, shifting subsequent words by 0.5–2.0 seconds onto mismatched frames.
- **Fix:** Set `include_empty_intervals=True` in `preprocessor/preprocessor.py`, map empty intervals and timestamp gaps to explicit `"sp"` tokens, enforce a strict pre-slice frame invariant ($|\text{raw\_frames} - \sum d_i| \le 2$), and normalize pitch/energy strictly on `train.txt` bounds.

For a full breakdown of all modifications, see [`CHANGES.md`](CHANGES.md) and [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf).

---

## 5. Model Checkpoints & Pretrained Weights

Due to GitHub's 100 MB file size limit, full training checkpoints (456 MB each) are preserved on Google Drive and can be downloaded from:

- **English FastSpeech 2 Checkpoints (Step 60,000):**
  - Path on Drive: `/content/drive/MyDrive/FastSpeech2_CleanRun/English/checkpoints/60000.pth.tar`
  - Backup Archive: `/content/drive/MyDrive/FastSpeech2_CleanRun/English/English_preprocessed.tar.gz`
- **Telugu FastSpeech 2 Checkpoints (Step 60,000):**
  - Path on Drive: `/content/drive/MyDrive/FastSpeech2_CleanRun/Telugu/checkpoints/60000.pth.tar`
  - Backup Archive: `/content/drive/MyDrive/FastSpeech2_CleanRun/Telugu/Telugu_preprocessed.tar.gz`
- **Pretrained HiFi-GAN Universal Vocoder:**
  - Placed in `English/hifigan/generator_universal.pth.tar` and `Telugu/hifigan/generator_universal.pth.tar` (downloaded from [`ming024/FastSpeech2/hifigan`](https://github.com/ming024/FastSpeech2/tree/master/hifigan)).

To place checkpoints locally for standalone inference:
```bash
# Place checkpoints in:
English/checkpoints/60000.pth.tar
Telugu/checkpoints/60000.pth.tar
```

---

## 6. Reproducibility & End-to-End Execution Guide

### Option A: One-Click Execution via Google Colab
Open [`notebooks/FastSpeech2_Colab.ipynb`](notebooks/FastSpeech2_Colab.ipynb) in Google Colab:
- Run **Cells 1–7** sequentially.
- Automatically handles dependencies, MFA alignment, feature extraction, 60,000-step training with Google Drive auto-resume, inference for all 10 sentences, loss plotting, 30 variance adaptor plots, and ZIP packaging.

### Option B: Local CLI Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. English pipeline
cd English
python prepare_data.py --train_hours 5.0 --val_hours 1.0 --seed 42
mfa align raw_data/mfa_corpus lexicon/svarah_mfa_dict.txt english_us_arpa preprocessed_data/TextGrid --clean
python preprocess.py config.yaml
python train.py --config config.yaml --total_step 60000
python inference.py --config config.yaml --restore_step 60000 --result_path ../Results/English

# 3. Telugu pipeline
cd ../Telugu
python prepare_data.py --train_hours 5.0 --val_hours 1.0 --seed 42
mfa train raw_data/mfa_corpus lexicon/telugu_mfa_dict.txt preprocessed_data/mfa_model.zip --output_directory preprocessed_data/TextGrid --clean
python preprocess.py config.yaml
python train.py --config config.yaml --total_step 60000
python inference.py --config config.yaml --restore_step 60000 --result_path ../Results/Telugu

# 4. Generate Technical Report PDF
cd ..
python Report/generate_report.py
```
