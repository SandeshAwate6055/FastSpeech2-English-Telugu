# Assignment Completion and Submission Verification Status

This document records the strict compliance audit, model artifact provenance, and delivery verification for the FastSpeech 2 English and Telugu Text-to-Speech assignment.

---

## 1. Compliance Audit Summary

The submission package has been strictly validated using `tools/audit_submission.py --strict`.

- **Audit Status:** Passed (`submission_ready: true`, `artifacts_valid: true`)
- **Errors:** `0`
- **Submission Blockers:** `0`
- **Timestamp:** 2026-11-04
- **Audit Report:** `Results/submission_audit.json`

---

## 2. Requirement-by-Requirement Evidence

| Assignment Requirement | Required Deliverable | Repository Evidence | Verification Status |
|---|---|---|---|
| **Dual-Language Acoustic Models** | Separate implementations for English (`Svarah`) and Telugu (`IndicTTS_Telugu`) with distinct frontends | `English/` and `Telugu/` directories with dedicated `train.py`, `preprocess.py`, `inference.py`, `config.yaml`, text cleaners, and symbol inventories; documented in `CHANGES.md` | **Verified** |
| **Data Partitioning & Subsampling** | ~5.0 hours training, ~1.0 hour validation; speaker-stratified; strictly disjoint splits ($0$ overlap) | Subsampling scripts in `English/preprocess.py` and `Telugu/preprocess.py`; metadata manifests in `preprocessed_data/`: <br>• English: 3,359 train clips (5.000h), 674 val clips (1.001h), 115 speakers, 0 overlap <br>• Telugu: 2,164 train clips (5.000h), 421 val clips (1.000h), 2 speakers, 0 overlap | **Verified** |
| **Loss Curves & Metrics** | Separate training and validation loss curves showing total, mel, duration, pitch, and energy losses | Preserved source CSV histories (600 train / 60 val entries through 60,000 steps) in `Results/<Language>/metrics/`; plotted curves in `Results/<Language>/{training,validation}_loss.png` | **Verified** |
| **Mandatory Test Audios** | Synthesized speech for all 5 test inputs per language (10 WAV files total) | 10 WAV files in `Results/<Language>/test_generated_audios/` (`E1`, `E2a`, `E2b`, `E3`, `E4`, `T1`, `T2a`, `T2b`, `T3`, `T4`) at mono 22,050 Hz 16-bit PCM; exact assignment texts preserved | **Verified** |
| **Variance Adaptor Visualizations** | Duration, pitch (F0), and energy prediction plots for every test sentence (30 plots total) | 30 high-resolution plots under `Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/` with physical unit scales and alignment markers; full predictions in `variance_predictions_summary.json` | **Verified** |
| **Trained Model Checkpoints** | Public evaluator access to the exact checkpoints and metadata used to generate test audios | Checkpoints and preprocessed data archives hosted on public Google Drive folder; direct download URLs, file sizes, and SHA-256 hashes recorded in `model_artifacts.json` | **Verified** |
| **Comprehensive Technical Report** | Structured PDF covering architecture, preprocessing, experiments, loss analysis, per-sentence breakdowns, and limitations | 20-page technical report at `Report/Technical_Report.pdf` with complete system diagrams, mathematical formulations, loss curves, spectrograms, and page-by-page sentence analyses | **Verified** |

---

## 3. Model Artifact Provenance & Google Drive Links

All trained checkpoints, logs, and preprocessed datasets are publicly accessible without authentication in the shared Google Drive folder:
- **Shared Folder URL:** [Google Drive Artifacts Folder](https://drive.google.com/drive/folders/1WsR1NHhzpkLvk3V0g8X3Jy-xsXNlsYHf?usp=sharing)

### Verified Artifact Inventory

| Artifact Description | Language | Step / Speaker | Size | SHA-256 Hash | Direct Download Link |
|---|---|---|---|---|---|
| **Acoustic Model Checkpoint** | English | Step 8,000 / Spk 14 | 420.5 MB | `b50f0558474eab89e95e8079df23cb3c364e339ce00fe6d88316beb4b86ff49a` | [Download 8000.pth.tar](https://drive.google.com/uc?export=download&id=1asQbbLU4y9qPAHd8Jvu2T1G-5VES6X7P) |
| **Preprocessed Data Archive** | English | 5h train + 1h val | 527.9 MB | `dee67de261922b2c8de338c58e39065ba849ccef0463ce7171e892932732dd84` | [Download English_preprocessed.tar.gz](https://drive.google.com/uc?export=download&id=1Jed_UntmRMt29ESQY5OOEENB3pAN1Q0s) |
| **Acoustic Model Checkpoint** | Telugu | Step 13,000 / Spk 0 | 420.9 MB | `49902507629887721818ec180dac2b419609f6c7a9e601c6d767a11233d1baa1` | [Download 13000.pth.tar](https://drive.google.com/uc?export=download&id=1loNuZ9gRVLTobXHzzwZGsDe8KkB_FB-p) |
| **Preprocessed Data Archive** | Telugu | 5h train + 1h val | 528.1 MB | `f2a7a6cf2baf6d917daf33be1cb85def63103eed321af33beed8a7c9d45c1ba1` | [Download Telugu_preprocessed.tar.gz](https://drive.google.com/uc?export=download&id=1rYjLsdr_YdcSUS23X1DwrqXEOSjwVr9s) |

*Note: The complete 60,000-step training endpoints (`60000.pth.tar`) and intermediate checkpoints (steps 5,000 through 55,000) are also preserved in the shared folder under their respective subdirectories.*

---

## 4. Evaluator Quickstart & Reproducibility

### Checkpoint Restoration & Strict Verification
```bash
# Clone the repository
git clone https://github.com/SandeshAwate6055/FastSpeech2-English-Telugu.git
cd FastSpeech2-English-Telugu

# Run the strict compliance audit
python tools/audit_submission.py --strict
```

### Synthesizing Test Sentences
```bash
# English Synthesis (Step 8,000, Speaker 14)
python English/inference.py \
  --checkpoint_path English/checkpoints/8000.pth.tar \
  --speaker_id 14 \
  --text "The quick brown fox jumps over the lazy dog." \
  --output_path Results/English/test_generated_audios/E1.wav

# Telugu Synthesis (Step 13,000, Speaker 0)
python Telugu/inference.py \
  --checkpoint_path Telugu/checkpoints/13000.pth.tar \
  --speaker_id 0 \
  --text "త్వరగా కదిలే గోధుమ రంగు నక్క సోమరి కుక్కపై నుండి దూకుతుంది." \
  --output_path Results/Telugu/test_generated_audios/T1.wav
```

---

## 5. Perceptual Observations and Known Limitations

1. **Checkpoint Selection Criteria:** The submitted audio files were generated using the validation-loss minima checkpoints (English Step 8,000, Telugu Step 13,000) rather than the 60,000-step overfitted endpoints.
2. **Vocoder Domain Transfer:** HiFi-GAN universal vocoder (`generator_universal.pth.tar`) was pre-trained on LJSpeech (22.05 kHz). While robust, fine-tuning HiFi-GAN on Indian English accents and Telugu phonetics would further eliminate mild robotic timbre in unvoiced fricatives.
3. **Data Distribution:** English training data from Svarah contains Indian English speaker diversity (115 demographic speakers), leading to natural prosodic variation across speaker IDs. Telugu from IndicTTS contains 2 clean studio speakers (female and male), yielding highly consistent pitch tracks.
