<a id="top"></a>

# FastSpeech 2 · English & Telugu

Two independent **FastSpeech 2** acoustic models generate English and Telugu mel spectrograms; a pretrained **universal HiFi-GAN** produces speech. The project adapts `ming024/FastSpeech2` for **Svarah English** and **IndicTTS Telugu**, following [FastSpeech2_Assignment.pdf](FastSpeech2_Assignment.pdf).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SandeshAwate6055/FastSpeech2-English-Telugu/blob/main/FastSpeech2_Colab.ipynb)

| Important links | Open |
|---|---|
| **Colab pipeline** | [Run in Colab](https://colab.research.google.com/github/SandeshAwate6055/FastSpeech2-English-Telugu/blob/main/FastSpeech2_Colab.ipynb) · [Notebook source](FastSpeech2_Colab.ipynb) |
| **Main Drive: checkpoints & results** | [Shared artifact folder](https://drive.google.com/drive/folders/1WsR1NHhzpkLvk3V0g8X3Jy-xsXNlsYHf?usp=sharing) |
| **Technical report** | [20-page PDF](project/Report/Technical_Report.pdf) |
| **All ten WAVs** | [English audio](project/Results/English/test_generated_audios) · [Telugu audio](project/Results/Telugu/test_generated_audios) · [Individual samples](#audio) |
| **Results** | [English](project/Results/English) · [Telugu](project/Results/Telugu) |

## Navigate this README

> **Quick index: select a section to jump directly to it.**

1. **[Training graphs and results](#training-results)**
2. **[System architecture](#architecture)**
3. **[Test sentences, audio and variance plots](#audio)**
4. **[Datasets and training setup](#datasets)**
5. **[Checkpoint downloads](#checkpoints)**
6. **[Reproducibility and commands](#reproducibility)**
7. **[Technical report guide](#report)**
8. **[Code and references](#files)**
9. **[Assignment requirements and evidence](#assignment-map)**

<a id="training-results"></a>
## 1. Training graphs and results

**Submitted artifacts:** 10 WAVs, 30 variance plots and 4 loss figures. Both histories reach 60,000 steps; synthesis uses the best logged validation checkpoint for each language.

| English training | English validation |
|---|---|
| ![English training loss](project/Results/English/training_loss.png) | ![English validation loss](project/Results/English/validation_loss.png) |

| Telugu training | Telugu validation |
|---|---|
| ![Telugu training loss](project/Results/Telugu/training_loss.png) | ![Telugu validation loss](project/Results/Telugu/validation_loss.png) |

| Metric | English | Telugu |
|---|---:|---:|
| Selected checkpoint | **8,000** | **13,000** |
| Minimum validation total | **2.702820** | **1.965954** |
| Final validation total (60k) | 2.735320 | 1.979367 |
| Final sampled training total | 0.921556 | 0.662709 |
| Loss CSVs | [Train](project/Results/English/metrics/train_losses.csv) / [Validation](project/Results/English/metrics/val_losses.csv) | [Train](project/Results/Telugu/metrics/train_losses.csv) / [Validation](project/Results/Telugu/metrics/val_losses.csv) |

Validation improves early, then plateaus while training loss falls, consistent with overfitting. Lower loss across different corpora does not establish better speech quality. Detailed convergence analysis is in [report pages 5–8](project/Report/Technical_Report.pdf#page=5).

### Reading the losses

| Loss term | What it measures |
|---|---|
| Mel / PostNet | L1 error before and after residual mel refinement |
| Duration | MSE against `log(1 + reference frames)` for each phone |
| Pitch / energy | MSE against the normalized phone-level targets |
| Total | Sum of both mel losses and the three variance-prediction losses |

Training CSV rows are sampled minibatches; validation values are aggregates. The histories contain **600 training records and 60 validation records per language**. Checkpoint selection follows the validation total, rather than the lowest training loss or the latest saved step. Listening to held-out free-running speech remains necessary to assess the selected weights.

<a id="architecture"></a>
## 2. System architecture

English and Telugu use the same topology with **separate weights, phone inventories, speaker maps and statistics**. English uses lexicon/G2P phones; Telugu uses Unicode-based phone decomposition.

### Original paper architecture: source reference

The paper reference is **FastSpeech 2**, the mel-spectrogram model, rather than the direct-waveform FastSpeech 2s variant. Its original drawing is [Figure 1, page 3](https://arxiv.org/pdf/2006.04558v8#page=3); architectural details are in [Section 2.3, page 4](https://arxiv.org/pdf/2006.04558v8#page=4) and [Appendices A–C, pages 13–14](https://arxiv.org/pdf/2006.04558v8#page=13). Neither attached illustration is that original figure.

### Provided Image 1: conceptual overview, not the original paper figure

![FastSpeech 2 architecture overview](assets/fastspeech2-architecture.png)

[Open the full-resolution architecture image](assets/fastspeech2-architecture.png).

The supplied image illustrates the main components. In this repository, **multi-speaker conditioning is enabled for both languages** and **pitch/energy operate at phone level**. In the code, energy consumes pitch-conditioned hidden states, and PostNet refinement is added to the original mel projection as a residual.

### Verification of the supplied diagrams

**Image 1 caveat:** its six-block decoder, PostNet and HiFi-GAN are implementation-style choices, not the paper's stated configuration. It also omits CWT pitch modeling. Its direct pitch-predictor-to-energy-predictor arrow should not be read as the project's actual hidden-state connection.

### Provided Image 2: English–Telugu implementation overview

![English–Telugu FastSpeech 2 implementation overview](assets/fastspeech2-english-telugu-architecture.png)

[Open Image 2 at full resolution](assets/fastspeech2-english-telugu-architecture.png).

**Image 2 verification:** the encoder/decoder dimensions, enabled speaker conditioning, conditioning order, residual PostNet and universal HiFi-GAN match the checked-in configurations and forward pass. It remains a simplified schematic: the hidden-state bypasses into the two embedding additions are not drawn explicitly. Energy receives `encoder_states + speaker_embedding + pitch_embedding`, not the pitch predictor's scalar output. The [variance-adaptor forward pass](project/English/model/modules.py) defines these state paths.

### Verified technical comparison

| Component | Original paper (v8) | English–Telugu project |
|---|---|---|
| **Encoder / decoder** | 4 / 4 FFT blocks; hidden 256; 2 heads; dropout 0.1 | 4 / 6 blocks; same dimensions; dropout 0.2 |
| **Pitch and variance adaptor** | Frame-level CWT/iCWT; 256 log-spaced F0 bins | Direct normalized phone F0; 256 linear bins; conditioning before length expansion |
| **Energy conditioning** | Frame-level STFT L2 magnitude; 256 uniform bins | Normalized phone-average energy; 256 linear bins; pitch-conditioned hidden states |
| **Mel refinement / vocoder** | No separate PostNet specified; pretrained Parallel WaveGAN | Residual PostNet; separate pretrained universal HiFi-GAN |
| **Languages / speakers** | English LJSpeech experiment | Separate Svarah/IndicTTS models, language-specific frontends and learned speaker embeddings |

**Verification sources:** [English config](project/English/config.yaml), [Telugu config](project/Telugu/config.yaml), [Transformer](project/English/transformer/Models.py), [variance adaptor](project/English/model/modules.py), [residual PostNet forward pass](project/English/model/fastspeech2.py), [vocoder loader](project/English/utils/model.py), [English frontend](project/English/inference.py), and [Telugu frontend](project/Telugu/text/cleaners.py).

**Why these adaptations:** separate acoustic models and language-specific frontends meet the assignment's data/language requirements; speaker embeddings condition the multi-voice data; normalized phone-level targets support the saved phone-aligned duration/F0/energy analysis. The six-block decoder, phone-level representation, residual PostNet and HiFi-GAN support largely come from the [selected reference configuration](https://github.com/ming024/FastSpeech2/blob/d4e79eb52e8b01d24703b2dfc0385544092958f3/config/LJSpeech/model.yaml), [feature settings](https://github.com/ming024/FastSpeech2/blob/d4e79eb52e8b01d24703b2dfc0385544092958f3/config/LJSpeech/preprocess.yaml) and [model](https://github.com/ming024/FastSpeech2/blob/d4e79eb52e8b01d24703b2dfc0385544092958f3/model/fastspeech2.py). These are inherited implementation choices; the repository does not establish a separate ablation-based motivation or performance benefit for each difference from the paper.

**Training** conditions the model on reference duration, pitch and energy; **inference** uses their predictions. Energy receives pitch-conditioned states, and both embeddings are added before duration expansion. [Model](project/English/model/fastspeech2.py) · [Variance adaptor](project/English/model/modules.py) · [Loss function](project/English/model/loss.py).

### Components and behavior

| Component | Role in this implementation |
|---|---|
| FFT encoder | Converts language-specific phone IDs into contextual hidden states |
| Speaker embedding | Adds the selected speaker/profile representation to encoder states |
| Duration predictor | Predicts log-duration; integer frame counts control length expansion |
| Pitch predictor and embedding | Add phone-level F0 information to the hidden states |
| Energy predictor and embedding | Add energy information after pitch conditioning |
| Length regulator | Repeats each phone state according to its duration |
| FFT decoder and PostNet | Produce and refine the 80-channel mel spectrogram |
| Universal HiFi-GAN | Converts refined mel features into the final PCM waveform |

At default controls, the predicted duration and output length are:

```text
duration_i = max(round(exp(log_duration_i) - 1), 0)
mel_frames = sum(duration_i)
audio_seconds = mel_frames * 256 / 22050
refined_mel = projected_mel + PostNet(projected_mel)
```

A zero-duration phone contributes no repeated frames. Training uses reference durations and reference pitch/energy while still learning all three prediction heads; inference supplies their predictions without reference audio.

### Training objective

```text
total = L1(mel, target)
      + L1(postnet_mel, target)
      + MSE(log_duration, log(1 + reference_duration))
      + MSE(normalized_pitch, target_pitch)
      + MSE(normalized_energy, target_energy)
```

Padding masks exclude padded phones and frames. The two language models optimize this objective independently; a low reference-conditioned reconstruction loss does not guarantee intelligible free-running speech.

<a id="implementation"></a>
**Main adaptations:** separate corpus loaders/configurations, English lexicon plus `g2p_en` fallback, Telugu vowel/virama-aware phones and numeric expansion, speaker conditioning, alignment checks, training-only feature statistics, explicit checkpoint loading and prediction exports. This implementation uses phone-level F0/energy, a six-block decoder and HiFi-GAN; [report pages 2–3](project/Report/Technical_Report.pdf#page=2) explain the differences from the paper.

### Language-specific changes

| Area | English | Telugu |
|---|---|---|
| Dataset preparation | Svarah clips with demographic profile grouping | IndicTTS clips using female/male source labels |
| Pronunciation frontend | ARPAbet lexicon with `g2p_en` fallback for unseen words | Unicode decomposition with inherent vowels, vowel signs and virama |
| Text normalization | Abbreviations, dates, times and numbers | Telugu digits, numbers, ordinal dates and times |
| Shared source changes | Independent configurations/weights, alignment checks, feature statistics and prediction exports | Same mechanisms with separate Telugu metadata and symbols |

The Telugu frontend is a deterministic phonemic approximation; assimilation, allophones and loanwords remain limitations. English first-pronunciation lexicon lookup can miss context-sensitive stress, as illustrated by the two **conduct** sentences below.

<a id="audio"></a>
<a id="variance-gallery"></a>
## 3. Test sentences, audio and variance plots

All WAVs are **22,050 Hz mono, 16-bit PCM**. Each row links to the audio and its three original predictor plots: **D** = duration, **F0** = pitch, **E** = energy. The supplied broken-word spacing in E3/T3 is preserved. Exact raw/normalized text and prediction arrays are available in the [English JSON](project/Results/English/variance_predictions_summary.json) and [Telugu JSON](project/Results/Telugu/variance_predictions_summary.json).

### English · step 8,000 · speaker 14

| ID | Assignment sentence | WAV / length | Plots |
|---|---|---|---|
| <a id="sample-e1"></a>**E1** | `The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/English/test_generated_audios/E1.wav) / 5.817 s | [D](project/Results/English/variance_adaptor_plots/duration/E1_duration.png) · [F0](project/Results/English/variance_adaptor_plots/pitch/E1_pitch.png) · [E](project/Results/English/variance_adaptor_plots/energy/E1_energy.png) |
| <a id="sample-e2a"></a>**E2a** | `The orchestra's conduct during the ceremony was praised by the audience.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/English/test_generated_audios/E2a.wav) / 3.286 s | [D](project/Results/English/variance_adaptor_plots/duration/E2a_duration.png) · [F0](project/Results/English/variance_adaptor_plots/pitch/E2a_pitch.png) · [E](project/Results/English/variance_adaptor_plots/energy/E2a_energy.png) |
| <a id="sample-e2b"></a>**E2b** | `The researchers will conduct an experiment to evaluate the proposed method.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/English/test_generated_audios/E2b.wav) / 3.913 s | [D](project/Results/English/variance_adaptor_plots/duration/E2b_duration.png) · [F0](project/Results/English/variance_adaptor_plots/pitch/E2b_pitch.png) · [E](project/Results/English/variance_adaptor_plots/energy/E2b_energy.png) |
| <a id="sample-e3"></a>**E3** | `The rhyth  m of the orchestra changed unexpectedly.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/English/test_generated_audios/E3.wav) / 2.519 s | [D](project/Results/English/variance_adaptor_plots/duration/E3_duration.png) · [F0](project/Results/English/variance_adaptor_plots/pitch/E3_pitch.png) · [E](project/Results/English/variance_adaptor_plots/energy/E3_energy.png) |
| <a id="sample-e4"></a>**E4** | `The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/English/test_generated_audios/E4.wav) / 6.525 s | [D](project/Results/English/variance_adaptor_plots/duration/E4_duration.png) · [F0](project/Results/English/variance_adaptor_plots/pitch/E4_pitch.png) · [E](project/Results/English/variance_adaptor_plots/energy/E4_energy.png) |

### Telugu · step 13,000 · speaker 0

| ID | Assignment sentence | WAV / length | Plots |
|---|---|---|---|
| <a id="sample-t1"></a>**T1** | `న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/Telugu/test_generated_audios/T1.wav) / 5.236 s | [D](project/Results/Telugu/variance_adaptor_plots/duration/T1_duration.png) · [F0](project/Results/Telugu/variance_adaptor_plots/pitch/T1_pitch.png) · [E](project/Results/Telugu/variance_adaptor_plots/energy/T1_energy.png) |
| <a id="sample-t2a"></a>**T2a** | `తాడుపై అతని పట్టు చాలా బలంగా ఉంది.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/Telugu/test_generated_audios/T2a.wav) / 2.577 s | [D](project/Results/Telugu/variance_adaptor_plots/duration/T2a_duration.png) · [F0](project/Results/Telugu/variance_adaptor_plots/pitch/T2a_pitch.png) · [E](project/Results/Telugu/variance_adaptor_plots/energy/T2a_energy.png) |
| <a id="sample-t2b"></a>**T2b** | `ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/Telugu/test_generated_audios/T2b.wav) / 2.740 s | [D](project/Results/Telugu/variance_adaptor_plots/duration/T2b_duration.png) · [F0](project/Results/Telugu/variance_adaptor_plots/pitch/T2b_pitch.png) · [E](project/Results/Telugu/variance_adaptor_plots/energy/T2b_energy.png) |
| <a id="sample-t3"></a>**T3** | `పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/Telugu/test_generated_audios/T3.wav) / 2.334 s | [D](project/Results/Telugu/variance_adaptor_plots/duration/T3_duration.png) · [F0](project/Results/Telugu/variance_adaptor_plots/pitch/T3_pitch.png) · [E](project/Results/Telugu/variance_adaptor_plots/energy/T3_energy.png) |
| <a id="sample-t4"></a>**T4** | `ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.` | [Play](https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main/project/Results/Telugu/test_generated_audios/T4.wav) / 8.998 s | [D](project/Results/Telugu/variance_adaptor_plots/duration/T4_duration.png) · [F0](project/Results/Telugu/variance_adaptor_plots/pitch/T4_pitch.png) · [E](project/Results/Telugu/variance_adaptor_plots/energy/T4_energy.png) |

Per-sentence observations and measurements are in [report pages 9–18](project/Report/Technical_Report.pdf#page=9). Duration is measured in mel frames; F0 is an inverse-normalized phone prediction, not a generated-waveform F0 measurement; energy uses STFT-magnitude units. Human pronunciation/naturalness review remains pending.

### What to inspect in each sample

| Sample | Evaluation focus / saved evidence |
|---|---|
| E1 | Rare long word; one saved `AH0` token has zero predicted duration |
| E2a / E2b | Noun versus verb **conduct**; both saved sequences use the same noun-stressed pronunciation |
| E3 | Broken input `rhyth  m`; normalization collapses spaces without repairing the word |
| E4 | Time/date expansion and **andaroo**; two `sp` tokens include a pause introduced at the hyphen in the expanded year |
| T1 | Long technical word, consonant clusters and vowel length |
| T2a / T2b | Different contexts of **పట్టు**, including geminate handling and phrase timing |
| T3 | Broken input `అడ  విలో`; source spacing does not establish a predicted silence token |
| T4 | Year/ordinal/time expansion; one comma pause expands to 52 frames, approximately 0.604 s |

**Plot interpretation:** duration bars show phone expansion; F0 and energy curves show predictor outputs after reversing normalization. Peaks identify positions to inspect for emphasis, while low values may occur around weak phones or pauses. Token identity, interpolation and the speaker also affect these patterns; the plots alone do not prove correct pronunciation.

For listening review, check intelligibility, target-word pronunciation, omissions/repetitions, stress, pauses and artifacts. The report separates these pending judgments from measured waveform and predictor properties.

<a id="datasets"></a>
## 4. Datasets and training setup

| Data | English | Telugu |
|---|---|---|
| Dataset | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) |
| Retained training | 3,359 utterances / 5.000 h | 2,164 utterances / 5.000 h |
| Retained validation | 674 utterances / 1.001 h | 421 utterances / 1.000 h |
| Training groups/voices | 115 demographic profiles | 2 source voices |

Counts/hours are from recovered evidence summarized in [report page 4](project/Report/Technical_Report.pdf#page=4). Selection uses seed 42, 1–20 s clips, waveform deduplication and voice/profile-aware partitions with split-specific reserves. English profiles are not verified unique speaker identities.

**Preprocessing:** resample to 22,050 Hz, normalize transcripts, obtain compatible MFA phone alignments, then extract mel, duration, PyWorld F0 and energy features. Preserve interior pauses and fit normalization/bounds on selected training data only. Alignment is required before `preprocess.py`; its full process is documented in the report.

### Selection and preprocessing summary

1. Resolve the dataset revision and create deterministic candidate pools with seed 42.
2. Partition within voice/profile groups, then filter clips and deduplicate waveforms.
3. Target about 5h training and 1h validation, with separate 0.75h/0.20h reserve pools.
4. Write normalized transcripts and mono audio; obtain compatible MFA phone alignments.
5. Reject invalid timelines/features, replenish each split from its own reserves, and fit statistics on retained training clips only.

The Svarah source pool is its dataset `test` split; the assignment's train/validation sets are created from that pool. Validation groups are represented in training. These are utterance-disjoint splits, rather than a claim of evaluation on entirely unseen speakers.

<a id="training-configuration"></a>
| Shared configuration | Value |
|---|---|
| Encoder / decoder | 4 / 6 FFT blocks; hidden 256; 2 heads |
| Variance heads | 256 channels; kernel 3; dropout 0.5; 256 pitch/energy bins |
| Features | 80 mel channels; FFT/window 1,024; hop 256; 0–8 kHz |
| Optimizer | Adam; betas (0.9, 0.98); epsilon 1e-9; batch 16; clipping 1.0 |
| Schedule | 4k warmup; 60k steps; anneal at 30k/45k/55k by 0.3 |
| Language configs | [English](project/English/config.yaml) · [Telugu](project/Telugu/config.yaml) |

<a id="split-integrity"></a>
<a id="limitations"></a>
**Evidence limits:** recovered manifests report zero ID overlap, but final audio hashes remain missing for 382 English and 339 Telugu reserve clips. The report also records 82 historical Telugu target/frontend mismatches. These and pending listening judgments are discussed on [page 19](project/Report/Technical_Report.pdf#page=19).

<a id="checkpoints"></a>
## 5. Checkpoint downloads

[Main Drive artifact folder](https://drive.google.com/drive/folders/1WsR1NHhzpkLvk3V0g8X3Jy-xsXNlsYHf?usp=sharing). The large acoustic weights are externally hosted; public download availability has not been revalidated in this documentation update.

| Model | Step / speaker | Download | Restore to |
|---|---|---|---|
| English | 8,000 / 14 | [8000.pth.tar](https://drive.google.com/uc?export=download&id=1asQbbLU4y9qPAHd8Jvu2T1G-5VES6X7P) | `project/English/checkpoints/8000.pth.tar` |
| Telugu | 13,000 / 0 | [13000.pth.tar](https://drive.google.com/uc?export=download&id=1loNuZ9gRVLTobXHzzwZGsDe8KkB_FB-p) | `project/Telugu/checkpoints/13000.pth.tar` |

SHA-256 checksums and sizes: [English checkpoint metadata](project/English/checkpoints/README.md) · [Telugu checkpoint metadata](project/Telugu/checkpoints/README.md). Restore each model's original `stats.json`, `speakers.json`, symbol/lexicon setup and `hifigan/generator_universal.pth.tar` together with its acoustic weights.

### Integrity and restoration

```text
English 8000.pth.tar: 420,505,334 bytes
SHA-256: b50f0558474eab89e95e8079df23cb3c364e339ce00fe6d88316beb4b86ff49a
Telugu 13000.pth.tar: 420,944,630 bytes
SHA-256: 49902507629887721818ec180dac2b419609f6c7a9e601c6d767a11233d1baa1
```

From the repository root, download with `gdown` or use the browser links above:

```bash
python -m pip install gdown
gdown 1asQbbLU4y9qPAHd8Jvu2T1G-5VES6X7P -O project/English/checkpoints/8000.pth.tar
gdown 1loNuZ9gRVLTobXHzzwZGsDe8KkB_FB-p -O project/Telugu/checkpoints/13000.pth.tar
```

Verify the downloaded bytes with `sha256sum` or PowerShell `Get-FileHash -Algorithm SHA256`. Keep the original speaker-ID mapping and normalization statistics coupled to the selected checkpoint; generating new metadata can change the meaning of the saved weights.

<a id="reproducibility"></a>
## 6. Reproducibility and commands

**Before running:** restore the omitted `project/tools/` helpers, matching model metadata and vocoder weights. Colab also references missing restoration/pipeline/probe scripts, `model_artifacts.json` and `Results/test_sentences.json`, and needs its root-level paths adjusted to `project/`. See [report page 20](project/Report/Technical_Report.pdf#page=20). These commands target a restored environment; fresh-clone GPU execution is not claimed.

```bash
git clone https://github.com/SandeshAwate6055/FastSpeech2-English-Telugu.git
cd FastSpeech2-English-Telugu
python -m pip install -r project/requirements.txt
python -c "import nltk; [nltk.download(n) for n in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']]"

# Generate the five English assignment samples.
cd project/English
python inference.py --config config.yaml \
  --ckpt_file checkpoints/8000.pth.tar --speaker_id 14 \
  --result_path ../Results/reproduced/English

# Generate the five Telugu assignment samples.
cd ../Telugu
python inference.py --config config.yaml \
  --ckpt_file checkpoints/13000.pth.tar --speaker_id 0 \
  --result_path ../Results/reproduced/Telugu
```

Omitting `--text` selects the built-in assignment sentences. Use new result directories to preserve the submitted outputs. Each language run writes five WAVs, fifteen plots and a prediction JSON.

For a new training run, execute the following independently from each language directory after restoring the helpers and choosing fresh experiment paths:

```bash
python prepare_data.py --train_hours 5.0 --val_hours 1.0 --seed 42
# Obtain compatible MFA phone TextGrids before the next step.
python preprocess.py --config config.yaml
# Validate features/provenance and held-out speech before full training.
python train.py --config config.yaml --total_step 60000
```

MFA is a separate training dependency. Svarah may require Hugging Face approval and `HF_TOKEN`. The requirements use lower version bounds rather than a complete historical environment lock.

### Custom inference and controls

From `project/English`, after restoring the required bundles:

```bash
python inference.py --config config.yaml \
  --ckpt_file checkpoints/8000.pth.tar --speaker_id 14 \
  --text "This is a custom synthesis example." --sample_id custom \
  --pitch_control 1.0 --energy_control 1.0 --duration_control 1.0 \
  --result_path ../Results/custom/English
```

All controls default to 1.0. Pitch/energy multipliers act in normalized feature space, so they do not guarantee the same percentage change in physical F0 or loudness. Custom sentences supplement the mandatory test set.

### Colab workflow

The notebook defaults to `MODE = "evaluate"`; `MODE = "train"` describes a fresh experiment. Once its missing support files and paths are restored, its stages cover:

1. Environment/Drive setup and model-bundle restoration.
2. Independent English and Telugu data preparation and alignment.
3. Vocoder reconstruction and a small overfit diagnostic.
4. Held-out pilots, checkpoint comparison and optional full training.
5. Exact test synthesis, audio playback, variance plots and packaging.

Expected output per language: **5 WAVs + 15 variance PNGs + prediction JSON**. Duration sums should match mel frames, token/predictor arrays should have equal lengths, and checkpoint/speaker IDs should match the selected bundle. Existing result summaries cause the scripts to stop unless overwriting is explicitly enabled.

<a id="report"></a>
## 7. Technical report guide

**[Open Technical_Report.pdf](project/Report/Technical_Report.pdf)** for the detailed demonstration and analysis.

| Topic | Pages |
|---|---|
| Architecture and language-specific changes | [2–3](project/Report/Technical_Report.pdf#page=2) |
| Data, preprocessing and configuration | [4](project/Report/Technical_Report.pdf#page=4) |
| English / Telugu loss analysis | [5–6](project/Report/Technical_Report.pdf#page=5) / [7–8](project/Report/Technical_Report.pdf#page=7) |
| English / Telugu sample analysis | [9–13](project/Report/Technical_Report.pdf#page=9) / [14–18](project/Report/Technical_Report.pdf#page=14) |
| Comparison, failures and improvements | [19](project/Report/Technical_Report.pdf#page=19) |
| Dependencies, commands and model links | [20](project/Report/Technical_Report.pdf#page=20) |

The PDF includes internal bookmarks, clickable source/audio links, corrected vector diagrams, all ten per-sentence analyses and checkpoint checksums. The report directory contains **only `Technical_Report.pdf`**, following the assignment structure.

For a full demonstration, read the architecture and data protocol first, compare the loss curves, then open each sample's WAV alongside its duration/F0/energy plots. Finish with the cross-language limitations and reproduction prerequisites. PDF page-fragment links depend on the viewer; use the printed page numbers if a browser does not jump to the selected page.

<a id="files"></a>
<a id="source-index"></a>
## 8. Code and references

```text
assets/fastspeech2-architecture.png  # README overview image
project/
├── English/    # prepare_data.py, preprocess.py, train.py, inference.py,
│               # config.yaml, checkpoints/, model/, text/, preprocessor/, ...
├── Telugu/     # independent language implementation and configuration
├── Results/
│   ├── English/ # loss figures, metrics/, test_generated_audios/, variance plots
│   └── Telugu/  # same result structure
├── Report/
│   └── Technical_Report.pdf
├── FastSpeech2_Colab.ipynb
├── README.md
└── requirements.txt
```

| Source | English | Telugu |
|---|---|---|
| Data preparation | [prepare_data.py](project/English/prepare_data.py) | [prepare_data.py](project/Telugu/prepare_data.py) |
| Feature extraction | [preprocess.py](project/English/preprocess.py) | [preprocess.py](project/Telugu/preprocess.py) |
| Training | [train.py](project/English/train.py) | [train.py](project/Telugu/train.py) |
| Test constants and synthesis | [inference.py](project/English/inference.py) | [inference.py](project/Telugu/inference.py) |
| Text frontend | [cleaners.py](project/English/text/cleaners.py) | [cleaners.py](project/Telugu/text/cleaners.py) |
| Acoustic model | [fastspeech2.py](project/English/model/fastspeech2.py) | [fastspeech2.py](project/Telugu/model/fastspeech2.py) |

<a id="references"></a>
References: [assignment specification](FastSpeech2_Assignment.pdf), [FastSpeech 2 paper](https://arxiv.org/abs/2006.04558), and [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (report reference commit `d4e79eb`). Dataset links are in [Section 4](#datasets).

<a id="assignment-map"></a>
## 9. Assignment requirements and evidence

| No. | Requirement | Submission evidence | Status |
|---|---|---|---|
| 1 | Separate English and Telugu models | [English code](project/English) · [Telugu code](project/Telugu) · [Configurations](#training-configuration) | Present |
| 2 | About 5h train / 1h validation, with separate splits | [Dataset summary](#datasets) · [Report p. 4](project/Report/Technical_Report.pdf#page=4) | Counts reported; final hash audit incomplete |
| 3 | Trained checkpoints for both languages | [Downloads and restoration paths](#checkpoints) | External links supplied |
| 4 | Training and validation loss plots | [Four loss graphs and CSVs](#training-results) | Present |
| 5 | Five test WAVs per language | [Ten exact test inputs and audio links](#audio) | Present |
| 6 | Duration, F0 and energy plots for every sample | [Thirty plot links](#variance-gallery) | Present |
| 7 | Detailed technical report | [20-page PDF](project/Report/Technical_Report.pdf) | Present; listening judgments pending |
| 8 | Required project structure and public evaluation artifacts | [Folder structure](#files) · [GitHub repository](https://github.com/SandeshAwate6055/FastSpeech2-English-Telugu) | Present |
| 9 | Preprocessing, training and inference instructions | [Commands and prerequisites](#reproducibility) | Documented; runtime restoration required |

[Back to top](#top) · [Main Drive](https://drive.google.com/drive/folders/1WsR1NHhzpkLvk3V0g8X3Jy-xsXNlsYHf?usp=sharing) · [Technical report](project/Report/Technical_Report.pdf)

