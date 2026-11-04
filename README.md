# FastSpeech 2: English and Telugu

Two independent FastSpeech 2 acoustic models adapted from
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), reference commit
`d4e79eb52e8b01d24703b2dfc0385544092958f3`,
with separate English (Svarah) and Telugu (IndicTTS Telugu) frontends.
Universal HiFi-GAN converts predicted mels into 22,050 Hz speech.

## Start here

- [Technical report](Report/Technical_Report.pdf): architecture, workflow, data protocol, CSV-derived losses, and analysis of all ten inputs.
- [Submission audit](Results/submission_audit.json): format/text/length checks and unresolved access requirements.
- [Model artifact registry](model_artifacts.json): the exact selected checkpoint steps and missing download URLs.
- [Colab notebook](notebooks/FastSpeech2_Colab.ipynb): seven numbered sections, some containing multiple cells.
- [Modifications](CHANGES.md): changes to the reference implementation.

**Status:** ten WAVs, thirty variance plots and four loss figures are present and structurally checked.
The checkpoint weights and matching inference metadata are not included locally and currently have no
public download links. Final retained-hour/split evidence and independent listening observations are
also outstanding. Do not interpret artifact presence as proof of intelligibility or complete reproducibility.

## Results and selected checkpoints

Training histories extend to 60,000 steps. The submitted audio uses earlier checkpoints,
consistent with the lowest validation totals in the supplied CSVs:

| Language | Audio checkpoint | Speaker ID | Minimum logged validation total | Validation total at 60k | Training total at 60k |
|---|---:|---:|---:|---:|---:|
| English | **8,000** | 14 | 2.702820 | 2.735320 | 0.921556 |
| Telugu | **13,000** | 0 | 1.965954 | 1.979367 | 0.662709 |

The selected steps come from prediction metadata; minima and endpoints come from
[English CSVs](Results/English/metrics/) and [Telugu CSVs](Results/Telugu/metrics/).
Training entries are sampled minibatch losses; validation entries are aggregates.
Lower losses across different corpora do not establish better speech quality.

| ID | Audio | Prediction plots |
|---|---|---|
| E1 | [WAV](Results/English/test_generated_audios/E1.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E1_duration.png), [F0](Results/English/variance_adaptor_plots/pitch/E1_pitch.png), [Energy](Results/English/variance_adaptor_plots/energy/E1_energy.png) |
| E2a | [WAV](Results/English/test_generated_audios/E2a.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E2a_duration.png), [F0](Results/English/variance_adaptor_plots/pitch/E2a_pitch.png), [Energy](Results/English/variance_adaptor_plots/energy/E2a_energy.png) |
| E2b | [WAV](Results/English/test_generated_audios/E2b.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E2b_duration.png), [F0](Results/English/variance_adaptor_plots/pitch/E2b_pitch.png), [Energy](Results/English/variance_adaptor_plots/energy/E2b_energy.png) |
| E3 | [WAV](Results/English/test_generated_audios/E3.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E3_duration.png), [F0](Results/English/variance_adaptor_plots/pitch/E3_pitch.png), [Energy](Results/English/variance_adaptor_plots/energy/E3_energy.png) |
| E4 | [WAV](Results/English/test_generated_audios/E4.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E4_duration.png), [F0](Results/English/variance_adaptor_plots/pitch/E4_pitch.png), [Energy](Results/English/variance_adaptor_plots/energy/E4_energy.png) |
| T1 | [WAV](Results/Telugu/test_generated_audios/T1.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T1_duration.png), [F0](Results/Telugu/variance_adaptor_plots/pitch/T1_pitch.png), [Energy](Results/Telugu/variance_adaptor_plots/energy/T1_energy.png) |
| T2a | [WAV](Results/Telugu/test_generated_audios/T2a.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T2a_duration.png), [F0](Results/Telugu/variance_adaptor_plots/pitch/T2a_pitch.png), [Energy](Results/Telugu/variance_adaptor_plots/energy/T2a_energy.png) |
| T2b | [WAV](Results/Telugu/test_generated_audios/T2b.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T2b_duration.png), [F0](Results/Telugu/variance_adaptor_plots/pitch/T2b_pitch.png), [Energy](Results/Telugu/variance_adaptor_plots/energy/T2b_energy.png) |
| T3 | [WAV](Results/Telugu/test_generated_audios/T3.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T3_duration.png), [F0](Results/Telugu/variance_adaptor_plots/pitch/T3_pitch.png), [Energy](Results/Telugu/variance_adaptor_plots/energy/T3_energy.png) |
| T4 | [WAV](Results/Telugu/test_generated_audios/T4.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T4_duration.png), [F0](Results/Telugu/variance_adaptor_plots/pitch/T4_pitch.png), [Energy](Results/Telugu/variance_adaptor_plots/energy/T4_energy.png) |

[English loss figures](Results/English/) and [Telugu loss figures](Results/Telugu/) contain training and validation curves.
Exact raw text, normalized text, speaker IDs, durations, F0 and energy are in each language's
`variance_predictions_summary.json`. E3 and T3 retain intentional double spaces and Telugu diacritics.
These ten required inputs are test inputs, not training examples to memorize.

## Assignment structure

```text
English/ and Telugu/
  prepare_data.py, preprocess.py, train.py, inference.py, config.yaml
  audio/, text/, transformer/, model/, preprocessor/, utils/, hifigan/
  checkpoints/                 selected weights required; not currently supplied
Results/{English,Telugu}/
  training_loss.png, validation_loss.png
  test_generated_audios/
  variance_adaptor_plots/{duration,pitch,energy}/
  variance_predictions_summary.json
  metrics/{train_losses.csv,val_losses.csv}
Results/submission_audit.json
Report/{Technical_Report.pdf,generate_report.py,requirements.txt}
tools/audit_submission.py
model_artifacts.json
notebooks/FastSpeech2_Colab.ipynb
CHANGES.md, README.md, requirements.txt
```

## Architecture and data protocol

- Four FFT encoder blocks, six FFT decoder blocks, hidden size 256, two attention heads.
- Learned speaker embedding added to encoder states; both language configurations enable multi-speaker synthesis.
- Phone-level duration, pitch and energy predictors; pitch/energy embeddings precede length expansion.
- Eighty mel channels, residual PostNet, universal HiFi-GAN.
- Five summed losses: mel and PostNet L1, log-duration MSE, normalized pitch MSE, normalized energy MSE.

This differs from paper v8's four-layer decoder, frame-level CWT pitch, frame-level energy,
absence of the extra PostNet, and Parallel WaveGAN. It is a ming024-based adaptation.

Preparation scripts default to seed 42, 1-20 second clips, mono 22,050 Hz, SHA-1 waveform deduplication,
and speaker-aware utterance splits targeting approximately 5h train / 1h validation plus reserves.
English uses demographic grouping; a profile is not proven to be a unique person.
Telugu uses both source voices and a Unicode-based phonemic decomposition.
Final processed hours, exact source revisions and split counts must be supplied from the actual
run's manifests and preprocessing summaries. Earlier unverified 19-speaker/count claims have been removed.

Interior MFA pauses are preserved with `include_empty_intervals=True`.
Pre-slice timeline checks reject substantial drift; selected training examples alone determine
normalization statistics and quantization bounds. These checks do not prove phonetic boundary accuracy.

## Environment

Python 3.10 or a tested compatible environment is recommended for the acoustic pipeline.
The root requirements file contains lower bounds, **not a pinned environment lock**.
Record the actual PyTorch/CUDA and library versions with each experiment.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

Install MFA in a separate environment matching your alignment model (historical run: 2.2.17).
Supply the matching universal HiFi-GAN checkpoint and configuration from the reference implementation.
Do not use an unrelated vocoder with merely matching mel dimensions.

## Reproduce the submitted speech

First obtain the model bundles listed in `model_artifacts.json`.
The model owner supplied this [Drive artifact folder](https://drive.google.com/drive/folders/1WsR1NHhzpkLvk3V0g8X3Jy-xsXNlsYHf?usp=sharing).
Its contents and evaluator access are awaiting verification; the folder URL alone does not establish
which checkpoint or preprocessing metadata belongs to each submitted WAV.
Each bundle must contain the selected checkpoint, the exact run configuration, stats.json,
speakers.json, generated lexicon and frontend/symbol inventory. **A private path beginning
/content/drive/ is not a public download link.** Public URLs and hashes are still required.

Restore stats.json and speakers.json to each language's preprocessed_data directory and the
English generated dictionary to its lexicon directory, using the bundle's exact configuration.

Run each command from the indicated language directory:

```bash
cd English
python inference.py --config config.yaml --ckpt_file checkpoints/8000.pth.tar --preprocessed_path preprocessed_data --speaker_id 14 --result_path ../Results/English

cd ../Telugu
python inference.py --config config.yaml --ckpt_file checkpoints/13000.pth.tar --preprocessed_path preprocessed_data --speaker_id 0 --result_path ../Results/Telugu
```

These explicit checkpoint/speaker arguments reproduce the recorded selection. Replacing them
with step 60,000 changes the experiment. Download links and weights have not yet been independently verified.

## Prepare and train a new experiment

Use new output directories to preserve submitted artifacts. Svarah requires an authorized
Hugging Face account; keep credentials in environment variables or Colab Secrets.

English preparation and matching ARPAbet alignment:

```bash
cd English
python prepare_data.py --raw_dir raw_data --train_hours 5.0 --val_hours 1.0 --seed 42
mfa model download acoustic english_us_arpa
mfa align raw_data lexicon/english_svarah_mfa.dict english_us_arpa preprocessed_data/TextGrid --clean
python preprocess.py --config config.yaml
python train.py --config config.yaml --total_step 60000 --ckpt_path experiments/new/checkpoints --log_path experiments/new/logs --result_path experiments/new/results
```

Telugu preparation:

```bash
cd Telugu
python prepare_data.py --raw_dir raw_data --train_hours 5.0 --val_hours 1.0 --seed 42
```

Train the Telugu aligner on a separate corpus containing **only training/reserve-training**
WAV/LAB pairs selected by the manifests; never train it on validation recordings.
After creating that train-only corpus:

```bash
mfa train TRAIN_ONLY_CORPUS lexicon/telugu-lexicon.txt te_acoustic.zip --clean
mfa align raw_data lexicon/telugu-lexicon.txt te_acoustic.zip preprocessed_data/TextGrid --clean
python preprocess.py --config config.yaml
python train.py --config config.yaml --total_step 60000 --ckpt_path experiments/new/checkpoints --log_path experiments/new/logs --result_path experiments/new/results
```

`TRAIN_ONLY_CORPUS` is a required prepared directory, not a literal existing path.
Inspect real-mel vocoder reconstruction, alignment quality, retained hours, and free-running
held-out speech before a full run. Never assume a fixed training duration guarantees intelligibility.

## Colab

[Open the current notebook](https://colab.research.google.com/github/SandeshAwate6055/FastSpeech2-English-Telugu/blob/main/notebooks/FastSpeech2_Colab.ipynb).
Review cells before running: the notebook can restore existing archives and resume checkpoints.
Archive size or TextGrid count alone does not verify that restored features match the selected
checkpoint or regenerated audio. A full rerun does not reproduce earlier checkpoint selection automatically.
Drive mounting and runtime access happen in your authenticated Colab session.

## Audit and rebuild the report

The structural audit uses only the Python standard library:

```bash
python tools/audit_submission.py --output Results/submission_audit.json
python tools/audit_submission.py --strict
```

Normal mode fails on inconsistent artifacts. Strict mode also fails when selected weights,
inference metadata, loss CSVs or retained-hour evidence are missing. Structural checks are
not a listening evaluation or verification that remote download links work.

Build the report and render page previews:

```bash
pip install -r Report/requirements.txt
python Report/generate_report.py
python Report/render_report.py
```

On Windows the report uses Nirmala UI through the native text renderer for Telugu shaping.
Elsewhere install a Telugu font, provide its path in TELUGU_FONT, and use Pillow with libraqm.
The report computes numerical loss tables from the preserved CSVs and per-sentence metrics from
prediction JSON; it does not invent MOS, pronunciation ratings or missing dataset counts.

## Remaining evaluator-access requirements

1. Add verified public checkpoint and metadata download URLs/hashes to model_artifacts.json.
2. Include the actual retained-hour, split/hash and preprocessing evidence.
3. Record independent listening observations for all ten WAVs, including Telugu-proficient review.
4. Supply the original training environment/config provenance.
5. Rerun the strict audit and verify all download links without account access before submission.

## References

- [FastSpeech 2, supplied v8 paper](https://arxiv.org/abs/2006.04558)
- [ming024 reference implementation](https://github.com/ming024/FastSpeech2)
- [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)
