# Assignment requirements and current evidence

The assignment specification is authoritative. This file separates checked artifacts from evidence that still needs recovery or listening review. Updated 4 November 2026.

## Requirement matrix

| Requirement | Evidence | Current status |
|---|---|---|
| Separate English/Svarah and Telugu/IndicTTS models | Language implementations, configs and selected checkpoint registry | Local selected checkpoint hashes match; live external links not rechecked |
| Approximately 5h train / 1h validation per language | Published preprocessing summaries and selected ID manifests in `Results/<Language>/metadata/` | Recovered summaries: English 5.000h/1.001h; Telugu 5.000h/1.000h |
| No training/validation overlap | Selected manifests and source preparation statistics | Final ID overlap is zero; final reserve audio hashes are incomplete |
| Complete preprocessing, training and inference code | Required entrypoints and reference-derived modules in both language folders | Confirmed review issues repaired; full GPU execution of this revision remains untested |
| Ten exact test inputs and WAVs | Frozen `Results/test_sentences.json`, inference constants and prediction summaries | Structural/text/length checks pass; E3/T3 double spaces retained |
| Thirty duration/F0/energy plots | Required `Results/<Language>/variance_adaptor_plots/` paths | All required images are present and decodable |
| Four loss curves | Required PNGs plus preserved CSV histories under `metrics/` | 600 training and 60 validation records per language, through 60k steps |
| Pronunciation and speech-quality observations for every sentence | `Results/listening_review.json`, local `Report/audio_review.html` and report | Pending human review; measurements are not substituted for listening |
| Detailed technical report | `Report/Technical_Report.pdf` and its generator | 20 pages with diagrams, workflows, losses, ten analyses, measured spectrograms and clickable links |
| Reproducibility and evaluator access | README quickstart, metadata bundles, checkpoint READMEs, artifact registry and Colab notebook | Local restoration/checks covered; live downloads and full inference need an enabled runtime |
| Prescribed project folder structure | English/, Telugu/, Results/, Report/, README.md and requirements.txt | Retained; every supporting addition is explained in README |

## Machine audit

- Structural validity: **passes**.
- Final submission readiness: **pending**.
- Machine-detected artifact errors: **0**.
- Missing required evidence items: **4**, listed below.
- Full record: [Results/submission_audit.json](Results/submission_audit.json).

- English: audio hashes missing for 382 retained reserve utterances; complete final audio-overlap verification is pending
- English: per-sentence listening observations pending: E1, E2a, E2b, E3, E4
- Telugu: audio hashes missing for 339 retained reserve utterances; complete final audio-overlap verification is pending
- Telugu: per-sentence listening observations pending: T1, T2a, T2b, T3, T4

The strict audit exits with a nonzero status until these evidence requirements are satisfied. A nonempty URL or an existing summary file is not treated as proof of model quality or complete dataset disjointness.

## Checkpoints and inference

Use [model_artifacts.json](model_artifacts.json) for both download links, exact filenames, sizes and SHA-256 hashes. Per-language [English](English/checkpoints/README.md) and [Telugu](Telugu/checkpoints/README.md) checkpoint instructions explain the permitted external hosting.

From the repository root:

```bash
python tools/restore_artifacts.py --download
python English/inference.py --config English/config.yaml --ckpt_file English/checkpoints/8000.pth.tar --speaker_id 14 --result_path experiments/review/English
python Telugu/inference.py --config Telugu/config.yaml --ckpt_file Telugu/checkpoints/13000.pth.tar --speaker_id 0 --result_path experiments/review/Telugu
```

The commands synthesize the five fixed inputs per language. Existing result sets require a new output directory or the explicit `--overwrite` flag.

## Code corrections and historical results

Missing-checkpoint fallback, TextDataset cache access, configuration-relative paths, Telugu lexical boundaries, unknown-speech handling and experiment resume checks have been repaired. The notebook preserves dirty checkouts, prepares fresh alignments rather than trusting archive size/counts, and includes diagnostic and pilot listening gates.

The 82 historical Telugu target/frontend mismatches are recorded in [Results/frontend_audit.json](Results/frontend_audit.json). The current code rejects unknown-speech alignments for new runs. The existing checkpoints, WAVs, predictor arrays and loss histories remain historical outputs; they were not retrained or replaced by these changes.

## Finish the remaining evidence

1. Recover original reserve manifests or original selected WAVs. Fill blank `audio_sha1` values only from recovered source records or matching resampled/peak-normalized int16 PCM sample bytes (excluding the WAV header). Update bundle checksums and rerun the audit. Never manufacture hashes from utterance names.
2. Listen to all ten WAVs using the local reviewer. Export actual pronunciation, completeness and prosody observations with reviewer/date and matching audio hashes; replace `Results/listening_review.json` and rebuild the report.
3. Verify public model links and perform a full inference pass in the enabled Colab/GPU environment. If speech is poor, follow the notebook's fresh-alignment, overfit and held-out pilot gates before extending training.
4. Run the strict audit and visually inspect the rebuilt report. Submit the public repository URL through the assignment form after reviewing all remaining findings.
