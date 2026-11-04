# Assignment completion and evidence status

The assignment PDF is authoritative. This checklist distinguishes available outputs from missing evaluator access and experimental evidence.

| Assignment requirement | Available evidence | Remaining work |
|---|---|---|
| Separate English/Telugu implementation and modifications | Language directories, configs, CHANGES.md; report sections 1-2 | Confirm exact configurations used for the supplied checkpoints |
| Approximately 5h train / 1h validation; no overlap | Preparation/preprocessing code and documented protocol | Supply actual final manifests, data_stats.json, preprocess_summary.json, split/hash checks and dataset revisions |
| Training and validation losses | Four figures and four preserved source CSVs through 60,000 steps | Supply original environment, hardware/time and run provenance |
| Speech for all five inputs per language | Ten 22,050 Hz mono WAVs and exact raw text in prediction JSON | Independent listening/target-word pronunciation review, particularly Telugu |
| Duration/F0/energy predictions for each input | Thirty labeled plots, prediction arrays, report section 5 | Perceptual observations must be recorded rather than inferred from plots |
| Separately accessible trained checkpoints | Selected steps recorded in model_artifacts.json | English 8k / Telugu 13k weights and matching inference metadata; public download links |
| Technical report and reproducibility | Revised PDF, source generator, README and audit | Integrate remaining run evidence, model URLs and listening observations before claiming complete |

## Access needed from the model owner

Provide either a downloadable shared folder/link or local files under the project workspace. No password, token or blanket Google-account access is needed.

For each language, supply:

1. The checkpoint used to create the submitted audio: English step 8,000 and Telugu step 13,000. It may be stored as best_val.pth.tar; verify its embedded step before labeling it.
2. Matching stats.json and speakers.json, the exact run configuration, generated lexicon and text/symbol inventory. Never regenerate speaker IDs or statistics and assume they match existing weights.
3. Final train.txt/val.txt, primary/reserve selection manifests, data_stats.json, preprocess_summary.json and any recorded zero-overlap audit. These establish final data hours and identities.
4. Dataset revisions and the exported training environment, CUDA/PyTorch versions, hardware and elapsed-time records.
5. Listening notes for all ten WAVs, preferably including a Telugu-proficient reviewer. Record intelligibility, omissions/repetitions, target-word pronunciation, prosody and artifacts. Do not invent a MOS score.

Loss CSVs have already been recovered locally and copied unchanged to Results/<Language>/metrics/. The supplied histories place the lowest validation totals at steps 8,000 and 13,000, matching the prediction metadata.

## Checks

Run `python tools/audit_submission.py --strict` from the repository root. Strict mode intentionally fails until checkpoint/metadata access and final data evidence are supplied. A successful structural audit does not verify remote-link accessibility or speech quality; check those separately without requiring an evaluator to sign in.

After evidence is supplied, update model_artifacts.json and rebuild the report with `python Report/generate_report.py`. Render and inspect it with `python Report/render_report.py`. Keep selected-checkpoint documentation aligned with the actual WAV metadata.
