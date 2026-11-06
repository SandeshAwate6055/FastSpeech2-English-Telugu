# Technical report and interactive demonstration

The report follows the seven analysis requirements in [FastSpeech2_Assignment.pdf](../../FastSpeech2_Assignment.pdf). It preserves the existing 20-page structure and sample page numbers used by the repository README.

| Open | What it contains |
|---|---|
| [Technical_Report.pdf](Technical_Report.pdf) | Twenty pages with PDF bookmarks, internal navigation, corrected vector diagrams, source/audio links, datasets and settings, loss analysis, ten sample analyses, comparison and reproduction instructions. |
| [Interactive_Report.html](Interactive_Report.html) | A self-contained companion with all ten embedded WAVs, hoverable/keyboard-focusable prediction points, token tables, expandable sample sections, language filters and listening-note export. |
| [report_measurements.json](report_measurements.json) | Measurements recomputed from the submitted WAVs, prediction arrays and loss CSVs, including audio hashes, minimum validation steps and layout checks. |
| [evidence_context.json](evidence_context.json) | A labeled report-only snapshot of recovered dataset summaries, historical artifact links/hashes and pending listening records. This is not a complete acoustic runtime bundle. |

**Using the interactive companion:** GitHub displays HTML source rather than running this page. Download `Interactive_Report.html` using **Download raw file**, then open it in a modern browser, or open it directly from a cloned repository. All ten WAVs, loss figures and vector diagrams are embedded; no third-party scripts, server or internet access are required for those features. Links to Colab, Drive and repository files require internet access.

Listen before entering subjective judgments. Each sample has fields for the reviewer, intelligibility, pronunciation, completeness, prosody and overall quality. Notes are saved in browser-local storage where supported. **Export listening notes** downloads JSON; it does not send or publish the notes. Export before closing the page because browsers can restrict local-file storage.

## Report build

From the cloned repository root:

```bash
python -m pip install -r project/Report/requirements.txt
python project/Report/generate_report.py
```

The generator reads the existing `project/Results/` WAVs, prediction JSON, loss CSVs and four loss PNGs. It does not run acoustic inference/training or change those inputs. It writes only the PDF, HTML companion and measurements JSON inside `project/Report/`.

To check bookmarks, links, embedded audio hashes, HTML anchors, JavaScript syntax and PDF text boundaries, and render page PNGs for review:

```bash
python project/Report/verify_report.py --render-dir tmp/pdfs/report-review
```

The JavaScript syntax check uses Node.js. The measurements JSON records the report generator's tested Python and library versions; those are separate from the unknown historical acoustic-training environment.

For Telugu in the PDF, Windows uses GDI shaping with **Nirmala UI**. On other systems, use a Pillow build with libraqm, install a Telugu font and set `TELUGU_FONT` to its font path. Telugu source text remains selectable UTF-8 text in the HTML and prediction JSON; the PDF uses shaped text images for those sentences.

To incorporate completed reviewer records for the exact existing audio:

```bash
python project/Report/generate_report.py --review-json listening_review.json
```

The generator checks each record's audio SHA-256 against the submitted WAV. Reviewer-reported judgments are labeled separately from computed waveform/predictor measurements. Current published perceptual assessments remain pending; no MOS, WER or confirmed naturalness scores are fabricated.

## Requirement map

| Assignment technical-report requirement | PDF pages | Interactive section |
|---|---|---|
| Implementation approach, reference architecture, English/Telugu changes | 2-3 | Architecture and reference/adaptation sections |
| Dataset selection, subsampling, preprocessing, split and configuration | 4 | Data-to-speech process; linked code/configuration/evidence |
| Training/validation curves, convergence, instability and overfitting | 5-8 | Embedded original curves and measured component-loss tables |
| Per-sentence speech, pronunciation and overall quality observations | 9-18 | Ten audio players, objective waveform observations and pending human-review fields |
| Per-sentence duration/F0/energy plots and explanations | 9-18 | Hoverable plots, exact token tables and links to all thirty original PNGs |
| English/Telugu comparison, limitations, failures and improvements | 19 | Comparison and evidence-led improvement sequence |
| Dependencies, configurations and preprocessing/training/inference commands | 20 | Reproduction commands, model integrity table and external access links |

## Evidence limits

The report differentiates historical trained outputs from current source fixes. Complete final waveform-hash coverage is still missing for 382 English and 339 Telugu retained reserve clips in the recovered evidence. The earlier report records 82 Telugu target/frontend discrepancies. Current GitHub acoustic scripts/notebooks also reference omitted runtime helpers and metadata and contain a root-versus-`project/` notebook path mismatch. These are described, not silently repaired by this report update. External Drive downloads and successful GPU reproduction are not claimed by the report build.

Only files within `project/Report/` are part of this report revision. Acoustic code, configurations, notebook, README, checkpoints, WAVs, original plots and CSV histories are preserved.
