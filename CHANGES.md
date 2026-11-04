# Changes from ming024/FastSpeech2

Reference commit: `d4e79eb52e8b01d24703b2dfc0385544092958f3`.
This log describes implementation changes, not proof that every historical run used the current code.

## Repository and architecture

- Independent English/ and Telugu/ pipelines with unified config.yaml files.
- Both configurations enable speaker embeddings, phone-level pitch/energy and universal HiFi-GAN.
- Four encoder/six decoder FFT blocks, hidden size 256, two attention heads; maximum sequence length 2,000.
- CLI path, checkpoint and step overrides support isolated experiments.
- The reference PostNet is retained. Paper v8 differs: four decoder blocks, frame-level CWT pitch, frame-level energy, no added PostNet and Parallel WaveGAN.
- Model code is not wholly unmodified: energy conditioning uses e_control; length-regulator output tensors follow the input device.

## Language frontends

- English uses the supplied lexicon plus g2p_en OOV pronunciations and a generated MFA dictionary; time/date/number normalization precedes phonemization.
- Telugu uses Unicode decomposition with inherent vowels, dependent-vowel mapping and virama suppression. Anusvara/visarga are retained; number/time/ordinal expansion is language-specific.
- This is a rule-based phonemic approximation, not a demonstrated linguistically validated FST. Contextual pronunciations and assimilation remain limitations.
- Exact raw test strings, intact Telugu diacritics and intentional E3/T3 double spaces are preserved separately from normalized text.
- Unsupported phones and prediction-length mismatches raise errors instead of silently truncating arrays.

## Preparation and alignment

- Deterministic seed-42, speaker-aware utterance splits with waveform SHA-1 deduplication and partition-specific reserves.
- English demographic profiles must not be described as verified unique individuals. Earlier documents gave conflicting counts; actual run manifests are required.
- Both Telugu source voices are supported, with explicit speaker conditioning.
- Duration targets come from genuine TextGrids; fixed-weight duration fallbacks were removed.
- The reader retains empty intervals. Interior gaps become sp; overlapping/nonmonotonic intervals are rejected.
- Pre-slice feature/timeline checks guard against missing interior pauses and discarded tails. Passing a length check is not proof of phone-boundary accuracy.
- Normalization statistics and quantization bounds use selected training examples. Raw caches support idempotent regeneration; unselected reserves are excluded from final feature outputs.

## Training and inference

- In-memory dataset/validation-loader caching reduces repeated small-file reads.
- Checkpoint loading explicitly handles PyTorch's weights_only behavior; load only trusted artifacts.
- Resume prunes duplicate CSV steps and preserves best-validation state.
- CSVs record total plus five losses; validation intervals generate loss plots.
- Inference exports raw/normalized text, tokens, speaker/checkpoint IDs, log/integer durations, F0 and energy.

## Submission documentation revision

- Replaced the static report with an evidence-driven ReportLab generator, vector architecture/workflow diagrams, CSV loss tables and an analysis page for every test sentence.
- Preserved original CSV histories under Results/<Language>/metrics/ so tables can be regenerated without ignored local logs.
- Corrected selected checkpoints: English step 8,000/speaker 14; Telugu step 13,000/speaker 0. These match minima in the supplied validation histories.
- Removed unsupported claims about exact final data counts, guaranteed pronunciation, fixed phone counts and completely unchanged model code.
- Added model_artifacts.json, a standard-library submission audit, separate report dependencies and a page-rendering utility.
- Corrected README preparation/alignment/inference commands; distinguished dependency lower bounds from an environment lock.
- Download access, actual final manifests/run metadata and listening observations remain explicit requirements.

## Assignment review and presentation revision (4 November 2026)

- Inference now fails on missing/untrained weights and inconsistent checkpoint steps; result replacement requires `--overwrite`. TextDataset initializes its cache. Masks follow the input tensor device.
- Entry points resolve YAML paths relative to the config file. New training runs fingerprint feature bytes, metadata, source modules and model/optimizer settings; fresh-run overwrite and incompatible/legacy resume are refused.
- Telugu preparation shares Unicode word boundaries with inference, including dotted initials. Unknown-speech `spn` is rejected rather than mapped to silence. Historical phone mismatches are recorded, not silently rewritten.
- Both preparation scripts record resolved Hugging Face dataset revisions. Manifest parsing rejects malformed/duplicate IDs and primary/reserve overlap. The retained-hour floor is closer to the assignment's approximate 5h/1h target.
- Replaced archive-size/TextGrid-count restoration and destructive git checkout with an explicit evaluate-or-train notebook. Fresh alignment, selected-feature verification, 64-clip diagnostics, 5k pilots and held-out audio review precede optional production training.
- Published checksum-inventoried metadata and selected ID/hash manifests under Results. Missing reserve hashes are explicit. Frozen test text is independently checked; strict audit readiness now includes evidence coverage and matching human listening records.
- Rebuilt the 20-page report with a branching architecture, clickable contents/model/audio links, full data workflow, loss-gap charts, ten waveform/spectrogram views and per-input variance analysis. No unmeasured listening claims are added.
- README now includes Mermaid flows, expandable exact inputs/losses, all result links, valid root-directory commands, evaluator downloads and an explicit inventory of supporting files.
- Added an offline audio/plot reviewer that exports dated observations tied to audio hashes. Added focused regressions for the costly failure modes. Original checkpoint/audio/plot/CSV results are preserved.
