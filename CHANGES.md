# Modifications Made to `ming024/FastSpeech2`

This file documents all structural, architectural, and linguistic adaptations made to the reference implementation ([`ming024/FastSpeech2`](https://github.com/ming024/FastSpeech2), commit `d4e79eb52e8b01d24703b2dfc0385544092958f3`) for the **English (`Svarah`)** and **Telugu (`IndicTTS_Telugu`)** Text-to-Speech systems.

---

## 1. Repository Restructuring & Configuration Unification
- **Independent Language Modules (`English/` and `Telugu/`):** Restructured the codebase into two self-contained pipelines (`project/English/` and `project/Telugu/`), each containing `preprocess.py`, `train.py`, `inference.py`, `config.yaml`, and `checkpoints/`.
- **Unified `config.yaml` ([`English/config.yaml`](file:///D:/IITH_ASSIG/project/English/config.yaml), [`Telugu/config.yaml`](file:///D:/IITH_ASSIG/project/Telugu/config.yaml)):**
  - Merged the three separate configuration files in `ming024/FastSpeech2` (`preprocess.yaml`, `model.yaml`, `train.yaml`) into a single `config.yaml` per language.
  - Enabled `multi_speaker: True` for **both** languages (`115` demographic speaker groups in `Svarah`; `2` studio speakers `te_female` and `te_male` in `IndicTTS_Telugu` since neither single Telugu speaker has 6 hours alone).
  - Increased `max_seq_len` from `1000` to `2000` frames to accommodate utterances up to 20 seconds (`~1,723` frames at `22,050 Hz`, `hop_length=256`) without truncation.

## 2. Telugu Phonemic Frontend & Lexicon (`project/Telugu/`)
- **Telugu Phonemic Decomposition ([`Telugu/text/cleaners.py`](file:///D:/IITH_ASSIG/project/Telugu/text/cleaners.py)):**
  - Implemented `telugu_word_to_phones()` and `telugu_text_to_phones()` to convert Telugu words into true acoustic phoneme sequences rather than raw code points:
    - Consonants (*Hallulu*, `U+0C15–U+0C39`) emit their base consonant phone plus inherent vowel `'అ'` unless followed by a dependent vowel sign (*Gunintam*) or *Virama/Halant* (`్`, `U+0C4D`).
    - Dependent vowel signs (`ా, ి, ీ, ు, ూ, ృ, ె, ే, ై, ొ, ో, ౌ`) map to their canonical independent vowel phoneme symbols (*Achulu*: `ఆ, ఇ, ఈ, ఉ, ఊ, ఋ, ఎ, ఏ, ఐ, ఒ, ఓ, ఔ`).
    - *Virama* (`్`) suppresses the inherent vowel, accurately modeling consonant clusters (*samyukta aksharas*), geminates (e.g., `పట్టు` $\rightarrow$ `['ప', 'అ', 'ట', 'ట', 'ఉ']`), and word-final codas (e.g., `న్యూరోఫార్మకోలాజికల్`).
    - *Anusvara* (`ం`) and *Visarga* (`ః`) are preserved as distinct coda phonemes.
  - Implemented `expand_telugu_numbers()` for Telugu digits (`౦–౯`), ASCII numerals, clock times (`10:45` $\rightarrow$ `పది నలభై ఐదు`), ordinals (`12వ` $\rightarrow$ `పన్నెండవ`), and years (`2026` $\rightarrow$ `రెండు వేల ఇరవై ఆరు`).

## 3. English Text Frontend & OOV Lexicon (`project/English/`)
- **Time, Date, and Year Normalization ([`English/text/numbers.py`](file:///D:/IITH_ASSIG/project/English/text/numbers.py)):**
  - Added `_time_re` and `_expand_time()` to expand clock times and meridiems (`10:45 AM` $\rightarrow$ `ten forty five a m`) prior to lexicon and `g2p_en` grapheme-to-phoneme conversion.
- **Zero-OOV MFA Lexicon Generation ([`English/prepare_data.py`](file:///D:/IITH_ASSIG/project/English/prepare_data.py)):**
  - Automatically augments `librispeech-lexicon.txt` with `g2p_en` ARPAbet pronunciations for all out-of-vocabulary Indian-English words in `Svarah` and the test sentences (`andaroo`, `antidisestablishmentarianism`, etc.) into `lexicon/english_svarah_mfa.dict`.

## 4. Bug Fixes & Data Integrity Enforcement
- **Variance Adaptor Energy Control Fix ([`model/modules.py`](file:///D:/IITH_ASSIG/project/English/model/modules.py#L123-L125)):**
  - Fixed upstream bug in `ming024/FastSpeech2` where `get_energy_embedding()` was passed `p_control` instead of `e_control` in both `phoneme_level` and `frame_level` branches, and replaced global `device` in `LengthRegulator.LR` with `x.device`.
- **Speaker-Stratified Disjoint Splits & Deduplication ([`English/prepare_data.py`](file:///D:/IITH_ASSIG/project/English/prepare_data.py), [`Telugu/prepare_data.py`](file:///D:/IITH_ASSIG/project/Telugu/prepare_data.py)):**
  - Implemented deterministic (`seed=42`) speaker-stratified splits (~5h train + ~1h val + split-specific reserves) with waveform SHA-1 deduplication and verification that every validation speaker is present in training.
  - Removed heuristic/even duration fallbacks so training relies strictly on genuine MFA `.TextGrid` forced alignments.
- **Train-Only Normalization & Feature Invariants ([`preprocessor/preprocessor.py`](file:///D:/IITH_ASSIG/project/English/preprocessor/preprocessor.py)):**
  - Modified `StandardScaler` fitting for pitch and energy (`stats.json`) to fit **strictly on training utterances**, preventing validation data leakage.
  - Made normalization idempotent via raw feature caching and enforced strict feature invariants (`len(phone) == len(duration) == len(pitch) == len(energy)` and `sum(duration) == mel_frames`).
- **Resume-Safe Training & Strict Inference ([`train.py`](file:///D:/IITH_ASSIG/project/English/train.py), [`inference.py`](file:///D:/IITH_ASSIG/project/English/inference.py)):**
  - Updated `train.py` to prune duplicate CSV steps on `--restore_step`, save `best_val.pth.tar`, and render PNG loss plots at validation intervals (`val_step`) rather than every logging step.
  - Updated `inference.py` to preserve exact raw test strings from the PDF (including double spaces in `E3` and `T3`), assert strict token/prediction length equality, and export both `log_durations` and integer `durations_frames`.
