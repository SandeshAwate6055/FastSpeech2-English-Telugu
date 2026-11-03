"""
Phase 2 & 3: English Dataset Preparation & Speaker-Stratified Splitting (ai4bharat/Svarah)
- Downloads Svarah (commit verified in Phase 0 audit)
- Assigns deterministic 7-field speaker group IDs (115 speaker groups across 6,656 clips)
- Filters clips (1.0s to 20.0s, clean text, valid audio energy, SHA-1 audio deduplication)
- Creates speaker-stratified, utterance-disjoint splits (~5.0h train + ~1.0h val + reserves)
  ensuring every validation speaker is present in training and zero overlap
- Resamples audio to 22,050 Hz mono 16-bit WAV and writes normalized .lab transcripts for MFA
- Generates complete ARPAbet pronunciation dictionary (lexicon/english_svarah_mfa.dict) covering
  all corpus + test words so Montreal Forced Aligner (MFA) has 0 OOV words.
"""

import argparse
import collections
import hashlib
import io
import json
import os
import random
import re
from string import punctuation

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from g2p_en import G2p
from scipy.io import wavfile
from tqdm import tqdm

from text.cleaners import english_cleaners


def read_lexicon(lex_path):
    lexicon = {}
    if not os.path.exists(lex_path):
        return lexicon
    with open(lex_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 2 and parts[0].lower() not in lexicon:
                lexicon[parts[0].lower()] = parts[1:]
    return lexicon


def build_demographic_key(row):
    fields = [
        "gender",
        "age-group",
        "primary_language",
        "native_place_state",
        "native_place_district",
        "highest_qualification",
        "job_category",
        "occupation_domain",
    ]
    return tuple(str(row.get(col) or "unk").strip().lower() for col in fields)


def main():
    parser = argparse.ArgumentParser(description="Prepare Svarah English Dataset (Speaker-Stratified 5h/1h)")
    parser.add_argument("--raw_dir", type=str, default="./raw_data")
    parser.add_argument("--base_lexicon", type=str, default="./lexicon/librispeech-lexicon.txt")
    parser.add_argument("--out_lexicon", type=str, default="./lexicon/english_svarah_mfa.dict")
    parser.add_argument("--train_hours", type=float, default=5.0)
    parser.add_argument("--val_hours", type=float, default=1.0)
    parser.add_argument("--reserve_train_hours", type=float, default=0.75)
    parser.add_argument("--reserve_val_hours", type=float, default=0.20)
    parser.add_argument("--min_dur", type=float, default=1.0)
    parser.add_argument("--max_dur", type=float, default=20.0)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_lexicon), exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")

    print("Loading ai4bharat/Svarah from Hugging Face ...")
    ds = load_dataset("ai4bharat/Svarah", split="test", token=token)
    audio_col = "audio_filepath" if "audio_filepath" in ds.column_names else "audio"
    ds = ds.cast_column(audio_col, Audio(decode=False))

    # First pass: group candidate indices by deterministic demographic speaker key
    spk_key_to_id = {}
    spk_buckets = collections.defaultdict(list)
    for idx in range(len(ds)):
        row = ds[idx]
        d_key = build_demographic_key(row)
        if d_key not in spk_key_to_id:
            spk_key_to_id[d_key] = f"svarah_spk_{len(spk_key_to_id):03d}"
        spk_id = spk_key_to_id[d_key]
        spk_buckets[spk_id].append(idx)

    # Speaker-stratified ordering: within each speaker, shuffle clips and assign ~5:1 train/val
    random.seed(args.seed)
    train_candidates = []
    val_candidates = []
    for spk_id in sorted(spk_buckets.keys()):
        idxs = list(spk_buckets[spk_id])
        random.shuffle(idxs)
        if len(idxs) == 1:
            # Single-clip speaker goes only to training so validation never has an untrained speaker
            train_candidates.append((idxs[0], spk_id))
        else:
            n_val = max(1, int(round(len(idxs) / 6.0)))
            val_part = idxs[:n_val]
            train_part = idxs[n_val:]
            for i in train_part:
                train_candidates.append((i, spk_id))
            for i in val_part:
                val_candidates.append((i, spk_id))

    random.shuffle(train_candidates)
    random.shuffle(val_candidates)

    lexicon = read_lexicon(args.base_lexicon)
    g2p = G2p()
    vocab_words = set()
    seen_hashes = set()
    rejections = []

    def process_candidate_pool(candidates, target_sec, reserve_sec, pool_name):
        primary_records = []
        reserve_records = []
        acc_sec = 0.0
        res_sec = 0.0

        for idx, spk_id in tqdm(candidates, desc=f"Preparing {pool_name}"):
            if acc_sec >= target_sec and res_sec >= reserve_sec:
                break

            utt_id = f"svarah_{idx:05d}"
            row = ds[idx]
            raw_text = str(row.get("text") or "").strip()
            if not raw_text or len(raw_text) < 2:
                rejections.append({"id": utt_id, "reason": "empty_text"})
                continue
            if "[" in raw_text or "]" in raw_text or "<" in raw_text or ">" in raw_text:
                rejections.append({"id": utt_id, "reason": "noise_tag_in_text"})
                continue

            cleaned_lab = english_cleaners(raw_text)
            alpha_words = re.findall(r"[a-z']+", cleaned_lab)
            if not alpha_words:
                rejections.append({"id": utt_id, "reason": "no_alphabetic_words"})
                continue

            audio_data = row[audio_col]
            raw_bytes = audio_data.get("bytes")
            audio_path = audio_data.get("path")
            try:
                if raw_bytes is not None:
                    wav, orig_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                elif audio_path and os.path.exists(audio_path):
                    wav, orig_sr = sf.read(audio_path, dtype="float32")
                else:
                    rejections.append({"id": utt_id, "reason": "missing_audio"})
                    continue
            except Exception as e:
                rejections.append({"id": utt_id, "reason": f"decode_error:{e}"})
                continue

            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)

            dur = len(wav) / float(orig_sr)
            if dur < args.min_dur or dur > args.max_dur:
                rejections.append({"id": utt_id, "reason": f"duration_out_of_range:{dur:.2f}"})
                continue

            if orig_sr != args.sr:
                wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=args.sr)

            peak = float(np.max(np.abs(wav)))
            if peak < 1e-3:
                rejections.append({"id": utt_id, "reason": "silent_audio"})
                continue
            wav = (wav / peak) * 0.95
            wav_int16 = (wav * 32767.0).astype(np.int16)

            audio_sha1 = hashlib.sha1(wav_int16.tobytes()).hexdigest()
            if audio_sha1 in seen_hashes:
                rejections.append({"id": utt_id, "reason": "duplicate_audio_sha1"})
                continue
            seen_hashes.add(audio_sha1)

            spk_dir = os.path.join(args.raw_dir, spk_id)
            os.makedirs(spk_dir, exist_ok=True)
            wav_out_path = os.path.join(spk_dir, f"{utt_id}.wav")
            lab_out_path = os.path.join(spk_dir, f"{utt_id}.lab")

            wavfile.write(wav_out_path, args.sr, wav_int16)
            with open(lab_out_path, "w", encoding="utf-8") as f:
                f.write(cleaned_lab + "\n")

            for w in alpha_words:
                w_clean = w.strip("'")
                if w_clean:
                    vocab_words.add(w_clean.lower())

            rec = (utt_id, spk_id, dur, audio_sha1, raw_text)
            if acc_sec < target_sec:
                primary_records.append(rec)
                acc_sec += dur
            else:
                reserve_records.append(rec)
                res_sec += dur

        return primary_records, reserve_records, acc_sec, res_sec

    train_recs, res_train_recs, train_sec, res_train_sec = process_candidate_pool(
        train_candidates,
        args.train_hours * 3600.0,
        args.reserve_train_hours * 3600.0,
        "Train",
    )

    trained_speakers = {r[1] for r in train_recs}
    # Filter validation candidates to speakers present in training
    val_candidates_filtered = [c for c in val_candidates if c[1] in trained_speakers]

    val_recs, res_val_recs, val_sec, res_val_sec = process_candidate_pool(
        val_candidates_filtered,
        args.val_hours * 3600.0,
        args.reserve_val_hours * 3600.0,
        "Validation",
    )

    # Also add words from the 5 English test sentences to the lexicon
    test_words = [
        "antidisestablishmentarianism", "movement", "gained", "considerable", "momentum",
        "parliamentary", "proceedings", "orchestra's", "conduct", "ceremony", "praised",
        "audience", "researchers", "experiment", "evaluate", "proposed", "method",
        "rhythm", "rhyth", "m", "changed", "unexpectedly", "meeting", "november",
        "andaroo", "arrive",
    ]
    for tw in test_words:
        vocab_words.add(tw.lower())

    for w in sorted(vocab_words):
        if w not in lexicon:
            phones = [p for p in g2p(w) if p.strip() and p not in punctuation]
            if phones:
                lexicon[w] = phones

    with open(args.out_lexicon, "w", encoding="utf-8") as f:
        for w in sorted(lexicon.keys()):
            f.write(f"{w}\t{' '.join(lexicon[w])}\n")

    def _write_manifest(filename, records):
        with open(os.path.join(args.raw_dir, filename), "w", encoding="utf-8") as f:
            for utt_id, spk_id, dur, sha1, txt in records:
                f.write(f"{utt_id}|{spk_id}|{dur:.3f}|{sha1}|{txt}\n")

    _write_manifest("train_filelist.txt", train_recs)
    _write_manifest("val_filelist.txt", val_recs)
    _write_manifest("reserve_train_filelist.txt", res_train_recs)
    _write_manifest("reserve_val_filelist.txt", res_val_recs)

    all_train_ids = {r[0] for r in train_recs + res_train_recs}
    all_val_ids = {r[0] for r in val_recs + res_val_recs}
    all_train_hashes = {r[3] for r in train_recs + res_train_recs}
    all_val_hashes = {r[3] for r in val_recs + res_val_recs}

    assert len(all_train_ids & all_val_ids) == 0, "ID overlap between train and val!"
    assert len(all_train_hashes & all_val_hashes) == 0, "Audio SHA1 overlap between train and val!"
    assert {r[1] for r in val_recs}.issubset(trained_speakers), "Untrained speaker in validation!"

    stats = {
        "dataset": "ai4bharat/Svarah",
        "language": "English",
        "sampling_rate": args.sr,
        "seed": args.seed,
        "duration_filter_sec": [args.min_dur, args.max_dur],
        "train_utterances": len(train_recs),
        "val_utterances": len(val_recs),
        "reserve_train_utterances": len(res_train_recs),
        "reserve_val_utterances": len(res_val_recs),
        "train_hours_pre_align": round(train_sec / 3600.0, 3),
        "val_hours_pre_align": round(val_sec / 3600.0, 3),
        "reserve_train_hours": round(res_train_sec / 3600.0, 3),
        "reserve_val_hours": round(res_val_sec / 3600.0, 3),
        "train_speakers": len(trained_speakers),
        "val_speakers": len({r[1] for r in val_recs}),
        "id_overlap": len(all_train_ids & all_val_ids),
        "sha1_overlap": len(all_train_hashes & all_val_hashes),
        "rejected_clips": len(rejections),
    }
    with open(os.path.join(args.raw_dir, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=== English Data Preparation Complete ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
