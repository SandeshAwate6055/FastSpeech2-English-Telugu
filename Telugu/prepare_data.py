"""
Phase 2 & 3: Telugu Dataset Preparation, Phonemic Lexicon & Speaker-Stratified Splitting
(SPRINGLab/IndicTTS_Telugu)
- Uses BOTH speakers (te_female and te_male) with speaker-stratified, utterance-disjoint splits
  so we comfortably exceed 5.0h train + 1.0h val (with reserves) and zero overlap
- Filters clips (1.0s to 20.0s, clean Telugu script, SHA-1 audio deduplication)
- Resamples audio from 48 kHz to 22,050 Hz mono 16-bit WAV and writes normalized .lab transcripts
- Builds a phonemic Telugu pronunciation lexicon (lexicon/telugu-lexicon.txt) that properly
  models inherent vowel 'అ', dependent vowel signs (Gunintalu), Virama/Halant ('్' suppressing
  inherent vowel in consonant clusters/geminates), Anusvara ('ం'), and Visarga ('ః') for MFA.
"""

import argparse
import collections
import hashlib
import io
import json
import os
import random
import re

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.io import wavfile
from tqdm import tqdm

from text.cleaners import telugu_cleaners, telugu_word_to_phones, telugu_words


def main():
    parser = argparse.ArgumentParser(description="Prepare IndicTTS Telugu Dataset (Speaker-Stratified 5h/1h)")
    parser.add_argument("--raw_dir", type=str, default="./raw_data")
    parser.add_argument("--lexicon_path", type=str, default="./lexicon/telugu-lexicon.txt")
    parser.add_argument("--train_hours", type=float, default=5.0)
    parser.add_argument("--val_hours", type=float, default=1.0)
    parser.add_argument("--reserve_train_hours", type=float, default=0.75)
    parser.add_argument("--reserve_val_hours", type=float, default=0.20)
    parser.add_argument("--min_dur", type=float, default=1.0)
    parser.add_argument("--max_dur", type=float, default=20.0)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face dataset revision; resolved SHA is recorded")
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.lexicon_path), exist_ok=True)

    from huggingface_hub import HfApi
    revision = HfApi().dataset_info("SPRINGLab/IndicTTS_Telugu", revision=args.revision, token=os.environ.get("HF_TOKEN")).sha
    print("Loading SPRINGLab/IndicTTS_Telugu at revision", revision)
    ds_dict = load_dataset("SPRINGLab/IndicTTS_Telugu", revision=revision)
    split_name = list(ds_dict.keys())[0]
    ds = ds_dict[split_name]
    audio_col = "audio" if "audio" in ds.column_names else "audio_filepath"
    ds = ds.cast_column(audio_col, Audio(decode=False))

    # Group indices by speaker (te_female vs te_male) for balanced speaker-stratified split
    spk_buckets = collections.defaultdict(list)
    for idx in range(len(ds)):
        row = ds[idx]
        g_val = row.get("gender")
        spk_id = "te_female" if (g_val == 0 or str(g_val).lower() == "female") else "te_male"
        spk_buckets[spk_id].append(idx)

    random.seed(args.seed)
    train_candidates = []
    val_candidates = []
    for spk_id in sorted(spk_buckets.keys()):
        idxs = list(spk_buckets[spk_id])
        random.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) / 6.0)))
        val_part = idxs[:n_val]
        train_part = idxs[n_val:]
        for i in train_part:
            train_candidates.append((i, spk_id))
        for i in val_part:
            val_candidates.append((i, spk_id))

    # Shuffle the speaker-stratified pools; this does not enforce exact per-voice duration balance.
    random.shuffle(train_candidates)
    random.shuffle(val_candidates)

    word_lexicon = {}
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

            utt_id = f"indictts_te_{idx:05d}"
            row = ds[idx]
            raw_text = str(row.get("text") or "").strip()
            if not raw_text or len(raw_text) < 2:
                rejections.append({"id": utt_id, "reason": "empty_text"})
                continue

            cleaned_text = telugu_cleaners(raw_text)
            words = telugu_words(cleaned_text)
            if not words:
                rejections.append({"id": utt_id, "reason": "no_telugu_words"})
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
                f.write(" ".join(words) + "\n")

            for w in words:
                if w not in word_lexicon:
                    phones = telugu_word_to_phones(w)
                    if phones:
                        word_lexicon[w] = " ".join(phones)

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

    val_recs, res_val_recs, val_sec, res_val_sec = process_candidate_pool(
        val_candidates,
        args.val_hours * 3600.0,
        args.reserve_val_hours * 3600.0,
        "Validation",
    )

    with open(args.lexicon_path, "w", encoding="utf-8") as f:
        for word in sorted(word_lexicon.keys()):
            f.write(f"{word}\t{word_lexicon[word]}\n")

    def _write_manifest(filename, records):
        with open(os.path.join(args.raw_dir, filename), "w", encoding="utf-8") as f:
            for utt_id, spk_id, dur, sha1, txt in records:
                clean_txt = re.sub(r"[\r\n]+", " ", str(txt)).strip()
                f.write(f"{utt_id}|{spk_id}|{dur:.3f}|{sha1}|{clean_txt}\n")

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

    stats = {
        "dataset": "SPRINGLab/IndicTTS_Telugu",
        "dataset_revision": revision,
        "language": "Telugu",
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
        "train_speakers": len({r[1] for r in train_recs}),
        "val_speakers": len({r[1] for r in val_recs}),
        "lexicon_words": len(word_lexicon),
        "id_overlap": len(all_train_ids & all_val_ids),
        "sha1_overlap": len(all_train_hashes & all_val_hashes),
        "rejected_clips": len(rejections),
    }
    with open(os.path.join(args.raw_dir, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n=== Telugu Data Preparation Complete ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
