"""Fail before training when selected arrays, manifests, hours or statistics disagree."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np
from run_support import sha256


def validate(language_dir, preprocessed=None, raw_path=None):
    language_dir = Path(language_dir).resolve()
    directory = Path(preprocessed or language_dir / "preprocessed_data").resolve()
    raw_directory = Path(raw_path or language_dir / "raw_data")
    sys.path.insert(0, str(language_dir))
    from text import text_to_sequence
    if language_dir.name == "Telugu":
        from text.cleaners import telugu_text_to_phones
    stats = json.loads((directory / "stats.json").read_text(encoding="utf-8"))
    speaker_map = json.loads((directory / "speakers.json").read_text(encoding="utf-8"))
    if set(speaker_map.values()) != set(range(len(speaker_map))):
        raise ValueError("Speaker IDs must be contiguous from zero")
    raw = {}
    for name in ("train_filelist.txt", "val_filelist.txt", "reserve_train_filelist.txt", "reserve_val_filelist.txt"):
        path = raw_directory / name
        if not path.is_file():
            raise FileNotFoundError(f"New runs require all primary/reserve manifests: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            uid, speaker, _, digest, _ = line.split("|", 4)
            if len(digest) != 40 or uid in raw:
                raise ValueError(f"Missing hash or duplicated source ID: {uid}")
            raw[uid] = (speaker, digest, "val" if "val" in name else "train")
    splits = {}
    train_values = {"pitch": [], "energy": []}
    train_clean = {"pitch": [], "energy": []}
    for split in ("train", "val"):
        ids, hashes, speakers, frames = set(), set(), set(), 0
        lines = (directory / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        for line in lines:
            uid, speaker, phone_text, text = line.split("|", 3)
            if uid in ids or uid not in raw or raw[uid][0] != speaker or raw[uid][2] != split or speaker not in speaker_map:
                raise ValueError(f"Split membership mismatch: {split}/{uid}")
            ids.add(uid); hashes.add(raw[uid][1]); speakers.add(speaker)
            with wave.open(str(raw_directory / speaker / f"{uid}.wav"), "rb") as audio:
                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (22050, 1, 2):
                    raise ValueError(f"Raw WAV format mismatch: {uid}")
                audio_hash = hashlib.sha1(audio.readframes(audio.getnframes())).hexdigest()
            if audio_hash != raw[uid][1]:
                raise ValueError(f"Raw PCM SHA-1 disagrees with the source manifest: {uid}")
            phones = phone_text.strip("{}").split()
            if "spn" in phones or len(text_to_sequence(phone_text, [])) != len(phones):
                raise ValueError(f"Unsupported/unknown phone: {uid}")
            if language_dir.name == "Telugu":
                expected = [p for p in telugu_text_to_phones(text) if p not in ("sp", "sil")]
                actual = [p for p in phones if p not in ("sp", "sil")]
                if expected != actual:
                    raise ValueError(f"Telugu alignment/frontend phones disagree: {uid}; inspect punctuation and OOVs")
            arrays = {kind: np.load(directory / kind / f"{speaker}-{kind}-{uid}.npy", allow_pickle=False) for kind in ("mel", "pitch", "energy", "duration")}
            mel, duration = arrays["mel"], arrays["duration"]
            if mel.ndim != 2 or mel.shape[1] != 80 or not len(mel):
                raise ValueError(f"Invalid mel shape: {uid}")
            if any(not np.all(np.isfinite(values)) for values in arrays.values()):
                raise ValueError(f"Non-finite feature: {uid}")
            if any(len(arrays[kind]) != len(phones) for kind in ("pitch", "energy", "duration")):
                raise ValueError(f"Token/feature length mismatch: {uid}")
            if np.any(duration < 0) or np.any(duration != np.floor(duration)) or int(duration.sum()) != len(mel):
                raise ValueError(f"Duration/mel mismatch: {uid}")
            frames += len(mel)
            if split == "train":
                for kind in ("pitch", "energy"):
                    values = arrays[kind]
                    train_values[kind].append(values)
                    lo, hi = np.percentile(values, [25, 75])
                    cleaned = values[(values > lo - 1.5 * (hi-lo)) & (values < hi + 1.5 * (hi-lo))]
                    if len(cleaned):
                        train_clean[kind].append(cleaned)
        splits[split] = {"utterances": len(ids), "hours": frames * 256 / 22050 / 3600, "ids": ids, "hashes": hashes, "speakers": speakers}
        if len(hashes) != len(ids):
            raise ValueError(f"Duplicate audio within {split}")
    train, val = splits["train"], splits["val"]
    if train["ids"] & val["ids"] or train["hashes"] & val["hashes"] or val["speakers"] - train["speakers"]:
        raise ValueError("ID/audio overlap or unseen validation speaker")
    if not 4.5 <= train["hours"] <= 5.5 or not .85 <= val["hours"] <= 1.2:
        raise ValueError(f"Retained-hour gate failed: {train['hours']:.3f}/{val['hours']:.3f}")
    for kind in ("pitch", "energy"):
        values = np.concatenate(train_values[kind])
        cleaned = np.concatenate(train_clean[kind])
        lo, hi, mean, std = stats[kind]
        if not np.isfinite([lo, hi, mean, std]).all() or std <= 0:
            raise ValueError(f"Invalid {kind} statistics")
        if not np.allclose([values.min(), values.max()], [lo, hi], atol=1e-4):
            raise ValueError(f"{kind} quantization bounds do not match selected training arrays")
        if abs(float(cleaned.mean())) > 1e-3 or abs(float(cleaned.std()) - 1) > 1e-3:
            raise ValueError(f"{kind} scaler is inconsistent with selected training targets")
    output = {"passed": True, "language": language_dir.name, "splits": {key: {"utterances": row["utterances"], "hours": row["hours"], "speakers": len(row["speakers"])} for key, row in splits.items()}, "id_overlap": 0, "audio_sha1_overlap": 0, "missing_audio_sha1": 0, "metadata_sha256": {name: sha256(directory / name) for name in ("train.txt", "val.txt", "stats.json", "speakers.json")}}
    (directory / "feature_integrity.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-path", type=Path)
    parser.add_argument("--raw-path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.language_dir, args.preprocessed_path, args.raw_path), indent=2))
