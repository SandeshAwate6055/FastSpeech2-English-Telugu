"""Export small reviewable evidence without redistributing corpus audio/transcripts."""
import ast
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from run_support import sha256

ROOT = Path(__file__).resolve().parents[1]


def export(overwrite=False):
    tests = {}
    registry = json.loads((ROOT / "model_artifacts.json").read_text(encoding="utf-8"))
    for language in ("English", "Telugu"):
        base = ROOT / language
        metadata = ROOT / "Results" / language / "metadata"
        if (metadata / "bundle_manifest.json").exists() and not overwrite:
            raise RuntimeError("Published evidence already exists; preserve it or explicitly use --overwrite after reviewing recovered records")
        metadata.mkdir(parents=True, exist_ok=True)
        for name in ("stats.json", "speakers.json", "preprocess_summary.json"):
            shutil.copy2(base / "preprocessed_data" / name, metadata / name)
        shutil.copy2(base / "raw_data/data_stats.json", metadata / "data_stats.json")
        raw = {}
        for name in ("train_filelist.txt", "val_filelist.txt", "reserve_train_filelist.txt", "reserve_val_filelist.txt"):
            path = base / "raw_data" / name
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    uid, speaker, seconds, audio_hash, _ = line.split("|", 4)
                    raw[uid] = {"speaker": speaker, "source_seconds": float(seconds), "audio_sha1": audio_hash}
        rows = []
        for split in ("train", "val"):
            for line in (base / "preprocessed_data" / f"{split}.txt").read_text(encoding="utf-8").splitlines():
                uid, speaker, phones, text = line.split("|", 3)
                row = {"id": uid, "speaker": speaker, "split": split, "phone_count": len(phones.strip("{}").split()), "phones_sha256": hashlib.sha256(phones.encode()).hexdigest(), "transcript_sha256": hashlib.sha256(text.encode()).hexdigest(), **raw.get(uid, {})}
                row.setdefault("audio_sha1", "")
                row.setdefault("source_seconds", "")
                rows.append(row)
        columns = ["id", "speaker", "split", "phone_count", "phones_sha256", "transcript_sha256", "audio_sha1", "source_seconds"]
        with (metadata / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        # Preserve the pre-review source snapshot: future repairs do not describe historical training.
        for source in [base / "config.yaml", base / "hifigan/config.json", *sorted((base / "text").glob("*.py"))]:
            rel = source.relative_to(base)
            target = metadata / "source_snapshot" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = subprocess.check_output(["git", "show", f"e5e1f4b:{language}/{rel.as_posix()}"], cwd=ROOT)
            target.write_bytes(data)
        lexicon = "english_svarah_mfa.dict" if language == "English" else "telugu-lexicon.txt"
        target = metadata / "source_snapshot/lexicon" / lexicon
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / "lexicon" / lexicon, target)
        files = {p.relative_to(metadata).as_posix(): sha256(p) for p in sorted(metadata.rglob("*")) if p.is_file() and p.name != "bundle_manifest.json"}
        (metadata / "bundle_manifest.json").write_text(json.dumps({"source_checkout": "e5e1f4b", "note": "Recovered local metadata plus source snapshot; source checkout is not independently established as the training commit. Missing reserve audio hashes are explicit blank fields.", "files": files}, indent=2) + "\n", encoding="utf-8", newline="\n")
        registry[language]["published_metadata_directory"] = metadata.relative_to(ROOT).as_posix()
        registry[language]["vocoder_sha256"] = sha256(base / "hifigan/generator_universal.pth.tar")
        tree = ast.parse((base / "inference.py").read_text(encoding="utf-8"))
        tests[language] = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == f"{language.upper()}_TEST_SENTENCES" for t in n.targets))
    frozen = ROOT / "Results/test_sentences.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != tests:
        raise RuntimeError("Inference text differs from the visually reviewed assignment manifest; frozen text was preserved")
    frozen.write_text(json.dumps(tests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry["verification_note"] = "Local checkpoint bytes were hashed during this revision. Prior Drive accessibility claims have not been reverified in this network-restricted session."
    registry["shared_folder_access_verified"] = False
    registry["shared_folder_status"] = "Owner-supplied folder and historical download links; live access requires an independent check."
    (ROOT / "model_artifacts.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review = ROOT / "Results/listening_review.json"
    if not review.exists():
        records = {}
        for language, samples in tests.items():
            for sid in samples:
                records[sid] = {"language": language, "audio_sha256": sha256(ROOT / "Results" / language / "test_generated_audios" / f"{sid}.wav"), "reviewer": "", "reviewed_at": "", "intelligibility": "pending", "pronunciation": "", "completeness": "", "prosody": "", "notes": ""}
        review.write_text(json.dumps({"schema_version": 1, "samples": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    export(parser.parse_args().overwrite)
