"""Evidence checks kept separate from the waveform/prediction structural audit."""
import csv
import json
import math

from run_support import sha256


def extend_audit(root, result, registry):
    errors, blockers = result["errors"], result["submission_blockers"]
    result["schema_version"] = 2
    result["remote_access_checked"] = False
    result["warnings"] = []
    listening = json.loads((root / "Results/listening_review.json").read_text(encoding="utf-8"))
    for language in ("English", "Telugu"):
        item = registry[language]
        metadata = root / item["published_metadata_directory"]
        bundle = json.loads((metadata / "bundle_manifest.json").read_text(encoding="utf-8"))
        for name, expected in bundle["files"].items():
            path = metadata / name
            if not path.is_file() or sha256(path) != expected:
                errors.append(f"{language}: metadata checksum mismatch: {name}")
        summary = json.loads((metadata / "preprocess_summary.json").read_text(encoding="utf-8"))["summary"]
        stats = json.loads((metadata / "stats.json").read_text(encoding="utf-8"))
        for kind in ("pitch", "energy"):
            values = stats[kind]
            if len(values) != 4 or not all(math.isfinite(v) for v in values) or values[3] <= 0 or values[0] >= values[1]:
                errors.append(f"{language}: invalid {kind} statistics")
        with (metadata / "selected_manifest.csv").open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        selections = {split: [r for r in rows if r["split"] == split] for split in ("train", "val")}
        ids = {split: {r["id"] for r in values} for split, values in selections.items()}
        hashes = {split: {r["audio_sha1"] for r in values if r["audio_sha1"]} for split, values in selections.items()}
        speakers = {split: {r["speaker"] for r in values} for split, values in selections.items()}
        for split in ("train", "val"):
            if len(ids[split]) != len(selections[split]) or len(selections[split]) != summary[f"{split}_utterances"]:
                errors.append(f"{language}: selected {split} manifest count mismatch")
        if ids["train"] & ids["val"] or hashes["train"] & hashes["val"] or speakers["val"] - speakers["train"]:
            errors.append(f"{language}: split overlap or unseen validation speaker")
        if not 4.5 <= summary["train_hours_retained"] <= 5.5 or not .85 <= summary["val_hours_retained"] <= 1.2:
            errors.append(f"{language}: reported retained hours fail target")
        missing = sum(not r["audio_sha1"] for r in rows)
        if missing:
            blockers.append(f"{language}: audio hashes missing for {missing} retained reserve utterances; complete final audio-overlap verification is pending")
        checkpoint = root / item["checkpoint_file"]
        verified = checkpoint.is_file() and sha256(checkpoint) == item["sha256"]
        if checkpoint.is_file() and not verified:
            errors.append(f"{language}: checkpoint checksum mismatch")
        if not verified:
            result["warnings"].append(f"{language}: external checkpoint URL/checksum listed; bytes and live download access not checked locally")
        if not item.get("download_url") or len(item.get("sha256", "")) != 64:
            blockers.append(f"{language}: missing checkpoint download URL or checksum")
        histories = {}
        for split in ("train", "val"):
            path = root / "Results" / language / "metrics" / f"{split}_losses.csv"
            with path.open(encoding="utf-8", newline="") as source:
                history = list(csv.DictReader(source))
            steps = [int(r["step"]) for r in history]
            if not history or steps != sorted(set(steps)) or any(not math.isfinite(float(v)) for row in history for v in row.values()):
                errors.append(f"{language}: empty/duplicate/unordered/non-finite {split} history")
            histories[split] = {"rows": len(history), "sha256": sha256(path)}
        pending = []
        for sample in result["languages"][language]["samples"]:
            sid = sample["id"]
            note = listening.get("samples", {}).get(sid, {})
            complete = note.get("audio_sha256") == sample["sha256"] and note.get("intelligibility") in ("clear", "mixed", "poor") and all(str(note.get(k, "")).strip() for k in ("reviewer", "reviewed_at", "pronunciation", "completeness", "prosody"))
            sample["listening_record_complete"] = bool(complete)
            if not complete:
                pending.append(sid)
        if pending:
            blockers.append(f"{language}: per-sentence listening observations pending: {', '.join(pending)}")
        result["languages"][language].update({"checkpoint_locally_verified": verified, "preprocessing_summary": summary, "final_id_overlap": len(ids["train"] & ids["val"]), "known_audio_sha1_overlap": len(hashes["train"] & hashes["val"]), "missing_audio_sha1": missing, "histories": histories})
    result["artifacts_valid"] = not errors
    result["submission_ready"] = not errors and not blockers
    return result
