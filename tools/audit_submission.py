import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import wave


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("English", "Telugu")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_sentences(language):
    tree = ast.parse((ROOT / language / "inference.py").read_text(encoding="utf-8"))
    name = f"{language.upper()}_TEST_SENTENCES"
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError(f"Missing {name}")


def audit():
    errors = []
    blockers = []
    registry = json.loads((ROOT / "model_artifacts.json").read_text(encoding="utf-8"))
    result = {"languages": {}, "errors": errors, "submission_blockers": blockers}
    if not (ROOT / "Report" / "Technical_Report.pdf").is_file():
        errors.append("Missing Report/Technical_Report.pdf")
    for language in LANGUAGES:
        directory = ROOT / "Results" / language
        sentences = test_sentences(language)
        summary_path = directory / "variance_predictions_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if set(summary) != set(sentences):
            errors.append(f"{language}: summary IDs differ from inference inputs")
        actual_audio_ids = {path.stem for path in (directory / "test_generated_audios").glob("*.wav")}
        if actual_audio_ids != set(sentences):
            errors.append(f"{language}: WAV IDs differ from required inputs")
        records = []
        for sample_id, expected_text in sentences.items():
            row = summary[sample_id]
            checks = {"raw_text_matches": row["raw_text"] == expected_text}
            arrays = [row[key] for key in ("tokens", "durations_frames", "log_durations", "pitch_hz", "energy")]
            checks["aligned_array_lengths"] = len({len(values) for values in arrays}) == 1 and len(arrays[0]) > 0
            checks["finite_predictions"] = all(math.isfinite(value) for values in arrays[1:] for value in values)
            checks["valid_durations"] = all(value >= 0 and int(value) == value for value in row["durations_frames"])
            checks["frame_sum_matches"] = sum(row["durations_frames"]) == row["total_frames"]
            checks["checkpoint_matches_registry"] = row["checkpoint_step"] == registry[language]["checkpoint_step"]
            checks["speaker_matches_registry"] = row["speaker_id"] == registry[language]["speaker_id"]
            audio = directory / "test_generated_audios" / f"{sample_id}.wav"
            with wave.open(str(audio), "rb") as source:
                audio_info = {"sample_rate": source.getframerate(), "channels": source.getnchannels(), "sample_width": source.getsampwidth(), "samples": source.getnframes()}
            audio_info["seconds"] = audio_info["samples"] / audio_info["sample_rate"]
            checks["audio_format"] = audio_info["sample_rate"] == 22050 and audio_info["channels"] == 1 and audio_info["samples"] > 0
            checks["audio_frames_match"] = abs(audio_info["samples"] - row["total_frames"] * 256) <= 256
            for kind in ("duration", "pitch", "energy"):
                checks[f"{kind}_plot_exists"] = (directory / "variance_adaptor_plots" / kind / f"{sample_id}_{kind}.png").is_file()
            for check, passed in checks.items():
                if not passed:
                    errors.append(f"{language}/{sample_id}: {check}")
            records.append({"id": sample_id, "audio": audio_info, "sha256": sha256(audio), "checks": checks})
        for filename in ("training_loss.png", "validation_loss.png"):
            if not (directory / filename).is_file():
                errors.append(f"{language}: missing {filename}")
        artifact = registry[language]
        if not (ROOT / artifact["checkpoint_file"]).is_file() and not artifact["download_url"]:
            blockers.append(f"{language}: selected {artifact['checkpoint_step']}-step checkpoint has no local file or download URL")
        metadata = ROOT / artifact["metadata_directory"]
        if not all((metadata / name).is_file() for name in ("stats.json", "speakers.json")) and not artifact["metadata_download_url"]:
            blockers.append(f"{language}: matching inference metadata is unavailable")
        for filename in ("train_losses.csv", "val_losses.csv"):
            if not (ROOT / "Results" / language / "metrics" / filename).is_file() and not (ROOT / language / "logs" / filename).is_file():
                blockers.append(f"{language}: source {filename} not included; PNG loss evidence only")
        if not (metadata / "preprocess_summary.json").is_file():
            blockers.append(f"{language}: retained-hour/split evidence not included")
        result["languages"][language] = {"checkpoint_step": artifact["checkpoint_step"], "speaker_id": artifact["speaker_id"], "samples": records}
    result["artifacts_valid"] = not errors
    result["submission_ready"] = not errors and not blockers
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["errors"] or (args.strict and result["submission_blockers"]) else 0)


if __name__ == "__main__":
    main()
