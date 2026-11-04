"""Fresh preparation/alignment orchestration; no old feature/archive count heuristics."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from run_support import load_config, sha256

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd=None, env=None):
    subprocess.run([str(v) for v in command], cwd=cwd or ROOT, env=env, check=True)


def prepare(language, run_dir):
    import yaml
    home = run_dir / language
    config_path = home / "config.yaml"
    if config_path.exists():
        raise RuntimeError(f"Preparation already exists at {home}. Preserve it; choose a new run directory for regeneration.")
    home.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / language / "config.yaml")
    for key, folder in [("raw_path", "raw_data"), ("preprocessed_path", "preprocessed_data"), ("ckpt_path", "checkpoints"), ("log_path", "logs"), ("result_path", "results")]:
        cfg["path"][key] = str(home / folder)
    lexicon = home / "lexicon" / ("english_svarah_mfa.dict" if language == "English" else "telugu-lexicon.txt")
    cfg["path"]["lexicon_path"] = str(lexicon)
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    command = [sys.executable, ROOT / language / "prepare_data.py", "--raw_dir", cfg["path"]["raw_path"], "--seed", "42"]
    if language == "English":
        command += ["--base_lexicon", ROOT / language / "lexicon/librispeech-lexicon.txt", "--out_lexicon", lexicon]
    else:
        command += ["--lexicon_path", lexicon]
    run(command)
    mamba = Path("/usr/local/bin/micromamba")
    if not mamba.exists():
        raise RuntimeError("Install the notebook's separate MFA environment before preparation.")
    prefix = run_dir / "mfa_env"
    env = {**os.environ, "MPLBACKEND": "Agg", "MAMBA_ROOT_PREFIX": str(run_dir / "mamba_root")}
    if not prefix.exists():
        run([mamba, "create", "-y", "-p", prefix, "-c", "conda-forge", "montreal-forced-aligner=2.2.17", "openfst=1.8.2", "kaldi=5.5.1068", "joblib<1.4", "setuptools<81", "python=3.10"], env=env)
    mfa = [mamba, "run", "-p", prefix, "mfa"]
    raw = Path(cfg["path"]["raw_path"])
    if language == "English":
        run(mfa + ["model", "download", "acoustic", "english_us_arpa"], env=env)
        acoustic = "english_us_arpa"
    else:
        train_corpus = home / "aligner_train_only"
        train_corpus.mkdir()
        for name in ("train_filelist.txt", "reserve_train_filelist.txt"):
            for line in (raw / name).read_text(encoding="utf-8").splitlines():
                uid, speaker, *_ = line.split("|", 4)
                destination = train_corpus / speaker
                destination.mkdir(exist_ok=True)
                for extension in (".wav", ".lab"):
                    os.symlink(raw / speaker / (uid + extension), destination / (uid + extension))
        acoustic = home / "telugu_acoustic.zip"
        run(mfa + ["train", train_corpus, lexicon, acoustic, "--clean", "--num_jobs", "4"], env=env)
    run(mfa + ["align", raw, lexicon, acoustic, Path(cfg["path"]["preprocessed_path"]) / "TextGrid", "--clean", "--num_jobs", "4"], env=env)
    run([sys.executable, ROOT / language / "preprocess.py", "--config", config_path])
    run([sys.executable, ROOT / "tools/validate_features.py", "--language-dir", ROOT / language, "--preprocessed-path", cfg["path"]["preprocessed_path"], "--raw-path", raw])
    archive = home / f"{language}_preprocessed.tar.gz"
    with tarfile.open(archive.with_suffix(".part"), "w:gz") as target:
        for name in ("preprocessed_data", "lexicon", "config.yaml"):
            target.add(home / name, arcname=name)
        for name in ("data_stats.json", "train_filelist.txt", "val_filelist.txt", "reserve_train_filelist.txt", "reserve_val_filelist.txt"):
            target.add(raw / name, arcname=f"raw_data/{name}")
    archive.with_suffix(".part").replace(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(sha256(archive) + "\n")
    print(f"Validated new features and backed up {archive}")


def overfit(language, run_dir):
    import yaml
    cfg = load_config(run_dir / language / "config.yaml")
    source = Path(cfg["path"]["preprocessed_path"])
    home = run_dir / language / "overfit"
    if home.exists():
        raise RuntimeError(f"Overfit diagnostic already exists: {home}; inspect it or choose a new run")
    directory = home / "features"
    directory.mkdir(parents=True)
    training = (source / "train.txt").read_text(encoding="utf-8").splitlines()[:64]
    speakers = {line.split("|", 3)[1] for line in training}
    validation = [line for line in (source / "val.txt").read_text(encoding="utf-8").splitlines() if line.split("|", 3)[1] in speakers][:16]
    if len(training) != 64 or not validation:
        raise RuntimeError("Need 64 training clips and held-out clips from trained diagnostic speakers")
    for split, lines in [("train", training), ("val", validation)]:
        (directory / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for line in lines:
            uid, speaker, *_ = line.split("|", 3)
            for kind in ("mel", "pitch", "energy", "duration"):
                (directory / kind).mkdir(exist_ok=True)
                os.symlink(source / kind / f"{speaker}-{kind}-{uid}.npy", directory / kind / f"{speaker}-{kind}-{uid}.npy")
    for name in ("stats.json", "speakers.json"):
        shutil.copy2(source / name, directory / name)
    parent_audit = json.loads((source / "feature_integrity.json").read_text(encoding="utf-8"))
    if not parent_audit["passed"]:
        raise RuntimeError("Overfit requires a validated parent dataset")
    subset_audit = {"passed": True, "diagnostic_subset": True, "parent_integrity_sha256": sha256(source / "feature_integrity.json"), "metadata_sha256": {name: sha256(directory / name) for name in ("train.txt", "val.txt", "stats.json", "speakers.json")}}
    (directory / "feature_integrity.json").write_text(json.dumps(subset_audit, indent=2) + "\n")
    cfg["diagnostic_subset"] = True
    cfg["path"].update({"preprocessed_path": str(directory), "ckpt_path": str(home / "checkpoints"), "log_path": str(home / "logs"), "result_path": str(home / "results")})
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    run([sys.executable, ROOT / language / "train.py", "--config", path, "--total_step", "2000", "--val_step", "500", "--save_step", "2000"])
    run([sys.executable, ROOT / "tools/probe_audio.py", "--language", language, "--config", path, "--ckpt-file", home / "checkpoints/2000.pth.tar", "--output", home / "audio"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "overfit"])
    parser.add_argument("language", choices=["English", "Telugu"])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else overfit)(args.language, args.run_dir.resolve())
