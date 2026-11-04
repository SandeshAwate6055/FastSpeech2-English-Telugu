"""Shared path, checksum and experiment provenance helpers (see README additions)."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    """Resolve YAML paths relative to the configuration, independent of shell cwd."""
    import yaml
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, value in config.get("path", {}).items():
        if isinstance(value, str):
            config["path"][name] = str((path.parent / value).resolve())
    return config


def feature_fingerprint(language_dir, preprocessed_path, configs):
    directory = Path(preprocessed_path)
    language_dir = Path(language_dir)
    files = [directory / name for name in ("train.txt", "val.txt", "stats.json", "speakers.json")]
    for sub in ("text", "model", "transformer", "audio", "utils", "preprocessor"):
        files += sorted((language_dir / sub).glob("*.py"))
    # Include feature bytes: a matching filename alone cannot authorize checkpoint resume.
    for kind in ("mel", "pitch", "energy", "duration"):
        files += sorted((directory / kind).glob("*.npy"))
    if not any((directory / "mel").glob("*.npy")):
        raise RuntimeError("No mel features. Restore and validate the feature bundle before training.")
    hashes = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        base = directory if path.is_relative_to(directory) else language_dir
        prefix = "features/" if base == directory else "source/"
        hashes[prefix + path.relative_to(base).as_posix()] = sha256(path)
    preprocess, model, train = configs
    settings = {
        "preprocessing": preprocess["preprocessing"],
        "model": {key: model[key] for key in ("transformer", "variance_predictor", "variance_embedding", "multi_speaker", "max_seq_len", "vocoder")},
        "optimizer": train["optimizer"],
    }
    payload = {"file_hashes": hashes, "settings": settings}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def require_feature_validation(preprocessed_path, diagnostic=False):
    directory = Path(preprocessed_path)
    record = directory / "feature_integrity.json"
    if not record.is_file():
        raise RuntimeError("Training requires a successful tools/validate_features.py audit; historical archives are not trusted by existence")
    findings = json.loads(record.read_text(encoding="utf-8"))
    if not findings.get("passed") or bool(findings.get("diagnostic_subset", False)) != bool(diagnostic):
        raise RuntimeError("Feature validation status does not match this training mode")
    for name in ("train.txt", "val.txt", "stats.json", "speakers.json"):
        if findings["metadata_sha256"].get(name) != sha256(directory / name):
            raise RuntimeError(f"Feature validation is stale: {name} changed")
    if not diagnostic:
        if not 4.5 <= findings["splits"]["train"]["hours"] <= 5.5 or not .85 <= findings["splits"]["val"]["hours"] <= 1.2:
            raise RuntimeError("Production training requires the assignment's retained-hour target")


def ensure_run_provenance(language_dir, configs, restore_step):
    """Refuse mixed features, legacy resumes and accidental fresh-run overwrites."""
    preprocess, _, train = configs
    directory = Path(train["path"]["ckpt_path"])
    record = directory / "run_provenance.json"
    existing = list(directory.glob("*.pth.tar"))
    if restore_step == 0 and (existing or record.exists()):
        raise RuntimeError(f"Existing experiment at {directory}; use a new directory or explicitly resume its step.")
    if restore_step == 0:
        log_dir = Path(train["path"]["log_path"])
        if any((log_dir / name).exists() for name in ("train_losses.csv", "val_losses.csv")):
            raise RuntimeError("Fresh training requires a new log directory; previous logs will be preserved.")
    digest, payload = feature_fingerprint(language_dir, preprocess["path"]["preprocessed_path"], configs)
    if restore_step:
        if not record.is_file():
            raise RuntimeError("Resume has no run_provenance.json. Use a new run; legacy weights remain available for inference.")
        if json.loads(record.read_text(encoding="utf-8"))["fingerprint"] != digest:
            raise RuntimeError("Resume rejected: feature bytes, text frontend or training settings changed.")
        return digest
    directory.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("torch", "torchaudio", "numpy", "librosa", "pyworld", "tgt", "datasets", "g2p-en"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(["git", "-C", str(language_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    record.write_text(json.dumps({"fingerprint": digest, "commit": commit, "python": platform.python_version(), "packages": versions, **payload}, indent=2) + "\n", encoding="utf-8")
    (directory / "resolved_configs.json").write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    return digest


def safe_extract(archive, destination):
    """Extract ordinary files/directories only, rejecting path traversal and links."""
    import tarfile
    destination = Path(destination).resolve()
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        import inspect
        options = {"filter": "data"} if "filter" in inspect.signature(source.extractall).parameters else {}
        source.extractall(destination, members=members, **options)
