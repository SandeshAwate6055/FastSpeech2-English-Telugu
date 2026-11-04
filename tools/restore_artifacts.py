"""Restore published inference metadata and checksum-verified model artifacts."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

from run_support import sha256

ROOT = Path(__file__).resolve().parents[1]


def download_verified(url, destination, expected_hash):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected_hash:
            raise ValueError(f"Existing file has a different checksum; preserved: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            if "text/html" in response.headers.get("Content-Type", ""):
                raise RuntimeError("Drive returned a confirmation/login page. Download through the browser, place the file at the documented destination, then run verification again.")
            shutil.copyfileobj(response, stream)
        if sha256(temporary) != expected_hash:
            raise ValueError(f"Download checksum mismatch for {destination.name}; final destination was not changed")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def restore_metadata(language, registry):
    source = ROOT / registry[language]["published_metadata_directory"]
    manifest = json.loads((source / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Metadata checksum mismatch: {language}/{name}")
    target = ROOT / language / "preprocessed_data"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("stats.json", "speakers.json", "preprocess_summary.json"):
        destination = target / name
        if destination.exists() and sha256(destination) != sha256(source / name):
            raise RuntimeError(f"Existing metadata belongs to another experiment; preserved {destination}")
        shutil.copy2(source / name, destination)
    for path in (source / "source_snapshot/lexicon").glob("*"):
        destination = ROOT / language / "lexicon" / path.name
        if destination.exists():
            # Git may check out LF while a recovered archive used CRLF. Compare
            # pronunciation lines without overwriting a semantically matching file.
            if destination.read_text(encoding="utf-8").splitlines() != path.read_text(encoding="utf-8").splitlines():
                raise RuntimeError(f"Existing lexicon differs; preserved {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["English", "Telugu", "both"], default="both")
    parser.add_argument("--download", action="store_true", help="Download missing acoustic checkpoints and reference vocoder; hash every file")
    args = parser.parse_args()
    registry = json.loads((ROOT / "model_artifacts.json").read_text(encoding="utf-8"))
    languages = ("English", "Telugu") if args.language == "both" else (args.language,)
    for language in languages:
        restore_metadata(language, registry)
        item = registry[language]
        checkpoint = ROOT / item["checkpoint_file"]
        if args.download:
            download_verified(item["download_url"], checkpoint, item["sha256"])
        if checkpoint.exists() and sha256(checkpoint) != item["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
        vocoder = ROOT / language / "hifigan/generator_universal.pth.tar"
        if args.download and not vocoder.exists():
            import zipfile
            url = "https://raw.githubusercontent.com/ming024/FastSpeech2/d4e79eb52e8b01d24703b2dfc0385544092958f3/hifigan/generator_universal.pth.tar.zip"
            with tempfile.TemporaryDirectory() as stage:
                archive = Path(stage) / "vocoder.zip"
                urllib.request.urlretrieve(url, archive)
                with zipfile.ZipFile(archive) as source:
                    matches = [name for name in source.namelist() if Path(name).name == vocoder.name]
                    if len(matches) != 1:
                        raise RuntimeError("Unexpected reference vocoder archive")
                    candidate = Path(stage) / vocoder.name
                    candidate.write_bytes(source.read(matches[0]))
                    if sha256(candidate) != item["vocoder_sha256"]:
                        raise ValueError("Reference vocoder checksum mismatch")
                    shutil.copy2(candidate, vocoder)
        if vocoder.exists() and sha256(vocoder) != item["vocoder_sha256"]:
            raise ValueError(f"Vocoder checksum mismatch: {vocoder}")
        print(f"{language}: metadata verified; checkpoint {'verified' if checkpoint.exists() else 'not downloaded'}; vocoder {'verified' if vocoder.exists() else 'not downloaded'}")


if __name__ == "__main__":
    main()
