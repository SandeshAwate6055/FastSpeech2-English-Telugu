"""Focused regressions for failures that can invalidate a costly training run."""
import ast
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import ensure_run_provenance, safe_extract
from run_support import load_config, sha256
from run_support import require_feature_validation
from audit_evidence import extend_audit
import restore_artifacts


def isolated_node(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class DummyModel:
    def __init__(self, *args):
        self.loaded = False

    def to(self, device): return self
    def load_state_dict(self, state): self.loaded = True
    def eval(self): return self
    def requires_grad_(self, flag): return self


class CheckpointTests(unittest.TestCase):
    def loader(self, language, checkpoint=None):
        namespace = {"os": os, "FastSpeech2": DummyModel, "torch": types.SimpleNamespace(load=lambda *a, **k: checkpoint)}
        return isolated_node(ROOT / language / "utils/model.py", "get_model", namespace)

    def test_missing_explicit_checkpoint_is_not_random_synthesis(self):
        for language in ("English", "Telugu"):
            with self.subTest(language=language), self.assertRaises(FileNotFoundError):
                self.loader(language)(types.SimpleNamespace(ckpt_file="/missing/fs2-checkpoint", restore_step=0), ({}, {}, {}), "cpu")

    def test_inference_requires_trained_weights(self):
        with self.assertRaises(RuntimeError):
            self.loader("English")(types.SimpleNamespace(ckpt_file=None, restore_step=0), ({}, {}, {}), "cpu")

    def test_checkpoint_step_cannot_be_mislabeled(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            args = types.SimpleNamespace(ckpt_file=checkpoint.name, restore_step=13000)
            with self.assertRaises(ValueError):
                self.loader("English", {"step": 8000, "model": {}})(args, ({}, {}, {}), "cpu")

    def test_text_dataset_initializes_its_cache(self):
        for language in ("English", "Telugu"):
            namespace = {"Dataset": object, "json": json, "os": os, "np": np, "text_to_sequence": lambda *a: [1]}
            cls = isolated_node(ROOT / language / "dataset.py", "TextDataset", namespace)
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "speakers.json").write_text('{"voice":0}')
                path = Path(directory, "text.txt")
                path.write_text("id|voice|{AA}|hello\n")
                obj = cls(path, {"path": {"preprocessed_path": directory}, "preprocessing": {"text": {"text_cleaners": []}}})
                self.assertEqual(obj[0][0], "id")
                self.assertEqual(obj._cache, {})


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "English/preprocessor/preprocessor.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_alignment")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "alignment", "exec"), namespace)
        self.align = namespace["get_alignment"]
        self.preprocessor = types.SimpleNamespace(sampling_rate=22050, hop_length=256)

    def interval(self, start, end, phone):
        return types.SimpleNamespace(start_time=start, end_time=end, text=phone)

    def test_empty_intervals_and_gaps_preserve_pause_frames(self):
        for intervals in ([self.interval(.1,.4,"AA"),self.interval(.4,.7,""),self.interval(.7,1.,"B")], [self.interval(.1,.4,"AA"),self.interval(.7,1.,"B")]):
            phones, durations, start, end = self.align(self.preprocessor, types.SimpleNamespace(_objects=intervals))
            self.assertEqual(phones, ["AA", "sp", "B"])
            self.assertEqual(sum(durations), round(end*22050/256)-round(start*22050/256))

    def test_overlapping_boundaries_fail(self):
        with self.assertRaises(ValueError):
            self.align(self.preprocessor, types.SimpleNamespace(_objects=[self.interval(.1,.5,"AA"),self.interval(.4,.8,"B")]))

    def test_unknown_speech_is_not_silence(self):
        with self.assertRaisesRegex(ValueError, "unknown_speech"):
            self.align(self.preprocessor, types.SimpleNamespace(_objects=[self.interval(.1,.5,"spn")]))

    def test_telugu_dotted_initials_use_shared_word_boundaries(self):
        sys.path.insert(0, str(ROOT / "Telugu"))
        from text.cleaners import telugu_words, telugu_word_to_phones, telugu_text_to_phones
        text = "సి.పార్థసారధి ఎస్.జీ.ఆర్.వై"
        expected = [p for word in telugu_words(text) for p in telugu_word_to_phones(word)]
        self.assertEqual(expected, [p for p in telugu_text_to_phones(text) if p != "sp"])


class ProvenanceTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory) / "English"
        features = base / "features"
        for sub in ("text", "features/mel", "features/pitch", "features/energy", "features/duration"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        for name in ("train.txt", "val.txt", "stats.json", "speakers.json"):
            (features / name).write_text("fixture")
        (features / "mel/clip.npy").write_bytes(b"original feature bytes")
        cfg = {"path": {"preprocessed_path": str(features), "ckpt_path": str(base / "checkpoints"), "log_path": str(base / "logs")}, "preprocessing": {}, "optimizer": {}, **{k: {} for k in ("transformer", "variance_predictor", "variance_embedding", "multi_speaker", "max_seq_len", "vocoder")}}
        return base, features, (cfg, cfg, cfg)

    def test_unchanged_resume_matches_but_changed_features_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            base, features, configs = self.fixture(directory)
            digest = ensure_run_provenance(base, configs, 0)
            self.assertEqual(ensure_run_provenance(base, configs, 1000), digest)
            (features / "mel/clip.npy").write_bytes(b"changed bytes")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                ensure_run_provenance(base, configs, 1000)

    def test_fresh_run_does_not_overwrite_existing_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            base, _, configs = self.fixture(directory)
            ensure_run_provenance(base, configs, 0)
            with self.assertRaises(RuntimeError):
                ensure_run_provenance(base, configs, 0)

    def test_legacy_resume_needs_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            base, _, configs = self.fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "no run_provenance"):
                ensure_run_provenance(base, configs, 5000)

    def test_archive_paths_cannot_escape_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("../escape.txt");member.size=1
                stream.addfile(member, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                safe_extract(archive, Path(directory) / "stage")

    def test_configuration_paths_follow_yaml_location(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("fixture")
            fake_yaml = types.SimpleNamespace(safe_load=lambda text: {"path": {"raw_path": "raw", "result_path": "../results"}})
            with patch.dict(sys.modules, {"yaml": fake_yaml}):
                result = load_config(config)
            self.assertEqual(result["path"]["raw_path"], str(Path(directory, "raw").resolve()))
            self.assertEqual(result["path"]["result_path"], str((Path(directory).parent / "results").resolve()))

    def test_stale_or_missing_feature_audit_prevents_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                require_feature_validation(directory)
            base = Path(directory)
            for name in ("train.txt", "val.txt", "stats.json", "speakers.json"):
                (base/name).write_text("fixture")
            record = {"passed": True, "splits": {"train": {"hours": 5.}, "val": {"hours": 1.}}, "metadata_sha256": {name: sha256(base/name) for name in ("train.txt", "val.txt", "stats.json", "speakers.json")}}
            (base/"feature_integrity.json").write_text(json.dumps(record))
            require_feature_validation(directory)
            (base/"train.txt").write_text("different split")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                require_feature_validation(directory)


class EvidenceTests(unittest.TestCase):
    def test_restore_accepts_git_lf_and_archived_crlf_lexicons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Results/Telugu/metadata"
            lexicon = source / "source_snapshot/lexicon/telugu-lexicon.txt"
            lexicon.parent.mkdir(parents=True)
            lexicon.write_bytes(b"word\tphone\r\n")
            for name in ("stats.json", "speakers.json", "preprocess_summary.json"):
                (source/name).write_text("{}")
            files = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*") if p.is_file()}
            (source/"bundle_manifest.json").write_text(json.dumps({"files":files}))
            existing = root / "Telugu/lexicon/telugu-lexicon.txt"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"word\tphone\n")
            with patch.object(restore_artifacts,"ROOT",root):
                restore_artifacts.restore_metadata("Telugu", {"Telugu":{"published_metadata_directory":"Results/Telugu/metadata"}})
            self.assertEqual(existing.read_bytes(), b"word\tphone\n")

    def fixture(self, directory):
        root = Path(directory)
        registry = {}
        result = {"errors": [], "submission_blockers": [], "languages": {}}
        (root / "Results").mkdir()
        (root / "Results/listening_review.json").write_text('{"samples":{}}')
        for language in ("English", "Telugu"):
            metadata = root / "Results" / language / "metadata"
            metadata.mkdir(parents=True)
            (metadata / "preprocess_summary.json").write_text(json.dumps({"summary": {"train_utterances": 1, "val_utterances": 1, "train_hours_retained": 5., "val_hours_retained": 1.}}))
            (metadata / "stats.json").write_text('{"pitch":[-1,1,100,10],"energy":[-1,1,10,1]}')
            (metadata / "selected_manifest.csv").write_text("id,speaker,split,audio_sha1\ntrain,voice,train,\nval,voice,val,\n")
            (metadata / "bundle_manifest.json").write_text(json.dumps({"files": {name: sha256(metadata/name) for name in ("stats.json", "preprocess_summary.json", "selected_manifest.csv")}}))
            metrics = metadata.parent / "metrics"
            metrics.mkdir()
            for split in ("train", "val"):
                (metrics / f"{split}_losses.csv").write_text("step,total_loss\n1000,1.0\n")
            checkpoint = root / f"{language}.pth.tar"
            checkpoint.write_bytes(b"fixture weights")
            registry[language] = {"published_metadata_directory": metadata.relative_to(root).as_posix(), "checkpoint_file": checkpoint.name, "sha256": sha256(checkpoint), "download_url": "https://example.invalid/model"}
            result["languages"][language] = {"samples": [{"id": language, "sha256": "fixture_audio_hash"}]}
        return root, registry, result

    def test_missing_reserve_hashes_and_listening_prevent_ready_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root, registry, result = self.fixture(directory)
            output = extend_audit(root, result, registry)
            self.assertTrue(output["artifacts_valid"])
            self.assertFalse(output["submission_ready"])
            self.assertEqual(len(output["submission_blockers"]), 4)
            self.assertFalse(output["remote_access_checked"])

    def test_corrupt_local_checkpoint_is_an_error_even_with_a_download_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root, registry, result = self.fixture(directory)
            (root / registry["English"]["checkpoint_file"]).write_bytes(b"wrong bytes")
            output = extend_audit(root, result, registry)
            self.assertTrue(any("checkpoint checksum mismatch" in message for message in output["errors"]))

    def test_changed_metadata_bytes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, registry, result = self.fixture(directory)
            path = root / registry["Telugu"]["published_metadata_directory"] / "stats.json"
            path.write_text('{"pitch":[-1,2,100,10],"energy":[-1,1,10,1]}')
            output = extend_audit(root, result, registry)
            self.assertTrue(any("metadata checksum mismatch" in message for message in output["errors"]))


if __name__ == "__main__":
    unittest.main()
