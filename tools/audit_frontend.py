"""Inspect historical Telugu phone targets without changing saved results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def audit_telugu(directory):
    sys.path.insert(0, str(ROOT / "Telugu"))
    from text.cleaners import telugu_text_to_phones
    records = []
    counts = {}
    for split in ("train", "val"):
        count = 0
        for line in (directory / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            uid, _, phone_text, text = line.split("|", 3)
            actual = [p for p in phone_text.strip("{}").split() if p not in ("sp", "sil", "spn")]
            expected = [p for p in telugu_text_to_phones(text) if p not in ("sp", "sil", "spn")]
            count += 1
            if actual != expected:
                records.append({"id": uid, "split": split, "saved_phones": len(actual), "frontend_phones": len(expected)})
        counts[split] = count
    return {"language": "Telugu", "checked": counts, "mismatch_count": len(records), "mismatches": records, "note": "Historical targets compared after excluding silence/unknown-speech labels. Inspect source TextGrids and OOVs before retraining; this comparison alone does not establish perceptual errors."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-path", type=Path, default=ROOT / "Telugu/preprocessed_data")
    parser.add_argument("--output", type=Path, default=ROOT / "Results/frontend_audit.json")
    args = parser.parse_args()
    result = audit_telugu(args.preprocessed_path)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {sum(result['checked'].values())} utterances: {result['mismatch_count']} historical mismatches")
