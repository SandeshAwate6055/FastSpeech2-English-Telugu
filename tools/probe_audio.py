"""Generate real-mel, reference-conditioned and free-running held-out probes."""
import argparse
from pathlib import Path
import sys

from run_support import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["English", "Telugu"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt-file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / args.language))
    import numpy as np
    import torch
    from scipy.io import wavfile
    from dataset import Dataset
    from utils.model import get_model, get_vocoder, vocoder_infer
    from utils.tools import to_device
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder = get_vocoder(cfg, device)
    dataset = Dataset("val.txt", cfg, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    model = None
    if args.ckpt_file:
        checkpoint_args = argparse.Namespace(ckpt_file=args.ckpt_file, restore_step=0)
        model = get_model(checkpoint_args, (cfg, cfg, cfg), device, train=False)
    # Cover every voice for Telugu and up to three English profiles.
    speakers = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["speaker"] in speakers:
            continue
        speakers.add(sample["speaker"])
        batch = to_device(dataset.reprocess([sample], [0]), device)
        mels = {"real_mel": batch[6].transpose(1, 2)}
        if model is not None:
            with torch.no_grad():
                mels["reference_conditioned"] = model(*(batch[2:]))[1].transpose(1, 2)
                mels["free_running"] = model(*(batch[2:6]))[1].transpose(1, 2)
        for name, mel in mels.items():
            wav = vocoder_infer(mel, vocoder, cfg, cfg)[0]
            wavfile.write(args.output / f"{sample['id']}_{name}.wav", 22050, wav)
        print(sample["id"], sample["speaker"], sample["raw_text"])
        if len(speakers) >= (2 if args.language == "Telugu" else 3):
            break


if __name__ == "__main__":
    main()
