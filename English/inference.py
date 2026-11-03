import argparse
import glob
import json
import os
import re
from string import punctuation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from g2p_en import G2p
from scipy.io import wavfile

from text import text_to_sequence
from text.cleaners import english_cleaners
from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exact English Test Sentences from Section 3 of FastSpeech2_Assignment.pdf
# Note: E3 preserves the exact broken-word double space ("rhyth  m") as provided.
ENGLISH_TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth  m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}


def find_latest_step(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "*.pth.tar"))
    steps = []
    for f in files:
        base = os.path.basename(f).replace(".pth.tar", "")
        if base.isdigit():
            steps.append(int(base))
    if not steps:
        raise FileNotFoundError(f"No checkpoints (*.pth.tar) found in {ckpt_dir}")
    return max(steps)


def read_lexicon(lex_path):
    lexicon = {}
    if not os.path.exists(lex_path):
        return lexicon
    with open(lex_path, "r", encoding="utf-8") as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            if len(temp) >= 2:
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    norm_text = english_cleaners(text.rstrip(punctuation))
    lex_path = preprocess_config["path"]["lexicon_path"]
    mfa_lex_path = os.path.join(os.path.dirname(lex_path), "english_svarah_mfa.dict")
    lexicon = read_lexicon(mfa_lex_path if os.path.exists(mfa_lex_path) else lex_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", norm_text)
    for w in words:
        if not w or w.isspace():
            continue
        if w in punctuation:
            if phones and phones[-1] != "sp":
                phones.append("sp")
        elif w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += [p for p in g2p(w) if p.strip() and p not in punctuation]

    if not phones:
        raise ValueError(f"Empty phoneme sequence produced for input: {text!r}")

    phones_str = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones_str, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    if len(sequence) != len(phones):
        raise ValueError(
            f"Token/ID length mismatch in English frontend: {len(phones)} phones vs {len(sequence)} IDs"
        )
    return sequence, phones, norm_text


def save_variance_plots(
    sample_id,
    raw_text,
    norm_text,
    tokens,
    log_durations,
    durations,
    pitch_norm,
    energy_norm,
    stats,
    result_dir,
    speaker_id,
    restore_step,
):
    if not (len(tokens) == len(durations) == len(pitch_norm) == len(energy_norm)):
        raise ValueError(
            f"Length mismatch for {sample_id}: tokens={len(tokens)}, "
            f"dur={len(durations)}, pitch={len(pitch_norm)}, energy={len(energy_norm)}"
        )

    dur_dir = os.path.join(result_dir, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(result_dir, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(result_dir, "variance_adaptor_plots", "energy")
    for d in [dur_dir, pitch_dir, energy_dir]:
        os.makedirs(d, exist_ok=True)

    n = len(tokens)
    pitch_mean, pitch_std = stats["pitch"][2], stats["pitch"][3]
    energy_mean, energy_std = stats["energy"][2], stats["energy"][3]
    pitch_hz = pitch_norm * pitch_std + pitch_mean
    energy_val = energy_norm * energy_std + energy_mean

    x = np.arange(n)
    fig_width = max(12, n * 0.28)

    # 1. Duration plot
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(x, durations, color="steelblue", edgecolor="navy", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=60, fontsize=8)
    ax.set_ylabel("Predicted Duration (frames, 11.6 ms/frame)")
    ax.set_xlabel("ARPAbet Phonemes")
    ax.set_title(f"[{sample_id}] Variance Adaptor — Predicted Duration\n\"{raw_text}\"", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(dur_dir, f"{sample_id}_duration.png"), dpi=200)
    plt.close(fig)

    # 2. Pitch (F0) plot
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.plot(x, pitch_hz, marker="o", markersize=4, color="tomato", linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=60, fontsize=8)
    ax.set_ylabel("Predicted Pitch F0 (Hz)")
    ax.set_xlabel("ARPAbet Phonemes")
    ax.set_title(f"[{sample_id}] Variance Adaptor — Predicted Pitch (F0)\n\"{raw_text}\"", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(pitch_dir, f"{sample_id}_pitch.png"), dpi=200)
    plt.close(fig)

    # 3. Energy plot
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.plot(x, energy_val, marker="s", markersize=4, color="darkviolet", linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=60, fontsize=8)
    ax.set_ylabel("Predicted Energy (STFT L2 Magnitude)")
    ax.set_xlabel("ARPAbet Phonemes")
    ax.set_title(f"[{sample_id}] Variance Adaptor — Predicted Energy\n\"{raw_text}\"", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(energy_dir, f"{sample_id}_energy.png"), dpi=200)
    plt.close(fig)

    return {
        "id": sample_id,
        "raw_text": raw_text,
        "normalized_text": norm_text,
        "speaker_id": int(speaker_id),
        "checkpoint_step": int(restore_step),
        "tokens": tokens,
        "log_durations": [round(float(ld), 4) for ld in log_durations],
        "durations_frames": [int(d) for d in durations],
        "total_frames": int(np.sum(durations)),
        "pitch_hz": [round(float(p), 2) for p in pitch_hz],
        "energy": [round(float(e), 2) for e in energy_val],
    }


def run_inference(args, configs):
    preprocess_config, model_config, train_config = configs

    if args.restore_step is None or args.restore_step == 0:
        args.restore_step = find_latest_step(train_config["path"]["ckpt_path"])
        print(f"Auto-selected checkpoint step: {args.restore_step}")

    # Auto-select the speaker with the most training clips if default speaker_id=0 has sparse data
    pre_dir = preprocess_config["path"]["preprocessed_path"]
    spk_json = os.path.join(pre_dir, "speakers.json")
    train_txt = os.path.join(pre_dir, "train.txt")
    if args.speaker_id == 0 and os.path.exists(spk_json) and os.path.exists(train_txt):
        import collections
        with open(spk_json, "r", encoding="utf-8") as f:
            spk_map = json.load(f)
        counts = collections.Counter()
        with open(train_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    counts[parts[1]] += 1
        if counts:
            best_spk_name, best_cnt = counts.most_common(1)[0]
            if best_spk_name in spk_map:
                args.speaker_id = int(spk_map[best_spk_name])
                print(f"Auto-selected best-trained English speaker: {best_spk_name} (speaker_id={args.speaker_id}, {best_cnt} training clips)")

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    result_dir = train_config["path"]["result_path"]
    audio_out_dir = os.path.join(result_dir, "test_generated_audios")
    os.makedirs(audio_out_dir, exist_ok=True)

    stats_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    sentences = (
        {args.sample_id: args.text}
        if args.text is not None
        else ENGLISH_TEST_SENTENCES
    )

    summary_records = {}
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

    for sample_id, raw_text in sentences.items():
        print(f"\nSynthesizing [{sample_id}]: {raw_text}")
        seq, tokens, norm_text = preprocess_english(raw_text, preprocess_config)
        ids = [sample_id]
        raw_texts = [raw_text]
        speakers = np.array([args.speaker_id])
        texts = np.array([seq])
        text_lens = np.array([len(seq)])
        batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))
        batch = to_device(batch, device)

        with torch.no_grad():
            output = model(
                *(batch[2:]),
                p_control=args.pitch_control,
                e_control=args.energy_control,
                d_control=args.duration_control,
            )

        src_len = output[8][0].item()
        mel_len = output[9][0].item()
        mel_postnet = output[1][0:1, :mel_len].detach().transpose(1, 2)
        log_durations = output[4][0, :src_len].detach().cpu().numpy()
        durations = output[5][0, :src_len].detach().cpu().numpy()
        pitch_pred = output[2][0, :src_len].detach().cpu().numpy()
        energy_pred = output[3][0, :src_len].detach().cpu().numpy()

        lengths = np.array([mel_len * hop_length])
        wav_pred = vocoder_infer(
            mel_postnet, vocoder, model_config, preprocess_config, lengths=lengths
        )[0]

        wav_path = os.path.join(audio_out_dir, f"{sample_id}.wav")
        wavfile.write(wav_path, sampling_rate, wav_pred)
        print(f"Saved audio -> {wav_path}")

        record = save_variance_plots(
            sample_id,
            raw_text,
            norm_text,
            tokens,
            log_durations,
            durations,
            pitch_pred,
            energy_pred,
            stats,
            result_dir,
            args.speaker_id,
            args.restore_step,
        )
        summary_records[sample_id] = record

    summary_path = os.path.join(result_dir, "variance_predictions_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)
    print(f"\nAll English test samples & variance adaptor plots saved under: {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastSpeech 2 English Inference & Variance Adaptor Analysis")
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--text", type=str, default=None, help="Optional single sentence override")
    parser.add_argument("--sample_id", type=str, default="custom")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        unified = yaml.load(f, Loader=yaml.FullLoader)
    configs = (unified, unified, unified)
    run_inference(args, configs)
