import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.io import wavfile

from text import text_to_sequence
from text.cleaners import telugu_cleaners, telugu_text_to_phones
from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exact Telugu Test Sentences from Section 3 of FastSpeech2_Assignment.pdf
# Note: T3 preserves the exact broken-word spacing ("అడ  విలో") as provided.
TELUGU_TEST_SENTENCES = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
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


def preprocess_telugu(text, preprocess_config):
    norm_text = telugu_cleaners(text)
    phones = telugu_text_to_phones(text)
    if not phones:
        raise ValueError(f"Empty Telugu phoneme sequence produced for input: {text!r}")

    phones_str = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones_str, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    if len(sequence) != len(phones):
        raise ValueError(
            f"Token/ID length mismatch in Telugu frontend: {len(phones)} phones vs {len(sequence)} IDs"
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
    fig_width = max(12, n * 0.30)

    # 1. Duration plot
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(x, durations, color="teal", edgecolor="darkslategray", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=60, fontsize=8)
    ax.set_ylabel("Predicted Duration (frames, 11.6 ms/frame)")
    ax.set_xlabel("Telugu Phoneme Tokens")
    ax.set_title(f"[{sample_id}] Variance Adaptor — Predicted Duration", fontsize=11)
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
    ax.set_xlabel("Telugu Phoneme Tokens")
    ax.set_title(f"[{sample_id}] Variance Adaptor — Predicted Pitch (F0)", fontsize=11)
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
    ax.set_xlabel("Telugu Phoneme Tokens")
    ax.set_title(f"[{sample_id}] Variance Adaptor — Predicted Energy", fontsize=11)
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

    if not getattr(args, "ckpt_file", None) and (args.restore_step is None or args.restore_step == 0):
        args.restore_step = find_latest_step(train_config["path"]["ckpt_path"])
        print(f"Auto-selected checkpoint step: {args.restore_step}")

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
        else TELUGU_TEST_SENTENCES
    )

    summary_records = {}
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

    for sample_id, raw_text in sentences.items():
        print(f"\nSynthesizing [{sample_id}]: {raw_text}")
        seq, tokens, norm_text = preprocess_telugu(raw_text, preprocess_config)
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
    print(f"\nAll Telugu test samples & variance adaptor plots saved under: {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastSpeech 2 Telugu Inference & Variance Adaptor Analysis")
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--ckpt_file", type=str, default=None, help="Optional explicit checkpoint file path")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--preprocessed_path", type=str, default=None)
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--text", type=str, default=None, help="Optional single sentence override")
    parser.add_argument("--sample_id", type=str, default="custom")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        unified = yaml.load(f, Loader=yaml.FullLoader)
    if args.ckpt_path is not None:
        unified["path"]["ckpt_path"] = args.ckpt_path
    if args.result_path is not None:
        unified["path"]["result_path"] = args.result_path
    if args.preprocessed_path is not None:
        unified["path"]["preprocessed_path"] = args.preprocessed_path
    configs = (unified, unified, unified)
    run_inference(args, configs)
