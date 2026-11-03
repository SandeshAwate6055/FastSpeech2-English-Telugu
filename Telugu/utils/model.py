import os
import json
import zipfile

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if getattr(args, "restore_step", 0):
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, getattr(args, "restore_step", 0)
        )
        if getattr(args, "restore_step", 0):
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def _ensure_hifigan_ckpt(hifigan_dir, filename):
    ckpt_path = os.path.join(hifigan_dir, filename)
    if os.path.exists(ckpt_path):
        return ckpt_path
    zip_path = ckpt_path + ".zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(hifigan_dir)
        if os.path.exists(ckpt_path):
            return ckpt_path
    raise FileNotFoundError(
        f"HiFi-GAN checkpoint not found at {ckpt_path}. "
        f"Extract {filename}.zip from ming024/FastSpeech2/hifigan into {hifigan_dir}."
    )


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        hifigan_dir = os.path.join(BASE_DIR, "hifigan")
        with open(os.path.join(hifigan_dir, "config.json"), "r") as f:
            hifi_config = json.load(f)
        hifi_config = hifigan.AttrDict(hifi_config)
        vocoder = hifigan.Generator(hifi_config)
        if speaker == "LJSpeech":
            ckpt_file = _ensure_hifigan_ckpt(hifigan_dir, "generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt_file = _ensure_hifigan_ckpt(hifigan_dir, "generator_universal.pth.tar")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
