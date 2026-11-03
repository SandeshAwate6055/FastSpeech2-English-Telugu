import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset
from model import FastSpeech2Loss
from utils.model import get_model, get_param_num, get_vocoder
from utils.tools import log, synth_one_sample, to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_COLUMNS = [
    "step",
    "total_loss",
    "mel_loss",
    "mel_postnet_loss",
    "pitch_loss",
    "energy_loss",
    "duration_loss",
]


def prune_csv_after_step(csv_path, restore_step):
    """On resume, remove any logged steps > restore_step to prevent duplicate entries."""
    if not os.path.exists(csv_path) or restore_step <= 0:
        return
    kept_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["step"]) <= restore_step:
                kept_rows.append(row)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOSS_COLUMNS)
        writer.writeheader()
        writer.writerows(kept_rows)


def append_loss_csv(csv_path, step, losses):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LOSS_COLUMNS)
        writer.writerow([step] + [float(x) for x in losses])


def save_loss_plot(csv_path, out_png_path, title_prefix):
    if not os.path.exists(csv_path):
        return
    step_to_row = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_to_row[int(row["step"])] = row

    if not step_to_row:
        return

    steps = sorted(step_to_row.keys())
    series = {col: [float(step_to_row[s][col]) for s in steps] for col in LOSS_COLUMNS[1:]}

    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{title_prefix} Loss Curves", fontsize=14, fontweight="bold")
    titles = [
        ("total_loss", "Total Loss", "tab:blue"),
        ("mel_loss", "Mel Loss (MAE)", "tab:orange"),
        ("mel_postnet_loss", "PostNet Mel Loss (MAE)", "tab:green"),
        ("pitch_loss", "Pitch (F0) Loss (MSE)", "tab:red"),
        ("energy_loss", "Energy Loss (MSE)", "tab:purple"),
        ("duration_loss", "Duration Loss (MSE)", "tab:brown"),
    ]
    for ax, (col, label, color) in zip(axes.flat, titles):
        ax.plot(steps, series[col], color=color, linewidth=1.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_png_path, dpi=200)
    plt.close(fig)


def evaluate_with_losses(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    loss_sums = [0.0 for _ in range(6)]
    last_batch = None
    last_output = None
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                output = model(*(batch[2:]))
                losses = Loss(batch, output)
                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])
                last_batch = batch
                last_output = output

    loss_means = [loss_sum / max(1, len(dataset)) for loss_sum in loss_sums]
    message = (
        "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, "
        "Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
            *([step] + loss_means)
        )
    )

    if logger is not None and last_batch is not None and last_output is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            last_batch,
            last_output,
            vocoder,
            model_config,
            preprocess_config,
        )
        log(logger, step, losses=loss_means)
        log(logger, fig=fig, tag="Validation/step_{}_{}".format(step, tag))
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message, loss_means


def main(args, configs):
    print(f"Preparing training on device: {device} ...")

    preprocess_config, model_config, train_config = configs

    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    eff_group_size = group_size if batch_size * group_size < len(dataset) else 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size * eff_group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    vocoder = get_vocoder(model_config, device)

    for key in ["ckpt_path", "log_path", "result_path"]:
        if key in train_config["path"]:
            os.makedirs(train_config["path"][key], exist_ok=True)

    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    train_csv_path = os.path.join(train_config["path"]["log_path"], "train_losses.csv")
    val_csv_path = os.path.join(train_config["path"]["log_path"], "val_losses.csv")
    prune_csv_after_step(train_csv_path, args.restore_step)
    prune_csv_after_step(val_csv_path, args.restore_step)

    result_dir = train_config["path"]["result_path"]
    train_plot_path = os.path.join(result_dir, "training_loss.png")
    val_plot_path = os.path.join(result_dir, "validation_loss.png")
    dataset_label = preprocess_config.get("dataset", "FastSpeech2")

    step = args.restore_step + 1
    epoch = 1
    best_val_loss = float("inf")
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                output = model(*(batch[2:]))

                losses = Loss(batch, output)
                total_loss = losses[0]

                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    loss_vals = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = (
                        "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, "
                        "Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                            *loss_vals
                        )
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a", encoding="utf-8") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=loss_vals)
                    append_loss_csv(train_csv_path, step, loss_vals)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message, val_loss_vals = evaluate_with_losses(
                        model, step, configs, val_logger, vocoder
                    )
                    with open(os.path.join(val_log_path, "log.txt"), "a", encoding="utf-8") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    append_loss_csv(val_csv_path, step, val_loss_vals)
                    save_loss_plot(train_csv_path, train_plot_path, f"{dataset_label} Training")
                    save_loss_plot(val_csv_path, val_plot_path, f"{dataset_label} Validation")

                    if val_loss_vals[0] < best_val_loss:
                        best_val_loss = val_loss_vals[0]
                        torch.save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                                "step": step,
                                "val_loss": best_val_loss,
                            },
                            os.path.join(train_config["path"]["ckpt_path"], "best_val.pth.tar"),
                        )
                    model.train()

                if step % save_step == 0 or step == total_step:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                            "step": step,
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step >= total_step:
                    save_loss_plot(train_csv_path, train_plot_path, f"{dataset_label} Training")
                    save_loss_plot(val_csv_path, val_plot_path, f"{dataset_label} Validation")
                    print(f"Training complete at step {step}. Plots saved to {result_dir}.")
                    return
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastSpeech 2 Training")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to unified config.yaml",
    )
    parser.add_argument("-p", "--preprocess_config", type=str, default=None)
    parser.add_argument("-m", "--model_config", type=str, default=None)
    parser.add_argument("-t", "--train_config", type=str, default=None)
    args = parser.parse_args()

    if args.preprocess_config and args.model_config and args.train_config:
        preprocess_config = yaml.load(
            open(args.preprocess_config, "r", encoding="utf-8"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(
            open(args.model_config, "r", encoding="utf-8"), Loader=yaml.FullLoader
        )
        train_config = yaml.load(
            open(args.train_config, "r", encoding="utf-8"), Loader=yaml.FullLoader
        )
    else:
        unified = yaml.load(
            open(args.config, "r", encoding="utf-8"), Loader=yaml.FullLoader
        )
        preprocess_config = model_config = train_config = unified

    configs = (preprocess_config, model_config, train_config)
    main(args, configs)
