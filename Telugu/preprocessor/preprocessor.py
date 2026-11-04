import os
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
from text import text_to_sequence


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"].get("val_size", 450)
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def _load_predefined_splits(self):
        train_list_path = os.path.join(self.in_dir, "train_filelist.txt")
        val_list_path = os.path.join(self.in_dir, "val_filelist.txt")
        reserve_train_path = os.path.join(self.in_dir, "reserve_train_filelist.txt")
        reserve_val_path = os.path.join(self.in_dir, "reserve_val_filelist.txt")

        if not (os.path.exists(train_list_path) and os.path.exists(val_list_path)):
            raise FileNotFoundError(
                f"Missing required split manifests in {self.in_dir}: "
                f"train_filelist.txt and val_filelist.txt must exist before preprocessing."
            )

        def _read_ids(p):
            if not os.path.exists(p):
                return []
            with open(p, "r", encoding="utf-8") as f:
                return [line.strip().split("|")[0] for line in f if line.strip()]

        train_ids = _read_ids(train_list_path)
        val_ids = _read_ids(val_list_path)
        res_train_ids = _read_ids(reserve_train_path)
        res_val_ids = _read_ids(reserve_val_path)

        all_train = set(train_ids) | set(res_train_ids)
        all_val = set(val_ids) | set(res_val_ids)
        overlap = all_train.intersection(all_val)
        if overlap:
            raise ValueError(f"Split manifest overlap detected ({len(overlap)} IDs)!")

        return train_ids, val_ids, res_train_ids, res_val_ids

    def build_from_path(self):
        for sub in ["mel", "pitch", "energy", "duration"]:
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

        print("Processing aligned data and enforcing feature invariants ...")
        train_ids, val_ids, res_train_ids, res_val_ids = self._load_predefined_splits()
        train_set = set(train_ids)
        val_set = set(val_ids)
        res_train_set = set(res_train_ids)
        res_val_set = set(res_val_ids)

        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        speakers = {}
        speaker_dirs = sorted(
            [
                d
                for d in os.listdir(self.in_dir)
                if os.path.isdir(os.path.join(self.in_dir, d))
            ]
        )

        processed_map = {}
        rejections = []
        raw_pitch_cache = {}
        raw_energy_cache = {}

        for i, speaker in enumerate(tqdm(speaker_dirs, desc="Speakers")):
            speakers[speaker] = i
            speaker_path = os.path.join(self.in_dir, speaker)
            for wav_name in sorted(os.listdir(speaker_path)):
                if not wav_name.endswith(".wav"):
                    continue

                basename = wav_name[:-4]
                if basename not in (train_set | val_set | res_train_set | res_val_set):
                    continue

                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if not os.path.exists(tg_path):
                    rejections.append({"id": basename, "speaker": speaker, "reason": "missing_textgrid"})
                    continue

                ret, reason = self.process_utterance(speaker, basename)
                if ret is None:
                    rejections.append({"id": basename, "speaker": speaker, "reason": reason})
                    continue

                info, pitch_clean, energy_clean, n_frames, raw_pitch, raw_energy = ret
                processed_map[basename] = (info, n_frames, speaker, pitch_clean, energy_clean)
                raw_pitch_cache[f"{speaker}-pitch-{basename}.npy"] = raw_pitch
                raw_energy_cache[f"{speaker}-energy-{basename}.npy"] = raw_energy

        # Replenish from split-specific reserves if needed to hit target hours (5.0h train, 1.0h val)
        target_train_frames = int(5.0 * 3600 * self.sampling_rate / self.hop_length)
        target_val_frames = int(1.0 * 3600 * self.sampling_rate / self.hop_length)

        train_out = []
        val_out = []
        train_frames = 0
        val_frames = 0
        train_speakers = set()
        val_speakers = set()
        selected_train_files = set()
        selected_val_files = set()

        for uid in train_ids + res_train_ids:
            if uid not in processed_map:
                continue
            if uid in res_train_set and train_frames >= target_train_frames:
                continue
            info, nf, spk, p_clean, e_clean = processed_map[uid]
            train_out.append(info)
            train_frames += nf
            train_speakers.add(spk)
            selected_train_files.add((spk, uid))
            # Fit StandardScaler STRICTLY on training examples only (no validation leakage)
            if len(p_clean) > 0:
                pitch_scaler.partial_fit(p_clean.reshape((-1, 1)))
            if len(e_clean) > 0:
                energy_scaler.partial_fit(e_clean.reshape((-1, 1)))

        for uid in val_ids + res_val_ids:
            if uid not in processed_map:
                continue
            if uid in res_val_set and val_frames >= target_val_frames:
                continue
            info, nf, spk, _, _ = processed_map[uid]
            val_out.append(info)
            val_frames += nf
            val_speakers.add(spk)
            selected_val_files.add((spk, uid))

        kept_files = selected_train_files | selected_val_files
        # Remove any mel/duration files written for unselected reserve candidates
        for uid, (_, _, spk, _, _) in processed_map.items():
            if (spk, uid) not in kept_files:
                for sub, tag in [("mel", "mel"), ("duration", "duration")]:
                    stale_path = os.path.join(self.out_dir, sub, f"{spk}-{tag}-{uid}.npy")
                    if os.path.exists(stale_path):
                        os.remove(stale_path)

        untrained_val_speakers = val_speakers - train_speakers
        if untrained_val_speakers:
            raise ValueError(
                f"Validation contains speakers not present in training: {untrained_val_speakers}"
            )

        print("Computing train-only normalization statistics ...")
        if self.pitch_normalization and hasattr(pitch_scaler, "mean_"):
            pitch_mean = float(pitch_scaler.mean_[0])
            pitch_std = float(pitch_scaler.scale_[0])
        else:
            pitch_mean, pitch_std = 0.0, 1.0

        if self.energy_normalization and hasattr(energy_scaler, "mean_"):
            energy_mean = float(energy_scaler.mean_[0])
            energy_std = float(energy_scaler.scale_[0])
        else:
            energy_mean, energy_std = 0.0, 1.0

        train_pitch_keys = {f"{spk}-pitch-{uid}.npy" for spk, uid in selected_train_files}
        kept_pitch_keys = {f"{spk}-pitch-{uid}.npy" for spk, uid in kept_files}
        train_energy_keys = {f"{spk}-energy-{uid}.npy" for spk, uid in selected_train_files}
        kept_energy_keys = {f"{spk}-energy-{uid}.npy" for spk, uid in kept_files}

        # Idempotent normalization from raw cached values; quantization bounds strictly from training split
        pitch_min, pitch_max = self.normalize_from_cache(
            os.path.join(self.out_dir, "pitch"),
            raw_pitch_cache,
            pitch_mean,
            pitch_std,
            train_keys=train_pitch_keys,
            kept_keys=kept_pitch_keys,
        )
        energy_min, energy_max = self.normalize_from_cache(
            os.path.join(self.out_dir, "energy"),
            raw_energy_cache,
            energy_mean,
            energy_std,
            train_keys=train_energy_keys,
            kept_keys=kept_energy_keys,
        )

        with open(os.path.join(self.out_dir, "speakers.json"), "w", encoding="utf-8") as f:
            json.dump(speakers, f, indent=2)

        with open(os.path.join(self.out_dir, "stats.json"), "w", encoding="utf-8") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            json.dump(stats, f, indent=2)

        train_hours = train_frames * self.hop_length / self.sampling_rate / 3600.0
        val_hours = val_frames * self.hop_length / self.sampling_rate / 3600.0
        summary = {
            "train_utterances": len(train_out),
            "val_utterances": len(val_out),
            "train_hours_retained": round(train_hours, 3),
            "val_hours_retained": round(val_hours, 3),
            "train_speakers": len(train_speakers),
            "val_speakers": len(val_speakers),
            "rejected_count": len(rejections),
        }
        with open(os.path.join(self.out_dir, "preprocess_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rejections": rejections}, f, indent=2)

        print(
            f"Retained hours -> Train: {train_hours:.2f}h ({len(train_out)} clips), "
            f"Val: {val_hours:.2f}h ({len(val_out)} clips), Rejected: {len(rejections)}"
        )

        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train_out:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val_out:
                f.write(m + "\n")

        return train_out, val_out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        try:
            # MFA 2.x writes pauses as empty intervals; tgt drops them unless include_empty_intervals=True.
            textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones")
            )
        except Exception as e:
            return None, f"invalid_textgrid:{e}"

        if not phone or start >= end or any(d < 0 for d in duration) or sum(duration) <= 0:
            return None, "empty_or_invalid_durations"

        expected_span_frames = int(
            np.round(end * self.sampling_rate / self.hop_length)
            - np.round(start * self.sampling_rate / self.hop_length)
        )
        total_dur_frames = int(sum(duration))
        if abs(total_dur_frames - expected_span_frames) > 1:
            return None, f"span_duration_mismatch:{total_dur_frames}!={expected_span_frames}"

        text = "{" + " ".join(phone) + "}"
        seq_ids = text_to_sequence(text, self.config["preprocessing"]["text"]["text_cleaners"])
        if len(seq_ids) != len(phone):
            return None, f"unsupported_phone_token:{len(seq_ids)}!={len(phone)}"

        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        if len(pitch) < total_dur_frames or (len(pitch) - total_dur_frames) > 3:
            return None, f"pre_slice_f0_span_mismatch:{len(pitch)}_vs_{total_dur_frames}"
        pitch = pitch[:total_dur_frames]
        if np.sum(pitch != 0) <= 1:
            return None, "all_unvoiced_f0"

        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        pre_slice_mel_frames = int(mel_spectrogram.shape[1])
        if pre_slice_mel_frames < total_dur_frames or (pre_slice_mel_frames - total_dur_frames) > 3:
            return None, f"pre_slice_mel_span_mismatch:{pre_slice_mel_frames}_vs_{total_dur_frames}"
        mel_spectrogram = mel_spectrogram[:, :total_dur_frames]
        energy = energy[:total_dur_frames]

        if mel_spectrogram.shape[1] != total_dur_frames:
            return None, f"mel_frame_mismatch:{mel_spectrogram.shape[1]}!={total_dur_frames}"

        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            pos = 0
            phoneme_pitch = np.zeros(len(duration), dtype=np.float32)
            for i, d in enumerate(duration):
                if d > 0:
                    phoneme_pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    phoneme_pitch[i] = 0.0
                pos += d
            pitch = phoneme_pitch

        if self.energy_phoneme_averaging:
            pos = 0
            phoneme_energy = np.zeros(len(duration), dtype=np.float32)
            for i, d in enumerate(duration):
                if d > 0:
                    phoneme_energy[i] = np.mean(energy[pos : pos + d])
                else:
                    phoneme_energy[i] = 0.0
                pos += d
            energy = phoneme_energy

        # Enforce required invariants
        if not (len(phone) == len(duration) == len(pitch) == len(energy)):
            return None, "token_feature_length_mismatch"
        if not (np.all(np.isfinite(mel_spectrogram)) and np.all(np.isfinite(pitch)) and np.all(np.isfinite(energy))):
            return None, "non_finite_features"

        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), np.array(duration, dtype=np.int64))

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
            pitch.astype(np.float32),
            energy.astype(np.float32),
        ), None

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn", ""]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0

        # Ensure contiguous monotonic intervals by rejecting overlaps and filling any gap with silence
        intervals = []
        prev_end = None
        for t in tier._objects:
            if t.end_time < t.start_time:
                raise ValueError(f"Non-monotonic interval ({t.start_time} > {t.end_time})")
            if prev_end is not None and t.start_time < prev_end - 1e-5:
                raise ValueError(f"Overlapping interval ({t.start_time} < {prev_end})")
            if prev_end is not None and t.start_time - prev_end > 1e-6:
                intervals.append((prev_end, t.start_time, ""))
            intervals.append((t.start_time, t.end_time, t.text.strip()))
            prev_end = t.end_time

        for s, e, p in intervals:

            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append("sp")

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        if len(values) == 0:
            return values
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize_from_cache(self, out_subdir, raw_cache, mean, std, train_keys=None, kept_keys=None):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        if not raw_cache:
            return -1.0, 1.0
        for filename, raw_vals in raw_cache.items():
            if kept_keys is not None and filename not in kept_keys:
                continue
            filepath = os.path.join(out_subdir, filename)
            values = (raw_vals - mean) / std
            np.save(filepath, values.astype(np.float32))
            if train_keys is None or filename in train_keys:
                max_value = max(max_value, float(np.max(values)))
                min_value = min(min_value, float(np.min(values)))
        return min_value, max_value
