#!/usr/bin/env python3
"""
Preprocess THINGS-EEG into the project's RSVP npz format.

Pipeline:
1. Read BrainVision raw EEG (.vhdr + .eeg + .vmrk) and events.tsv per subject
2. Optional channel removal
3. Band-pass filter (default 0.3-40 Hz)
4. Common average reference (CAR)
5. Stimulus-locked epoch extraction [-0.2, 1.0] s
6. Baseline correction using [-0.2, 0.0] s
7. Keep the post-stimulus 0~1000 ms segment
8. Resample to 128 Hz and apply per-trial, per-channel z-score
9. Save final npz into Standard_128Hz

Output layout (only the final training-ready version is kept):

data/THINGS/
  Standard_128Hz/sub1.npz
  subjects_summary.csv
  preprocess_config.json

Each npz contains:
  x_data: (N, C, T)
  y_data: (N,)
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy import signal


OUTPUT_ROOT_DEFAULT = Path("data/THINGS")


def subject_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("-")[1])
    except Exception:
        return 10**9


def iter_subject_dirs(raw_root: Path) -> list[Path]:
    return sorted(
        [p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("sub-")],
        key=subject_sort_key,
    )


def save_npz(folder: Path, file_name: str, x_data: np.ndarray, y_data: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{file_name}.npz", x_data=x_data, y_data=y_data)


def zscore_per_trial_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # x: (N, C, T)
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    std = np.maximum(std, eps)
    return ((x - mean) / std).astype(np.float32)


def resample_trials(x: np.ndarray, original_fs: float, new_fs: int) -> np.ndarray:
    if int(round(original_fs)) == int(new_fs):
        return x.astype(np.float32, copy=False)
    n_time = x.shape[-1]
    n_new = int(round(n_time * float(new_fs) / float(original_fs)))
    x_rs = signal.resample(x, n_new, axis=-1)
    return x_rs.astype(np.float32)


def channel_name_list(raw: mne.io.BaseRaw) -> list[str]:
    return [str(ch) for ch in raw.ch_names]


def build_events_from_tsv(events_tsv: Path, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(events_tsv, sep="\t")
    if "istarget" not in df.columns:
        raise ValueError(f"{events_tsv} has no 'istarget' column.")
    # THINGS-EEG stores onset in sample units already (BIDS-side helper timing
    # columns like time_stimon/time_stimoff are in seconds). Multiplying by
    # sfreq would overflow event sample indices, so we use onset directly.
    onset_samples = np.round(df["onset"].to_numpy(dtype=float)).astype(int)
    labels = df["istarget"].to_numpy(dtype=int)
    events = np.column_stack([onset_samples, np.zeros_like(onset_samples), labels]).astype(int)
    return events, labels


def preprocess_subject_original_fs(
    subject_dir: Path,
    low_hz: float,
    high_hz: float,
    tmin: float,
    tmax: float,
    drop_channels: Iterable[str] | None = None,
    remove_baseline: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    eeg_dir = subject_dir / "eeg"
    vhdr_files = sorted(eeg_dir.glob("*_eeg.vhdr"))
    tsv_files = sorted(eeg_dir.glob("*_events.tsv"))
    if not vhdr_files or not tsv_files:
        raise FileNotFoundError(f"Missing .vhdr or _events.tsv in {eeg_dir}")

    raw = mne.io.read_raw_brainvision(vhdr_files[0], preload=True, verbose="ERROR")
    original_sfreq = float(raw.info["sfreq"])

    if drop_channels:
        existing = [ch for ch in drop_channels if ch in raw.ch_names]
        if existing:
            raw.drop_channels(existing)

    raw.filter(l_freq=low_hz, h_freq=high_hz, method="iir", verbose="ERROR")
    raw.set_eeg_reference(ref_channels="average", verbose="ERROR")

    events, labels = build_events_from_tsv(tsv_files[0], original_sfreq)
    # MNE event ids should be positive integers.
    events_mne = events.copy()
    events_mne[:, 2] = labels + 1
    event_id = {"nontarget": 1, "target": 2}

    effective_tmax = tmax - (1.0 / original_sfreq)

    epochs = mne.Epochs(
        raw,
        events_mne,
        event_id=event_id,
        tmin=tmin,
        tmax=effective_tmax,
        baseline=(tmin, 0.0) if remove_baseline else None,
        preload=True,
        verbose="ERROR",
        reject_by_annotation=False,
    )

    # Match the existing local pipeline: baseline-correct with [-0.2, 0.0] and
    # keep only the post-stimulus 0~1000 ms segment in the saved npz.
    epochs.crop(tmin=0.0, tmax=effective_tmax)

    x = epochs.get_data(copy=True).astype(np.float32)  # (N, C, T=original_fs)
    y = (epochs.events[:, 2] - 1).astype(np.int32)

    meta = {
        "subject": subject_dir.name,
        "original_sfreq": original_sfreq,
        "n_trials": int(x.shape[0]),
        "n_channels": int(x.shape[1]),
        "n_timepoints": int(x.shape[2]),
        "n_target": int(np.sum(y == 1)),
        "n_nontarget": int(np.sum(y == 0)),
        "channels": channel_name_list(raw),
        "vhdr_file": str(vhdr_files[0]),
        "events_tsv": str(tsv_files[0]),
    }
    return x, y, meta


def preprocess_subject_standard_chunked(
    subject_dir: Path,
    low_hz: float,
    high_hz: float,
    tmin: float,
    tmax: float,
    drop_channels: Iterable[str] | None = None,
    remove_baseline: bool = True,
    new_fs: int = 128,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, dict]:
    eeg_dir = subject_dir / "eeg"
    vhdr_files = sorted(eeg_dir.glob("*_eeg.vhdr"))
    tsv_files = sorted(eeg_dir.glob("*_events.tsv"))
    if not vhdr_files or not tsv_files:
        raise FileNotFoundError(f"Missing .vhdr or _events.tsv in {eeg_dir}")

    raw = mne.io.read_raw_brainvision(vhdr_files[0], preload=True, verbose="ERROR")
    original_sfreq = float(raw.info["sfreq"])

    if drop_channels:
        existing = [ch for ch in drop_channels if ch in raw.ch_names]
        if existing:
            raw.drop_channels(existing)

    raw.filter(l_freq=low_hz, h_freq=high_hz, method="iir", verbose="ERROR")
    raw.set_eeg_reference(ref_channels="average", verbose="ERROR")

    events, labels = build_events_from_tsv(tsv_files[0], original_sfreq)
    events_mne = events.copy()
    events_mne[:, 2] = labels + 1
    event_id = {"nontarget": 1, "target": 2}
    effective_tmax = tmax - (1.0 / original_sfreq)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    n_events = len(events_mne)

    for start in range(0, n_events, chunk_size):
        end = min(start + chunk_size, n_events)
        chunk_events = events_mne[start:end]
        epochs = mne.Epochs(
            raw,
            chunk_events,
            event_id=event_id,
            tmin=tmin,
            tmax=effective_tmax,
            baseline=(tmin, 0.0) if remove_baseline else None,
            preload=True,
            verbose="ERROR",
            reject_by_annotation=False,
        )
        epochs.crop(tmin=0.0, tmax=effective_tmax)
        x_chunk = epochs.get_data(copy=True).astype(np.float32)
        y_chunk = (epochs.events[:, 2] - 1).astype(np.int32)
        x_chunk = resample_trials(x_chunk, original_fs=original_sfreq, new_fs=new_fs)
        x_chunk = zscore_per_trial_per_channel(x_chunk)
        x_parts.append(x_chunk)
        y_parts.append(y_chunk)
        del epochs, x_chunk, y_chunk
        gc.collect()

    x = np.concatenate(x_parts, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_parts, axis=0).astype(np.int32, copy=False)

    meta = {
        "subject": subject_dir.name,
        "original_sfreq": original_sfreq,
        "n_trials": int(x.shape[0]),
        "n_channels": int(x.shape[1]),
        "n_timepoints": int(round((tmax - 0.0) * new_fs)),
        "n_target": int(np.sum(y == 1)),
        "n_nontarget": int(np.sum(y == 0)),
        "channels": channel_name_list(raw),
        "vhdr_file": str(vhdr_files[0]),
        "events_tsv": str(tsv_files[0]),
    }
    return x, y, meta


def preprocess_and_save_standard(
    raw_root: Path,
    output_root: Path,
    low_hz: float,
    high_hz: float,
    tmin: float,
    tmax: float,
    drop_channels: list[str],
    new_fs: int,
) -> None:
    standard_dir = output_root / f"Standard_{new_fs}Hz"
    standard_dir.mkdir(parents=True, exist_ok=True)
    subjects = iter_subject_dirs(raw_root)

    for idx, subject_dir in enumerate(subjects, start=1):
        file_name = f"sub{subject_sort_key(subject_dir)}"
        output_file = standard_dir / f"{file_name}.npz"
        if output_file.exists():
            print(f"[skip] {file_name}: existing {output_file.name}")
            continue
        x, y, meta = preprocess_subject_standard_chunked(
            subject_dir=subject_dir,
            low_hz=low_hz,
            high_hz=high_hz,
            tmin=tmin,
            tmax=tmax,
            drop_channels=drop_channels,
            remove_baseline=True,
            new_fs=new_fs,
        )
        save_npz(standard_dir, file_name, x, y)
        print(f"[save] {file_name}: x={x.shape}, target={meta['n_target']}, nontarget={meta['n_nontarget']}")

    summary_rows: list[dict] = []
    for idx, subject_dir in enumerate(subjects, start=1):
        file_name = f"sub{subject_sort_key(subject_dir)}"
        output_file = standard_dir / f"{file_name}.npz"
        if not output_file.exists():
            continue
        with np.load(output_file) as data:
            x = data["x_data"]
            y = data["y_data"]
            summary_rows.append(
                {
                    "subject": subject_dir.name,
                    "original_sfreq": None,
                    "n_trials": int(x.shape[0]),
                    "n_channels": int(x.shape[1]),
                    "n_timepoints": None,
                    "n_target": int(np.sum(y == 1)),
                    "n_nontarget": int(np.sum(y == 0)),
                    "channels": None,
                    "vhdr_file": str(next((subject_dir / "eeg").glob("*_eeg.vhdr"))),
                    "events_tsv": str(next((subject_dir / "eeg").glob("*_events.tsv"))),
                    "output_file": str(output_file),
                    "subject_index": idx,
                    "final_sfreq": int(new_fs),
                    "final_timepoints": int(x.shape[-1]),
                }
            )

    pd.DataFrame(summary_rows).to_csv(output_root / "subjects_summary.csv", index=False)
    with open(output_root / "preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_root": str(raw_root),
                "output_root": str(output_root),
                "low_hz": low_hz,
                "high_hz": high_hz,
                "tmin": tmin,
                "tmax": tmax,
                "drop_channels": drop_channels,
                "baseline": [tmin, 0.0],
                "poststim_only": True,
                "final_fs": int(new_fs),
                "saved_layout": f"Standard_{new_fs}Hz only",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess THINGS-EEG into RSVP npz format")
    parser.add_argument("--raw-root", type=Path, required=True, help="THINGS-EEG raw OpenNeuro root (ds003825)")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT, help="Output dataset root")
    parser.add_argument("--low-hz", type=float, default=0.3, help="Band-pass low cutoff")
    parser.add_argument("--high-hz", type=float, default=40.0, help="Band-pass high cutoff")
    parser.add_argument("--tmin", type=float, default=-0.2, help="Epoch start in seconds")
    parser.add_argument("--tmax", type=float, default=1.0, help="Epoch end in seconds")
    parser.add_argument("--new-fs", type=int, default=128, help="Resample rate for final output")
    parser.add_argument(
        "--drop-channels",
        nargs="*",
        default=[],
        help="Optional channel names to drop before preprocessing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_and_save_standard(
        raw_root=args.raw_root,
        output_root=args.output_root,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        tmin=args.tmin,
        tmax=args.tmax,
        drop_channels=args.drop_channels,
        new_fs=args.new_fs,
    )
    print("finish")


if __name__ == "__main__":
    main()
