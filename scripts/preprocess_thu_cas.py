"""Preprocess the THU and CAS RSVP datasets into per-subject NPZ files.

THU: Zhang et al., "A benchmark dataset for RSVP-based brain-computer interfaces",
     Front. Neurosci. 2020 (raw folders S1-S10.mat, ..., S61-S64.mat with sub<k>A/B.mat).
CAS: Zheng et al., "A cross-session dataset for collaborative brain-computer interfaces
     based on rapid serial visual presentation", Front. Neurosci. 2020 (G<g>D1.mat, G<g>D2.mat;
     each file holds the two participants of group g, so group g yields subjects 2g-1 and 2g).

Steps (Section II-B of the paper): remove the mastoid channels (THU), 4th-order zero-phase
Butterworth band-pass 0.3-40 Hz, common-average reference, epochs from -200 to 1000 ms,
baseline correction with the pre-stimulus 200 ms, post-stimulus 0-1000 ms kept, resampling
to 128 Hz, and z-scoring of every channel of every trial over time. Labels: 1 = target
(the rarer class), 0 = background. Trials keep their presentation order; for CAS the two
sessions are concatenated (session 1 first).

Output: <out_root>/<DATASET>/Standard_128Hz/sub<k>.npz with x_data [N, C, 128] and y_data [N].

usage: python scripts/preprocess_thu_cas.py --dataset THU --raw-dir <raw folder> --out-root data
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal
from sklearn import preprocessing

RAW_FS = {"THU": 250, "CAS": 1000}
THU_DROP_CHANNELS = [32, 42]  # M1 and M2


def band_pass(data: np.ndarray, fs: int, low: float = 0.3, high: float = 40.0) -> np.ndarray:
    b, a = signal.butter(4, [low * 2 / fs, high * 2 / fs], "bandpass")
    return signal.filtfilt(b, a, data, axis=-1).astype(np.float32)


def epochs(data: np.ndarray, triggers: np.ndarray, fs: int) -> np.ndarray:
    """[C, T_all] -> baseline-corrected post-stimulus epochs [N, C, fs]."""
    idx = (triggers - int(0.2 * fs))[:, None] + np.arange(int(1.2 * fs))[None, :]
    x = data[:, idx].transpose(1, 0, 2)
    x[:, :, int(0.2 * fs):] -= np.mean(x[:, :, : int(0.2 * fs)], axis=-1, keepdims=True)
    return x[:, :, int(0.2 * fs):]


def preprocess_block(data: np.ndarray, triggers: np.ndarray, fs: int, drop=None) -> np.ndarray:
    if drop is not None:
        data = np.delete(data, drop, axis=0)
    data = band_pass(data, fs)
    data = data - np.mean(data, axis=0)
    return epochs(data, triggers, fs)


def load_thu(path: Path):
    mat = sio.loadmat(path)
    blocks = [mat["EEGdata1"].astype(np.float32), mat["EEGdata2"].astype(np.float32)]
    return blocks, [mat["trigger_positions"][0], mat["trigger_positions"][1]], [mat["class_labels"][0], mat["class_labels"][1]]


def load_cas_participant(cell, n_blocks: int = 3):
    blocks = [cell[0, i] for i in range(n_blocks)]
    data = [b[:-1, :].astype(np.float32) for b in blocks]
    trig = [b[-1, :] for b in blocks]
    return data, [np.where(t != 0)[0] for t in trig], [t[t != 0] for t in trig]


def finalize(x: np.ndarray, y: np.ndarray, fs: int, new_fs: int = 128) -> tuple[np.ndarray, np.ndarray]:
    y = y.copy()
    classes = np.unique(y)
    members = [np.where(y == c)[0] for c in classes]
    target = int(np.argmin([len(members[0]), len(members[1])]))
    y[members[target]] = 1
    y[members[1 - target]] = 0
    n_new = int(x.shape[-1] * new_fs / fs)
    if x.shape[-1] != n_new:
        with ThreadPoolExecutor() as pool:
            x = np.array(list(pool.map(lambda trial: signal.resample(trial, n_new, axis=-1), x)))
    # z-score every channel of every trial over its time samples
    n, c, t = x.shape
    flat = x.transpose(2, 0, 1).reshape(t, -1)
    x = preprocessing.StandardScaler().fit_transform(flat).reshape(t, n, c).transpose(1, 2, 0)
    return x, y


def save(out_dir: Path, sid: int, x: np.ndarray, y: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"sub{sid}.npz", x_data=x, y_data=y)
    print(f"saved {out_dir / f'sub{sid}.npz'}  x={x.shape}", flush=True)


def run_thu(raw: Path, out_dir: Path, subjects: list[int]) -> None:
    fs = RAW_FS["THU"]
    for sid in subjects:
        lo = (sid - 1) // 10 * 10 + 1
        hi = lo + 9 if sid <= 60 else 64
        folder = raw / f"S{lo}-S{hi}.mat"
        xs, ys = [], []
        for part in ("A", "B"):
            blocks, trigs, labels = load_thu(folder / f"sub{sid}{part}.mat")
            for blk, trg, lab in zip(blocks, trigs, labels):
                xs.append(preprocess_block(blk, trg, fs, THU_DROP_CHANNELS))
                ys.append(lab)
        save(out_dir, sid, *finalize(np.concatenate(xs), np.concatenate(ys), fs))


def run_cas(raw: Path, out_dir: Path, groups: list[int]) -> None:
    fs = RAW_FS["CAS"]
    for g in groups:
        per = {"Sa": ([], []), "Sb": ([], [])}
        for day in ("D1", "D2"):
            mat = sio.loadmat(raw / f"G{g}{day}.mat")
            for key in ("Sa", "Sb"):
                data, trigs, labels = load_cas_participant(mat[key])
                for blk, trg, lab in zip(data, trigs, labels):
                    per[key][0].append(preprocess_block(blk, trg, fs))
                    per[key][1].append(lab)
        for offset, key in ((1, "Sa"), (0, "Sb")):
            sid = 2 * g - offset
            save(out_dir, sid, *finalize(np.concatenate(per[key][0]), np.concatenate(per[key][1]), fs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["THU", "CAS"], required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument("--subjects", type=str, default="",
                        help="comma-separated THU subjects or CAS groups (default: all)")
    args = parser.parse_args()
    out_dir = args.out_root / args.dataset / "Standard_128Hz"
    if args.dataset == "THU":
        ids = [int(s) for s in args.subjects.split(",")] if args.subjects else list(range(1, 65))
        run_thu(args.raw_dir, out_dir, ids)
    else:
        ids = [int(s) for s in args.subjects.split(",")] if args.subjects else list(range(1, 8))
        run_cas(args.raw_dir, out_dir, ids)


if __name__ == "__main__":
    main()
