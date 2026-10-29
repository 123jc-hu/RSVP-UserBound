#!/usr/bin/env python3
"""
Compute per-channel |d'(c)| in 300-700ms window for all THU source subjects.

For each subject:
  trial_amp[i, c] = mean(x[i, c, T0:T1])      # mean amplitude per trial
  mu_pos[c] = mean(trial_amp[target, c])
  mu_neg[c] = mean(trial_amp[nontarget, c])
  pooled_std[c] = sqrt(pooled variance over target + nontarget trials)
  |d'(c)| = |mu_pos[c] - mu_neg[c]| / pooled_std[c]

Output (used for Fig. 4):
  Experiments/analysis/erp_topo_bank_thu.npz
  Keys: "1", "2", ..., "64" (string-encoded subject IDs)
  Values: float32 array of shape [C=62]
"""

import numpy as np

from paths import ANALYSIS, DATA_DIRS

DATA_DIR = DATA_DIRS["THU"]
OUT_PATH = ANALYSIS / "erp_topo_bank_thu.npz"

FS = 128
T0_MS, T1_MS = 300, 700
T0 = int(round(T0_MS / 1000.0 * FS))  # 38
T1 = int(round(T1_MS / 1000.0 * FS))  # 90


def compute_abs_dprime(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute |Cohen's d| per channel in the P300 window.

    x: [N, C, T=128]
    y: [N] with values {0, 1}
    returns: [C] float32
    """
    # Mean amplitude in P300 window per trial: [N, C]
    trial_amp = x[:, :, T0:T1].mean(axis=2)

    pos = trial_amp[y == 1]   # [N_pos, C]
    neg = trial_amp[y == 0]   # [N_neg, C]
    n_pos, n_neg = pos.shape[0], neg.shape[0]

    mu_pos = pos.mean(axis=0)   # [C]
    mu_neg = neg.mean(axis=0)   # [C]

    var_pos = pos.var(axis=0, ddof=1) if n_pos > 1 else np.zeros(pos.shape[1])
    var_neg = neg.var(axis=0, ddof=1) if n_neg > 1 else np.zeros(neg.shape[1])

    # Pooled standard deviation
    pooled_var = (var_pos * (n_pos - 1) + var_neg * (n_neg - 1)) / max(n_pos + n_neg - 2, 1)
    pooled_std = np.sqrt(np.maximum(pooled_var, 1e-10))

    return (np.abs(mu_pos - mu_neg) / pooled_std).astype(np.float32)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    subject_topos: dict[str, np.ndarray] = {}
    missing = []

    for sub_id in range(1, 65):
        npz_path = DATA_DIR / f"sub{sub_id}.npz"
        if not npz_path.exists():
            missing.append(sub_id)
            continue

        d = np.load(npz_path)
        x = d["x_data"].astype(np.float32)   # [N, C, T]
        y = d["y_data"].astype(np.int64)      # [N]

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())

        topo = compute_abs_dprime(x, y)
        peak_ch = int(topo.argmax())
        subject_topos[str(sub_id)] = topo

        print(
            f"sub{sub_id:02d}: n_pos={n_pos:5d}, n_neg={n_neg:6d}, "
            f"max_d={topo.max():.3f} @ ch{peak_ch}"
        )

    np.savez(OUT_PATH, **subject_topos)
    print(f"\nSaved {len(subject_topos)} subjects → {OUT_PATH}")
    if missing:
        print(f"Missing (skipped): {missing}")


if __name__ == "__main__":
    main()
