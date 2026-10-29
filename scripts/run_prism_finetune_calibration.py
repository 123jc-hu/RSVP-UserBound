"""Calibrate the calibration-free PRISM model on a user's own data (fine-tuning).

For held-out user s, the LOSO checkpoint of fold s (never trained on s) is
fine-tuned on the first half of s's trials and tested on the second half, the
same split as scripts/run_within_subject_calibration.py and the LOSO
second-half BA (for CAS the halves are the two sessions).

Fixed before any result was seen:
  * calibration span: first half ("half") or up to the 40th target of the first
    half ("K40"); its last 20% (by time) is a validation block, separated from
    the fine-tuning trials by a 10-trial gap
  * every epoch uses all fine-tuning targets plus as many randomly drawn
    non-targets; batch 64; AdamW, lr 1e-4, no weight decay; whole network
  * at most 100 epochs, early stopping on validation BA with patience 10, and
    the weights with the best validation BA are evaluated

PRISM has ~18k parameters, so this runs on the CPU. For every user the script
also records the second-half BA of the unadapted checkpoint ("BA_pre") as a
check against the stored LOSO predictions.

Outputs:
  Experiments/PRISM/<DS>/within-subject/<tag>/results.csv
  Experiments/PRISM/<DS>/within-subject/<tag>/trial_predictions/sub<k>.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from paths import EXP  # noqa: E402

RUNS = {
    "THU": "prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8",
    "CAS": "prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8",
    "THINGS": "prism_things_strict44_d16_bestsofar_sb32_prewsdrop010_regionconcat_lmr_m8_montagefix",
}
VAL_FRACTION = 0.2
GAP = 10
BATCH = 64


def load_cfg(dataset: str) -> dict:
    from Utils.config import build_config, load_config

    raw = load_config(str(ROOT / "Configs" / f"{RUNS[dataset]}.yaml"))
    cfg = build_config(str(raw["model"]), raw["dataset"])
    cfg.update(dict(raw.get("overrides", {})))
    return cfg


def data_path(cfg: dict, sid: int) -> Path:
    from Data.path_resolver import resolve_dataset_dir

    return Path(resolve_dataset_dir(cfg)) / f"sub{sid}.npz"


def calibration_span(y: np.ndarray, condition: str) -> np.ndarray:
    first = np.arange(len(y) // 2)
    if condition == "half":
        return first
    k = int(condition[1:])
    pos = first[y[first] == 1]
    return np.arange(pos[k - 1] + 1) if len(pos) >= k else np.arange(0)


def predict(model, x: np.ndarray, idx: np.ndarray, chunk: int = 2000) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            logits = model(torch.from_numpy(np.asarray(x[idx[i : i + chunk]], dtype=np.float32)).unsqueeze(1))
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            out.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(out)


def ba(y: np.ndarray, prob: np.ndarray) -> float:
    pred = prob >= 0.5
    return 0.5 * (float(np.mean(pred[y == 1])) + float(np.mean(~pred[y == 0])))


def run_subject(dataset: str, sid: int, condition: str, lr: float, max_epochs: int, patience: int,
                seed: int, pred_dir: str) -> dict | None:
    import torch
    from sklearn.metrics import f1_score, roc_auc_score
    from Train.trainer import build_model

    t0 = time.time()
    torch.manual_seed(seed + sid)
    rng = np.random.default_rng(seed + sid)
    cfg = load_cfg(dataset)
    model = build_model(cfg)
    ckpt = EXP / "PRISM" / dataset / "cross-subject" / RUNS[dataset] / "checkpoints" / f"sub{sid}" / "best_ba_model--2.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    d = np.load(data_path(cfg, sid))
    x, y = d["x_data"], d["y_data"].astype(np.int64).ravel()
    test = np.arange(len(y) // 2, len(y))
    prob_pre = predict(model, x, test)

    span = calibration_span(y, condition)
    if span.size == 0:
        print(f"[{dataset} {condition}] sub{sid}: too few targets, skipped", flush=True)
        return None
    n_val = int(round(VAL_FRACTION * len(span)))
    fit, val = span[: len(span) - n_val - GAP], span[len(span) - n_val :]
    fit_pos, fit_neg = fit[y[fit] == 1], fit[y[fit] == 0]
    x_val_y = y[val]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_state, best_ba, best_epoch, wait = copy.deepcopy(model.state_dict()), ba(x_val_y, predict(model, x, val)), 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        idx = np.concatenate([fit_pos, rng.choice(fit_neg, size=min(len(fit_neg), len(fit_pos)), replace=False)])
        idx = idx[rng.permutation(len(idx))]
        for i in range(0, len(idx), BATCH):
            b = idx[i : i + BATCH]
            if len(b) < 2:
                continue
            logits = model(torch.from_numpy(np.asarray(x[b], dtype=np.float32)).unsqueeze(1))
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            loss = loss_fn(logits, torch.from_numpy(y[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_ba = ba(x_val_y, predict(model, x, val))
        if val_ba > best_ba:
            best_state, best_ba, best_epoch, wait = copy.deepcopy(model.state_dict()), val_ba, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    prob = predict(model, x, test)
    y_te = y[test]
    pred = (prob >= 0.5).astype(np.int64)
    np.savez_compressed(Path(pred_dir) / f"sub{sid}.npz", target=y_te, prediction=pred,
                        positive_probability=prob.astype(np.float32), fit_index=fit, val_index=val)
    tpr = float(np.mean(pred[y_te == 1] == 1))
    fpr = float(np.mean(pred[y_te == 0] == 1))
    row = {
        "SUB": f"SUB{sid}", "AUC": float(roc_auc_score(y_te, prob)), "BA": 0.5 * (tpr + 1 - fpr),
        "F1": float(f1_score(y_te, pred, average="weighted")), "TPR": tpr, "FPR": fpr,
        "BA_pre": ba(y_te, prob_pre), "AUC_pre": float(roc_auc_score(y_te, prob_pre)),
        "best_epoch": best_epoch, "val_BA": best_ba,
        "n_fit_targets": int(len(fit_pos)), "n_val_targets": int((x_val_y == 1).sum()),
    }
    print(f"[{dataset} {condition}] sub{sid}: BA {row['BA_pre']:.4f} -> {row['BA']:.4f} "
          f"(epoch {best_epoch}, fit targets {row['n_fit_targets']}, {time.time() - t0:.0f}s)", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="THU,CAS,THINGS")
    parser.add_argument("--conditions", default="half,K40")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    args = parser.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads_per_worker)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from run_xdawn_riemann_loso import DATASETS

    for dataset in args.datasets.split(","):
        for condition in args.conditions.split(","):
            tag = f"prism_{dataset.lower()}_finetune_{condition.lower()}_lr{args.lr:g}_s{args.seed}"
            out_dir = EXP / "PRISM" / dataset / "within-subject" / tag
            pred_dir = out_dir / "trial_predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            subjects = DATASETS[dataset]["subjects"]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {sid: pool.submit(run_subject, dataset, sid, condition, args.lr, args.max_epochs,
                                            args.patience, args.seed, str(pred_dir)) for sid in subjects}
                rows = {sid: f.result() for sid, f in futures.items()}
            df = pd.DataFrame([rows[s] for s in subjects if rows[s] is not None])
            num = df.drop(columns="SUB")
            summary = pd.DataFrame([
                {"SUB": "AVG", **num.mean().round(4).to_dict()},
                {"SUB": "STD", **num.std(ddof=1).round(4).to_dict()},
            ])
            pd.concat([df, summary], ignore_index=True).to_csv(out_dir / "results.csv", index=False)
            (out_dir / "manifest.json").write_text(json.dumps(
                {**vars(args), "dataset": dataset, "condition": condition, "tag": tag, "base_run": RUNS[dataset],
                 "val_fraction": VAL_FRACTION, "gap": GAP, "batch": BATCH,
                 "test": "second half of trials in presentation order"}, indent=1))
            print(f"[{dataset} {condition}] BA_pre={num['BA_pre'].mean():.4f} BA={num['BA'].mean():.4f} "
                  f"(n={len(df)}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
