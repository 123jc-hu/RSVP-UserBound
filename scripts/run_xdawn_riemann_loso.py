"""Strict LOSO xDAWN + Riemannian tangent-space baseline (classical, untuned).

Pipeline (pyRiemann): XdawnCovariances(nfilter=4, Ledoit-Wolf) -> TangentSpace
(Riemannian mean) -> LogisticRegression(class_weight="balanced").  All
hyperparameters are fixed defaults chosen before any result was seen, so no
dataset-level model selection is involved.

Protocol matches the deep baselines: the held-out subject contributes nothing
to fitting; it is evaluated on all of its trials.  Each source subject keeps
all target trials plus `--neg-ratio` x as many randomly drawn non-targets
(fixed seed), which bounds memory and mirrors background downsampling.

Memory: pyRiemann's Xdawn.fit otherwise reshapes all source trials into one
(channels x samples) array to estimate the signal covariance (~16 GB per fold
on THU).  We pass that covariance via `baseline_cov`, assembled per fold from
per-subject sufficient statistics.  It is the sample covariance of the same
concatenated source signal; with ~10^7 samples, Ledoit-Wolf shrinkage is
negligible.  Folds whose predictions already exist are not recomputed.

Outputs (same layout as the other models):
  Experiments/XdawnRiemann/<DS>/cross-subject/<tag>/results.csv
  Experiments/XdawnRiemann/<DS>/cross-subject/<tag>/trial_predictions/sub<k>.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA_DIRS, EXP, SUBJECTS

DATASETS = {ds: {"root": DATA_DIRS[ds], "subjects": SUBJECTS[ds]} for ds in ("THU", "CAS", "THINGS")}

# Filled in the parent before the pool forks; workers read it copy-on-write.
# {subject: (x float32 [N, C, T], y [N], (n_samples, sum [C], outer-product sum [C, C]))}
SOURCE_CACHE: dict[int, tuple[np.ndarray, np.ndarray, tuple]] = {}


def load_subject(dataset: str, sid: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(DATASETS[dataset]["root"] / f"sub{sid}.npz")
    return d["x_data"], d["y_data"].astype(np.int64).ravel()


def subsample_source(x: np.ndarray, y: np.ndarray, neg_ratio: int, rng: np.random.Generator):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg = rng.choice(neg, size=min(len(neg), neg_ratio * len(pos)), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    xs = np.ascontiguousarray(x[idx], dtype=np.float32)
    x64 = xs.astype(np.float64)
    stats = (x64.shape[0] * x64.shape[2], x64.sum(axis=(0, 2)), np.einsum("nct,ndt->cd", x64, x64))
    return xs, y[idx], stats


def pooled_signal_cov(subjects: list[int]) -> np.ndarray:
    n = sum(SOURCE_CACHE[s][2][0] for s in subjects)
    total = sum(SOURCE_CACHE[s][2][1] for s in subjects)
    outer = sum(SOURCE_CACHE[s][2][2] for s in subjects)
    mean = total / n
    return outer / n - np.outer(mean, mean)


def fold_metrics(test_sid: int, y_te: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    tpr = float(np.mean(pred[y_te == 1] == 1))
    fpr = float(np.mean(pred[y_te == 0] == 1))
    return {
        "SUB": f"SUB{test_sid}",
        "AUC": float(roc_auc_score(y_te, prob)),
        "BA": 0.5 * (tpr + 1.0 - fpr),
        "F1": float(f1_score(y_te, pred, average="weighted")),
        "TPR": tpr,
        "FPR": fpr,
    }


def build_pipeline(nfilter: int, baseline_cov: np.ndarray):
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        XdawnCovariances(nfilter=nfilter, estimator="lwf", xdawn_estimator="lwf", baseline_cov=baseline_cov),
        TangentSpace(metric="riemann"),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )


def run_fold(dataset: str, test_sid: int, nfilter: int, pred_dir: str, chunk: int) -> dict:
    t0 = time.time()
    src = [s for s in SOURCE_CACHE if s != test_sid]
    x_src = np.concatenate([SOURCE_CACHE[s][0] for s in src])
    y_src = np.concatenate([SOURCE_CACHE[s][1] for s in src])
    clf = build_pipeline(nfilter, pooled_signal_cov(src)).fit(x_src, y_src)
    del x_src

    x_te, y_te = load_subject(dataset, test_sid)
    prob = np.concatenate(
        [clf.predict_proba(np.asarray(x_te[i : i + chunk], dtype=np.float64))[:, 1] for i in range(0, len(y_te), chunk)]
    )
    pred = (prob >= 0.5).astype(np.int64)
    np.savez_compressed(Path(pred_dir) / f"sub{test_sid}.npz", target=y_te, prediction=pred,
                        positive_probability=prob.astype(np.float32))
    row = fold_metrics(test_sid, y_te, pred, prob)
    print(f"[{dataset}] sub{test_sid}: BA={row['BA']:.4f} AUC={row['AUC']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="CAS,THU,THINGS")
    parser.add_argument("--nfilter", type=int, default=4)
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=4000)
    args = parser.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads_per_worker)

    for dataset in args.datasets.split(","):
        tag = f"xdawnrg_{dataset.lower()}_nf{args.nfilter}_lwf_neg{args.neg_ratio}_s{args.seed}"
        out_dir = EXP / "XdawnRiemann" / dataset / "cross-subject" / tag
        pred_dir = out_dir / "trial_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(args.seed)
        SOURCE_CACHE.clear()
        for sid in DATASETS[dataset]["subjects"]:
            x, y = load_subject(dataset, sid)
            SOURCE_CACHE[sid] = subsample_source(x, y, args.neg_ratio, rng)
            del x
        print(f"[{dataset}] cached {len(SOURCE_CACHE)} source subjects", flush=True)

        import multiprocessing as mp

        rows = {}
        for sid in DATASETS[dataset]["subjects"]:
            done = pred_dir / f"sub{sid}.npz"
            if done.exists():
                z = np.load(done)
                rows[sid] = fold_metrics(sid, z["target"], z["prediction"], z["positive_probability"])
        todo = [s for s in DATASETS[dataset]["subjects"] if s not in rows]
        print(f"[{dataset}] {len(rows)} folds already done, {len(todo)} to run", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
            futures = {sid: pool.submit(run_fold, dataset, sid, args.nfilter, str(pred_dir), args.chunk) for sid in todo}
            rows.update({sid: f.result() for sid, f in futures.items()})

        df = pd.DataFrame([rows[s] for s in DATASETS[dataset]["subjects"]])
        num = df.drop(columns="SUB")
        summary = pd.DataFrame(
            [{"SUB": "AVG", **num.mean().round(4).to_dict()}, {"SUB": "STD", **num.std(ddof=1).round(4).to_dict()}]
        )
        pd.concat([df, summary], ignore_index=True).to_csv(out_dir / "results.csv", index=False)
        (out_dir / "manifest.json").write_text(json.dumps({**vars(args), "dataset": dataset, "tag": tag}, indent=1))
        print(f"[{dataset}] BA={num['BA'].mean():.4f} AUC={num['AUC'].mean():.4f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
