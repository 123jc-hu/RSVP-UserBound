"""Within-subject (calibrated) reference decoders.

Tests whether calibration on a user's own data rescues the users that every
calibration-free decoder finds hard. Each subject's trials are split in
presentation order exactly as in analyze_diagnostic_circularity_reliability.py:
the first len(y)//2 trials are available for calibration and the second half
is the test set (for CAS the halves are the two sessions). The second-half BA
is therefore directly comparable with the second-half BA of the LOSO models.

Methods (--methods):
  slda     standard ERP classifier: mean amplitude of every channel in ten
           100-ms bins over the 0-1000 ms epoch, Ledoit-Wolf shrinkage LDA with
           equal priors (no tuned hyperparameter). All calibration trials used.
  xdawnrg  the LOSO xDAWN-RG pipeline of run_xdawn_riemann_loso.py fitted on the
           user's own data (targets plus `--neg-ratio` x as many non-targets).
           Kept for transparency: on four pilot users it could not be estimated
           reliably from one user's data (BA 0.59-0.73 where sLDA gave 0.82-0.91).

Calibration conditions (--conditions):
  half  the first half of the user's trials
  K<k>  only the span up to the k-th target of the first half, the same span as
        the K-target descriptor analysis

Outputs:
  Experiments/<SLDA|XdawnRiemann>/<DS>/within-subject/<tag>/results.csv
  Experiments/<SLDA|XdawnRiemann>/<DS>/within-subject/<tag>/trial_predictions/sub<k>.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_xdawn_riemann_loso import DATASETS, EXP, fold_metrics, load_subject  # noqa: E402

METHOD_DIR = {"slda": "SLDA", "xdawnrg": "XdawnRiemann"}
N_BINS = 10


def calibration_span(y: np.ndarray, condition: str) -> np.ndarray:
    first = np.arange(len(y) // 2)
    if condition == "half":
        return first
    k = int(condition[1:])
    pos = first[y[first] == 1]
    return np.arange(pos[k - 1] + 1) if len(pos) >= k else np.arange(0)


def bin_means(x: np.ndarray) -> np.ndarray:
    """Mean amplitude per channel in N_BINS equal bins over the epoch -> [N, C * N_BINS]."""
    bins = np.array_split(np.arange(x.shape[-1]), N_BINS)
    return np.concatenate([x[..., b].mean(axis=-1) for b in bins], axis=1)


def fit_predict(method: str, x: np.ndarray, y: np.ndarray, span: np.ndarray, test: np.ndarray,
                nfilter: int, neg_ratio: int, rng: np.random.Generator, chunk: int):
    if method == "slda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        train = span
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
        clf.fit(bin_means(np.asarray(x[train], dtype=np.float64)), y[train])
        prob = np.concatenate([
            clf.predict_proba(bin_means(np.asarray(x[test[i : i + chunk]], dtype=np.float64)))[:, 1]
            for i in range(0, len(test), chunk)
        ])
        return train, prob

    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pos = span[y[span] == 1]
    neg = span[y[span] == 0]
    neg = rng.choice(neg, size=min(len(neg), neg_ratio * len(pos)), replace=False)
    train = np.sort(np.concatenate([pos, neg]))
    clf = make_pipeline(
        XdawnCovariances(nfilter=nfilter, estimator="lwf", xdawn_estimator="lwf"),
        TangentSpace(metric="riemann"),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    ).fit(np.asarray(x[train], dtype=np.float64), y[train])
    prob = np.concatenate([
        clf.predict_proba(np.asarray(x[test[i : i + chunk]], dtype=np.float64))[:, 1]
        for i in range(0, len(test), chunk)
    ])
    return train, prob


def run_subject(dataset: str, sid: int, method: str, condition: str, nfilter: int, neg_ratio: int,
                seed: int, pred_dir: str, chunk: int) -> dict | None:
    t0 = time.time()
    x, y = load_subject(dataset, sid)
    span = calibration_span(y, condition)
    if span.size == 0:
        print(f"[{dataset} {method} {condition}] sub{sid}: too few targets, skipped", flush=True)
        return None
    test = np.arange(len(y) // 2, len(y))
    train, prob = fit_predict(method, x, y, span, test, nfilter, neg_ratio, np.random.default_rng(seed + sid), chunk)
    y_te = y[test]
    pred = (prob >= 0.5).astype(np.int64)
    np.savez_compressed(Path(pred_dir) / f"sub{sid}.npz", target=y_te, prediction=pred,
                        positive_probability=prob.astype(np.float32), train_index=train)
    row = fold_metrics(sid, y_te, pred, prob)
    row["n_calib_targets"] = int((y[train] == 1).sum())
    row["n_calib_trials"] = int(train.size)
    print(f"[{dataset} {method} {condition}] sub{sid}: BA={row['BA']:.4f} AUC={row['AUC']:.4f} "
          f"targets={row['n_calib_targets']} ({time.time() - t0:.0f}s)", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="THU")
    parser.add_argument("--methods", default="slda,xdawnrg")
    parser.add_argument("--conditions", default="half,K40")
    parser.add_argument("--nfilter", type=int, default=4)
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=4000)
    args = parser.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads_per_worker)

    for dataset in args.datasets.split(","):
        for method in args.methods.split(","):
            for condition in args.conditions.split(","):
                suffix = f"nf{args.nfilter}_lwf_neg{args.neg_ratio}" if method == "xdawnrg" else f"bin{N_BINS}_lw"
                tag = f"{method}_{dataset.lower()}_within_{condition.lower()}_{suffix}_s{args.seed}"
                out_dir = EXP / METHOD_DIR[method] / dataset / "within-subject" / tag
                pred_dir = out_dir / "trial_predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)

                subjects = DATASETS[dataset]["subjects"]
                rows = {}
                for sid in subjects:
                    done = pred_dir / f"sub{sid}.npz"
                    if done.exists():
                        z = np.load(done)
                        y_all = load_subject(dataset, sid)[1]
                        rows[sid] = {**fold_metrics(sid, z["target"], z["prediction"], z["positive_probability"]),
                                     "n_calib_targets": int((y_all[z["train_index"]] == 1).sum()),
                                     "n_calib_trials": int(z["train_index"].size)}
                todo = [s for s in subjects if s not in rows]
                print(f"[{dataset} {method} {condition}] {len(rows)} subjects done, {len(todo)} to run", flush=True)
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {sid: pool.submit(run_subject, dataset, sid, method, condition, args.nfilter,
                                                args.neg_ratio, args.seed, str(pred_dir), args.chunk) for sid in todo}
                    rows.update({sid: f.result() for sid, f in futures.items()})

                df = pd.DataFrame([rows[s] for s in subjects if rows.get(s) is not None])
                num = df.drop(columns="SUB")
                summary = pd.DataFrame([
                    {"SUB": "AVG", **num.mean().round(4).to_dict()},
                    {"SUB": "STD", **num.std(ddof=1).round(4).to_dict()},
                ])
                pd.concat([df, summary], ignore_index=True).to_csv(out_dir / "results.csv", index=False)
                (out_dir / "manifest.json").write_text(json.dumps(
                    {**vars(args), "dataset": dataset, "method": method, "condition": condition, "tag": tag,
                     "test": "second half of trials in presentation order"}, indent=1))
                print(f"[{dataset} {method} {condition}] BA={num['BA'].mean():.4f} AUC={num['AUC'].mean():.4f} "
                      f"(n={len(df)}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
