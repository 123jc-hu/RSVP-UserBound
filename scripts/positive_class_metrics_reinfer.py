"""Re-infer held-out PRISM LOSO folds and report positive-class metrics.

The script uses the final PRISM configs and best-BA checkpoints. It does not
train models. The trial predictions of the other decoders are distributed with
the results archive. Per-trial labels, predictions, and positive-class
probabilities are saved so every aggregate can be recomputed independently.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch.utils.data import DataLoader

from Data.datasets import SubjectDataset
from Data.path_resolver import resolve_dataset_dir
from Train.trainer import CHECKPOINT, build_model, predict
from Utils.config import build_config, load_config, set_random_seed
from paths import EXP


OUT_DIR = EXP / "analysis/positive_class_metrics"


@dataclass(frozen=True)
class RunSpec:
    model: str
    display_name: str
    dataset: str
    tag: str

    @property
    def config_path(self) -> Path:
        return ROOT / "Configs" / f"{self.tag}.yaml"

    @property
    def experiment_dir(self) -> Path:
        return EXP / self.model / self.dataset / "cross-subject" / self.tag


TAGS = {
    "PRISM": {
        "THU": "prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8",
        "CAS": "prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8",
        "THINGS": "prism_things_strict44_d16_bestsofar_sb32_prewsdrop010_regionconcat_lmr_m8_montagefix",
    },
}

DISPLAY_NAMES = {
    "DCTCNNDG": "DC-tCNN-DG",
    "EEGDG": "EEG-DG",
}


def selected_specs(models: set[str] | None, datasets: set[str] | None) -> list[RunSpec]:
    specs = []
    for model, dataset_tags in TAGS.items():
        if models and model not in models:
            continue
        for dataset, tag in dataset_tags.items():
            if datasets and dataset not in datasets:
                continue
            specs.append(RunSpec(model, DISPLAY_NAMES.get(model, model), dataset, tag))
    return specs


def load_run_config(spec: RunSpec) -> dict:
    raw = load_config(str(spec.config_path))
    cfg = build_config(str(raw["model"]), raw["dataset"])
    cfg.update(dict(raw.get("overrides", {})))
    return cfg


def subject_file_map(config: dict) -> dict[str, str]:
    dataset_dir = resolve_dataset_dir(config)
    mapping = {}
    for path in dataset_dir.glob("*.npz"):
        match = re.fullmatch(r"sub(\d+)\.npz", path.name)
        if match:
            mapping[f"sub{int(match.group(1))}"] = str(path)
    return mapping


def result_subjects(results_path: Path) -> list[int]:
    table = pd.read_csv(results_path)
    subjects = []
    for value in table["SUB"].astype(str):
        match = re.search(r"(\d+)", value)
        if match:
            subjects.append(int(match.group(1)))
    return subjects


def audit(specs: Iterable[RunSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        results_path = spec.experiment_dir / "results.csv"
        config_exists = spec.config_path.exists()
        results_exists = results_path.exists()
        subjects = result_subjects(results_path) if results_exists else []
        ckpt = CHECKPOINT
        missing = [
            sid for sid in subjects
            if not (spec.experiment_dir / "checkpoints" / f"sub{sid}" / ckpt).exists()
        ]
        rows.append(
            {
                "model": spec.display_name,
                "model_key": spec.model,
                "dataset": spec.dataset,
                "exp_tag": spec.tag,
                "config_path": str(spec.config_path.relative_to(ROOT)),
                "results_path": str(results_path),
                "config_exists": config_exists,
                "results_exists": results_exists,
                "n_result_subjects": len(subjects),
                "checkpoint_name": ckpt,
                "n_checkpoints_found": len(subjects) - len(missing),
                "missing_subjects": "|".join(map(str, missing)),
                "audit_pass": config_exists and results_exists and not missing and bool(subjects),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_at_fpr(targets: np.ndarray, scores: np.ndarray, limit: float) -> float:
    fpr, tpr, _ = roc_curve(targets, scores)
    eligible = tpr[fpr <= limit + 1e-12]
    return float(eligible.max()) if eligible.size else 0.0


def positive_metrics(targets: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict:
    negative_count = int((targets == 0).sum())
    false_positives = int(((targets == 0) & (predictions == 1)).sum())
    return {
        "n_trials": int(targets.size),
        "n_positive": int((targets == 1).sum()),
        "positive_prevalence": float((targets == 1).mean()),
        "AUC_reinfer": float(roc_auc_score(targets, scores)),
        "BA_reinfer": float(balanced_accuracy_score(targets, predictions)),
        "positive_precision": float(precision_score(targets, predictions, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(targets, predictions, pos_label=1, zero_division=0)),
        "positive_F1": float(f1_score(targets, predictions, pos_label=1, zero_division=0)),
        "AUPRC": float(average_precision_score(targets, scores)),
        "sensitivity_at_FPR_0.05": sensitivity_at_fpr(targets, scores, 0.05),
        "sensitivity_at_FPR_0.10": sensitivity_at_fpr(targets, scores, 0.10),
        "false_positives": false_positives,
        "false_positives_per_1000_nontarget": 1000.0 * false_positives / max(negative_count, 1),
    }


def evaluate_fold(device: torch.device, config: dict, spec: RunSpec, sid: int, fmap: dict) -> dict:
    model = build_model(config).to(device)
    ckpt_path = spec.experiment_dir / "checkpoints" / f"sub{sid}" / CHECKPOINT
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    n_trials = len(np.load(fmap[f"sub{sid}"])["y_data"])
    loader = DataLoader(SubjectDataset(fmap[f"sub{sid}"], np.arange(n_trials)), batch_size=1000, shuffle=False)
    outputs = predict(model, loader, device)
    targets = outputs["targets"].numpy().astype(np.int64, copy=False)
    predictions = outputs["predictions"].numpy().astype(np.int64, copy=False)
    probabilities = outputs["probabilities"].numpy()
    scores = probabilities[:, 1]
    metrics = positive_metrics(targets, predictions, scores)

    trial_dir = OUT_DIR / "trial_predictions" / spec.model / spec.dataset
    trial_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trial_dir / f"sub{sid}.npz",
        target=targets,
        prediction=predictions,
        positive_probability=scores.astype(np.float32),
    )

    return metrics


def evaluate_run(spec: RunSpec, subject_filter: set[int] | None, resume: bool) -> pd.DataFrame:
    config = load_run_config(spec)
    set_random_seed(int(config.get("random_seed", 2026)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fmap = subject_file_map(config)
    result_path = spec.experiment_dir / "results.csv"
    reference = pd.read_csv(result_path)
    reference["subject_id"] = pd.to_numeric(
        reference["SUB"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    reference = reference.dropna(subset=["subject_id"]).copy()
    reference["subject_id"] = reference["subject_id"].astype(int)

    run_stem = spec.tag if "trainseed" in spec.tag else f"{spec.model}_{spec.dataset}"
    out_path = OUT_DIR / "runs" / f"{run_stem}_subject_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(out_path) if resume and out_path.exists() else pd.DataFrame()
    finished = set(existing.get("subject_id", pd.Series(dtype=int)).astype(int).tolist())
    rows = existing.to_dict("records")

    for sid in result_subjects(result_path):
        if subject_filter and sid not in subject_filter:
            continue
        if sid in finished:
            continue
        print(f"[{spec.display_name} | {spec.dataset}] sub{sid}", flush=True)
        metric = evaluate_fold(device, config, spec, sid, fmap)
        ref = reference.loc[reference["subject_id"].eq(sid)].iloc[0]
        row = {
            "model": spec.display_name,
            "model_key": spec.model,
            "dataset": spec.dataset,
            "exp_tag": spec.tag,
            "subject_id": sid,
            **metric,
            "AUC_reported": float(ref["AUC"]),
            "BA_reported": float(ref["BA"]),
        }
        row["AUC_abs_error"] = abs(row["AUC_reinfer"] - row["AUC_reported"])
        row["BA_abs_error"] = abs(row["BA_reinfer"] - row["BA_reported"])
        rows.append(row)
        pd.DataFrame(rows).sort_values("subject_id").to_csv(out_path, index=False)
    return pd.DataFrame(rows)


def summarize(subject_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "AUC_reinfer", "BA_reinfer", "positive_precision", "positive_recall",
        "positive_F1", "AUPRC", "sensitivity_at_FPR_0.05",
        "sensitivity_at_FPR_0.10", "false_positives_per_1000_nontarget",
    ]
    rows = []
    for (model, dataset, tag), group in subject_df.groupby(["model", "dataset", "exp_tag"]):
        row = {"model": model, "dataset": dataset, "exp_tag": tag, "n_subjects": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        row["max_AUC_abs_error"] = group["AUC_abs_error"].max()
        row["max_BA_abs_error"] = group["BA_abs_error"].max()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "model"])


def parse_csv_set(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def parse_subjects(value: str) -> set[int] | None:
    if not value.strip():
        return None
    result = set()
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = map(int, token.split("-", 1))
            result.update(range(lo, hi + 1))
        elif token:
            result.add(int(token))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="", help="Comma-separated registry model keys.")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names.")
    parser.add_argument("--subjects", default="", help="Optional subject IDs/ranges.")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    specs = selected_specs(parse_csv_set(args.models), parse_csv_set(args.datasets))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_df = audit(specs)
    audit_path = OUT_DIR / "checkpoint_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(audit_df[["model", "dataset", "n_result_subjects", "n_checkpoints_found", "audit_pass"]].to_string(index=False))
    print(f"Audit: {audit_path}")
    if not bool(audit_df["audit_pass"].all()):
        raise SystemExit("Checkpoint audit failed; no inference was started.")
    if args.audit_only:
        return

    frames = []
    for spec in specs:
        frames.append(evaluate_run(spec, parse_subjects(args.subjects), args.resume))
    subject_df = pd.concat(frames, ignore_index=True)
    subject_df.to_csv(OUT_DIR / "positive_class_metrics_by_subject.csv", index=False)
    summarize(subject_df).to_csv(OUT_DIR / "positive_class_metrics_summary.csv", index=False)
    print(f"Wrote metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()
