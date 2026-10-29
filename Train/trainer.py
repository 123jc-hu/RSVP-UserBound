import logging
import os
import re
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from Data.datamodule import LOSODataModule
from Data.path_resolver import resolve_dataset_dir
from Models.domain_adversarial import DomainDiscriminator
from Models.PRISM import PRISM
from Utils.config import set_random_seed
from Utils.metrics import cal_F1_score, calculate_metrics

METRICS = ["AUC", "BA", "F1", "TPR", "FPR"]
SOURCE_ALIGNMENT = {"source_batch_coral", "source_batch_mmd", "source_batch_dann"}
CHECKPOINT = "best_ba_model--2.pth"


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build the decoder (on CPU); with DANN alignment, attach the domain discriminator."""
    if config["model"] != "PRISM":
        raise ValueError(f"Only PRISM is included in this repository, got model={config['model']}")
    model = PRISM(config)
    if config.get("alignment_loss") == "source_batch_dann" and float(config.get("lambda_align", 0.0)) > 0.0:
        model.domain_discriminator = DomainDiscriminator(
            num_domains=int(config["dann_num_domains"]),
            hidden_dim=int(config.get("dann_hidden_dim", 64)),
            dropout=float(config.get("dann_dropout", 0.1)),
        )
        # Materialize the discriminator's lazy input layer.
        dummy_x = torch.zeros(2, 1, int(config["n_channels"]), int(config["fs"]), dtype=torch.float32)
        with torch.no_grad():
            _, extras = model(dummy_x, return_features=True)
            model.domain_discriminator(extras["features"], grl_lambda=1.0)
    return model


def predict(model: nn.Module, loader, device: torch.device) -> Dict[str, torch.Tensor]:
    """Class probabilities, predictions, and targets of `model` on `loader`."""
    model.eval()
    probs, preds, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits, _ = model(x, return_features=True)
                probs.append(torch.softmax(logits, dim=1).cpu())
                preds.append(logits.argmax(1).cpu())
            targets.append(y)
    return {"probabilities": torch.cat(probs), "predictions": torch.cat(preds), "targets": torch.cat(targets)}


def compute_metrics(out: Dict[str, torch.Tensor]) -> List:
    """AUC, BA, F1, TPR, FPR (each rounded to 4 decimals)."""
    targets, predictions = out["targets"], out["predictions"]
    ba, _, tpr, fpr, auc = calculate_metrics(targets, predictions, y_prob=out["probabilities"])
    f1 = cal_F1_score(targets, predictions)
    return [v.numpy().round(4) for v in (auc, ba, f1, tpr, fpr)]


class OptimizedExperimentRunner:
    """Leave-one-subject-out training and evaluation over one dataset."""

    def __init__(self, config: Dict[str, Any], main_logger: logging.Logger):
        self.config = config
        self.log = main_logger
        use_gpu = bool(config.get("use_gpu", True))
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.exp_dir = self._setup_exp_dir()
        self.datamodule = LOSODataModule(config)

    def _setup_exp_dir(self) -> Path:
        """`Experiments/<model>/<dataset>/cross-subject[/<exp_tag>]`."""
        root = Path(os.environ.get("RSVP_RESULTS_ROOT", Path(__file__).resolve().parents[1] / "Experiments"))
        root = root / self.config["model"] / self.config["dataset"] / "cross-subject"
        exp_tag = str(self.config.get("exp_tag", "") or "").strip()
        if exp_tag:
            root = root / exp_tag
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _seed(self) -> None:
        set_random_seed(
            int(self.config.get("random_seed", 2026)),
            deterministic=bool(self.config.get("deterministic_run", True)),
            benchmark=bool(self.config.get("cudnn_benchmark", False)),
            matmul_precision=str(self.config.get("matmul_precision", "highest")),
        )

    def _subject_files(self) -> Tuple[Dict[str, str], List[int]]:
        """All participant files of the dataset (minus exclusions) and the held-out ids to run."""
        dataset_dir = resolve_dataset_dir(self.config)
        exclude = {int(s) for s in (self.config.get("exclude_subject_ids") or [])}
        file_map = {}
        for name in os.listdir(dataset_dir):
            m = re.fullmatch(r"sub(\d+)\.npz", name)
            if m and int(m.group(1)) not in exclude:
                file_map[f"sub{int(m.group(1))}"] = str(dataset_dir / name)
        if not file_map:
            raise RuntimeError(f"No subject .npz files found in dataset directory: {dataset_dir}")

        start = self.config.get("held_out_start_id", None)
        end = self.config.get("held_out_end_id", None)
        fold_ids = sorted(
            sid for sid in (int(key[3:]) for key in file_map)
            if (start is None or sid >= int(start)) and (end is None or sid <= int(end))
        )
        if self.config.get("n_fold"):
            fold_ids = fold_ids[: int(self.config["n_fold"])]
        return file_map, fold_ids

    def run_experiment(self):
        cfg = self.config
        self.log.info(
            f"Experiment | model={cfg['model']} | dataset={cfg['dataset']} | exp_tag={cfg.get('exp_tag')} | "
            f"seed={cfg.get('random_seed', 2026)} | alignment={cfg.get('alignment_loss')} "
            f"lambda={float(cfg.get('lambda_align', 0.0))}"
        )
        file_map, fold_ids = self._subject_files()
        results = {k: [] for k in METRICS}
        for subject_id in fold_ids:
            self._seed()
            self.datamodule.prepare_data(test_subject_id=subject_id, subject_file_map=file_map)
            metrics = self._run_subject(subject_id)
            self.log.info(
                f"Subject {subject_id} | AUC: {metrics['AUC']:.4f} | BA: {metrics['BA']:.4f} "
                f"| F1: {metrics['F1']:.4f} | TPR: {metrics['TPR']:.4f} | FPR: {metrics['FPR']:.4f}"
            )
            for k in METRICS:
                results[k].append(metrics[k])
        self._save_results(results, fold_ids)

    def _run_subject(self, subject_id: int) -> Dict[str, float]:
        """Train on the source participants of one fold, then test the best-validation-BA model."""
        model = build_model(self.config).to(self.device)
        self.datamodule.setup()
        class_weights = self.datamodule.source_class_weight_tensor(
            n_class=int(self.config.get("n_class", 2)), device=self.device
        )

        ckpt_dir = self.exp_dir / "checkpoints" / f"sub{subject_id}"
        log_dir = self.exp_dir / "logs" / f"sub{subject_id}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if log_dir.exists():
            for child in log_dir.iterdir():
                if child.is_dir() and child.name.startswith("metrics"):
                    rmtree(child, ignore_errors=True)
        writer = SummaryWriter(log_dir / f"metrics_sub{subject_id}" / "version_0")

        trainer = Trainer(self.config, model, self.device, class_weights=class_weights, writer=writer)
        trainer.fit(
            self.datamodule.source_train_dataloader(),
            self.datamodule.val_dataloader(batch_size=int(self.config["val_batch_size"])),
            epochs=int(self.config["epochs"]),
            checkpoint_path=ckpt_dir / CHECKPOINT,
        )
        writer.close()
        torch.save(model.state_dict(), ckpt_dir / "last_model--2.pth")

        model.load_state_dict(torch.load(ckpt_dir / CHECKPOINT, map_location="cpu"))
        out = predict(model, self.datamodule.test_dataloader(batch_size=1000), self.device)
        return {k: round(v, 4) for k, v in zip(METRICS, compute_metrics(out))}

    def _save_results(self, results: Dict[str, List[float]], subject_ids: List[int]):
        """Participant metrics with AVG and STD rows -> results.csv."""
        df = pd.DataFrame(results)
        df.insert(0, "SUB", [f"SUB{i}" for i in subject_ids])
        avg = df[METRICS].mean().round(4)
        std = df[METRICS].std().round(4)
        df = pd.concat([
            df,
            pd.Series({"SUB": "AVG", **avg}).to_frame().T,
            pd.Series({"SUB": "STD", **std}).to_frame().T,
        ], ignore_index=True)
        df.to_csv(self.exp_dir / "results.csv", index=False, float_format="%.4f")
        self.log.info(f"Average metrics: { {k: f'{avg[k]:.4f}+/-{std[k]:.4f}' for k in METRICS} }")


class Trainer:
    """Training of one LOSO fold: AdamW, cosine schedule, class-weighted cross-entropy, optional
    source-participant alignment, checkpointing and early stopping on source-validation BA."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        self.config = config
        self.model = model
        self.device = device
        self.class_weights = None if class_weights is None else class_weights.detach().to(device, dtype=torch.float32)
        self.writer = writer

        lr = float(config["learning_rate"])
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=float(config.get("weight_decay", 0.0))
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(config.get("cosine_t_max", config.get("epochs", 400)))),
            eta_min=lr * float(config.get("cosine_eta_min_factor", 0.01)),
        )
        self.scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

        self.lambda_align = float(config.get("lambda_align", 0.0))
        self.alignment_loss = str(config.get("alignment_loss", "none")).strip().lower()
        self.dann_grl_lambda = float(config.get("dann_grl_lambda", 1.0))

    def fit(self, train_loader, val_loader, *, epochs: int, checkpoint_path: Path):
        patience = int(self.config.get("patience", 10))
        early_stop_start_epoch = int(self.config.get("early_stop_start_epoch", 0))
        best_ba = float("-inf")
        no_improve = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self._train_epoch(train_loader)
            val_metrics = compute_metrics(predict(self.model, val_loader, self.device))
            val_ba = float(val_metrics[1])
            print(
                f"Epoch {epoch:03d}/{epochs:3d} | Train Loss: {train_loss['Total']:.4f} | Val BA: {val_ba:.4f}",
                flush=True,
            )
            self._log(epoch, "train", train_loss, train_metrics)
            self._log(epoch, "val", {}, val_metrics)

            if val_ba > best_ba:
                best_ba = val_ba
                no_improve = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            elif epoch > early_stop_start_epoch:
                no_improve += 1
            if epoch > early_stop_start_epoch and no_improve >= patience:
                print(
                    f"Early stopping triggered on BA | epoch={epoch} | start_epoch={early_stop_start_epoch} | "
                    f"patience={patience} | best_BA={best_ba:.4f}",
                    flush=True,
                )
                break
            self.scheduler.step()

    def _log(self, epoch: int, mode: str, losses: Dict[str, float], metrics: List) -> None:
        if self.writer is None:
            return
        for name, value in losses.items():
            self.writer.add_scalar(f"{mode}/Loss/{name}", value, epoch)
        for name, value in zip(METRICS, metrics):
            self.writer.add_scalar(f"{mode}/{name}", value, epoch)

    def _train_epoch(self, loader) -> Tuple[Dict[str, float], List]:
        self.model.train()
        sums: Dict[str, float] = {}
        n_batches = 0
        probs, preds, targets = [], [], []
        for x, y, domain_id in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            domain_id = domain_id.to(self.device, non_blocking=True)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                logits, extras = self.model(x, return_features=True)
                ce = F.cross_entropy(logits, y, weight=self.class_weights)
                loss = ce
                items = {"ce": float(ce.detach().item())}
                if self.lambda_align > 0.0 and self.alignment_loss in SOURCE_ALIGNMENT:
                    align = self._source_alignment(extras["features"], domain_id)
                    loss = loss + self.lambda_align * align
                    items["src_align"] = float(align.detach().item())
                prob = torch.softmax(logits, dim=1)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            n_batches += 1
            for name, value in {"Total": float(loss.detach().item()), **items}.items():
                sums[name] = sums.get(name, 0.0) + value
            probs.append(prob.detach().cpu())
            preds.append(logits.argmax(1).detach().cpu())
            targets.append(y.detach().cpu())
        out = {"probabilities": torch.cat(probs), "predictions": torch.cat(preds), "targets": torch.cat(targets)}
        return {k: v / max(n_batches, 1) for k, v in sums.items()}, compute_metrics(out)

    # ---------- source-participant alignment ----------
    def _source_alignment(self, feat: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """CORAL / MMD averaged over all pairs of participants in the batch, or the DANN
        domain-classification loss (participant identity as the domain label)."""
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        unique_ids = torch.unique(domain_id.detach(), sorted=True)
        if int(unique_ids.numel()) <= 1:
            return zero
        if self.alignment_loss == "source_batch_dann":
            labels = domain_id.long() - 1
            valid = (labels >= 0) & (labels < int(self.config["dann_num_domains"]))
            if int(valid.sum().item()) <= 1 or int(torch.unique(labels[valid]).numel()) <= 1:
                return zero
            domain_logits = self.model.domain_discriminator(feat[valid].float(), grl_lambda=self.dann_grl_lambda)
            return F.cross_entropy(domain_logits, labels[valid])

        groups = [feat[domain_id == d].float() for d in unique_ids]
        pair_loss = _coral_loss if self.alignment_loss == "source_batch_coral" else _mmd_rbf
        losses = [pair_loss(groups[i], groups[j]) for i in range(len(groups) - 1) for j in range(i + 1, len(groups))]
        return torch.stack(losses).mean()


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    def kernel(a, b):
        a2 = torch.sum(a * a, dim=1, keepdim=True)
        b2 = torch.sum(b * b, dim=1, keepdim=True).T
        dist2 = torch.clamp(a2 + b2 - 2.0 * (a @ b.T), min=0.0)
        return torch.exp(-dist2 / (2.0 * max(float(sigma) ** 2, 1e-12)))

    # Kernel evaluation order (xx, xy, yy) fixes the gradient accumulation order.
    term_xx = torch.mean(kernel(x, x))
    term_xy = torch.mean(kernel(x, y))
    term_yy = torch.mean(kernel(y, y))
    return term_xx + term_yy - 2.0 * term_xy


def _coral_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    def cov(a):
        n, d = int(a.shape[0]), int(a.shape[1])
        if n <= 1:
            return torch.eye(d, device=a.device, dtype=a.dtype)
        centered = a - torch.mean(a, dim=0, keepdim=True)
        return (centered.T @ centered) / float(n - 1) + float(eps) * torch.eye(d, device=a.device, dtype=a.dtype)

    d = int(x.shape[1])
    return torch.sum((cov(x) - cov(y)) ** 2) / float(4 * d * d)
