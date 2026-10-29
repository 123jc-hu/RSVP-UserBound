from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from Data.datasets import SubjectBatchLoader, SubjectDataset
from Data.npz_io import ensure_npz_cache

SOURCE_VAL_RATIO = 0.2


class LOSODataModule:
    """Data of one leave-one-subject-out fold.

    The held-out participant gives the test set (all trials). Every other participant is a
    source: its trials are split 80/20 into training and validation (class-stratified random
    split, or one contiguous temporal block with a guard gap), and background trials of the
    training part are downsampled to `source_train_bg_downsample_bg_to_pos_ratio` x targets.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.seed = int(config.get("random_seed", 2026))
        # The split seed defaults to the training seed; the training-seed runs fix it at 2026
        # so that the source split stays the same.
        self.source_split_seed = int(config.get("source_split_seed", self.seed))
        self.source_val_split_mode = str(config.get("source_val_split_mode", "random"))
        self.source_val_block_gap = int(config.get("source_val_block_gap", 10))
        self.num_workers = int(config.get("num_workers", 0))
        self.pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))
        self.persistent_workers = bool(config.get("persistent_workers", self.num_workers > 0))
        self.prefetch_factor = config.get("prefetch_factor", None)

        self.subject_file_map: Dict[str, str] = {}
        self.test_subject_id: Optional[int] = None
        self.source_subject_keys: List[str] = []
        self.source_train_dataset_map: Dict[str, SubjectDataset] = {}
        self.source_train_class_counts: Optional[np.ndarray] = None
        self.val_dataset = None
        self.test_dataset = None
        self._label_cache: Dict[str, np.ndarray] = {}

    def prepare_data(self, test_subject_id: int, subject_file_map: Dict[str, str]):
        self.subject_file_map = {key: str(path) for key, path in subject_file_map.items()}
        self.test_subject_id = int(test_subject_id)
        if f"sub{self.test_subject_id}" not in self.subject_file_map:
            raise KeyError(f"Missing held-out subject: sub{self.test_subject_id}")

    def setup(self):
        held_out = f"sub{self.test_subject_id}"
        self.source_subject_keys = sorted(
            [key for key in self.subject_file_map if key != held_out], key=lambda key: int(key[3:])
        )
        n_class = int(self.config.get("n_class", 2))
        class_counts = np.zeros(n_class, dtype=np.int64)
        self.source_train_dataset_map = {}
        val_parts = []
        for key in self.source_subject_keys:
            path = self.subject_file_map[key]
            labels = self._labels(path)
            tr_idx, va_idx = self._split(path, np.arange(labels.shape[0], dtype=np.int64))
            tr_idx = self._downsample_background(key, tr_idx)
            class_counts += np.bincount(labels[tr_idx].astype(np.int64), minlength=n_class)
            self.source_train_dataset_map[key] = SubjectDataset(path, tr_idx)
            val_parts.append(SubjectDataset(path, va_idx))

        test_path = self.subject_file_map[held_out]
        self.val_dataset = ConcatDataset(val_parts)
        self.test_dataset = SubjectDataset(test_path, np.arange(self._labels(test_path).shape[0], dtype=np.int64))
        self.source_train_class_counts = class_counts

    # ---------- loaders ----------
    def _loader_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"num_workers": self.num_workers, "pin_memory": self.pin_memory}
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return kwargs

    def source_train_dataloader(self) -> SubjectBatchLoader:
        per_subject_batch_size = int(self.config["subject_batch_size"])
        # Steps per epoch: one pass over the first source participant's training trials.
        ref_samples = len(self.source_train_dataset_map[self.source_subject_keys[0]])
        return SubjectBatchLoader(
            self.source_train_dataset_map,
            subjects_per_batch=int(self.config["subjects_per_batch"]),
            per_subject_batch_size=per_subject_batch_size,
            steps_per_epoch=max(1, ref_samples // per_subject_batch_size),
            seed=self.seed + int(self.test_subject_id) * 1000,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, **self._loader_kwargs())

    def test_dataloader(self, batch_size: int = 1000) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, **self._loader_kwargs())

    # ---------- source split ----------
    def _split(self, subject_file: str, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(self._labels(subject_file)[indices]).reshape(-1)
        if self.source_val_split_mode == "temporal_block":
            return self._temporal_block_split(subject_file, indices, labels)
        if self.source_val_split_mode != "random":
            raise ValueError(f"Unknown source_val_split_mode: {self.source_val_split_mode}")

        rng = np.random.default_rng(self.source_split_seed)
        tr_parts, va_parts = [], []
        for c in np.unique(labels):
            cls_idx = indices[labels == c]
            n_cls = int(cls_idx.shape[0])
            cls_idx = cls_idx[rng.permutation(n_cls)]
            n_val = 0 if n_cls == 1 else max(1, min(int(round(n_cls * SOURCE_VAL_RATIO)), n_cls - 1))
            va_parts.append(cls_idx[:n_val])
            tr_parts.append(cls_idx[n_val:])
        train_idx = np.concatenate(tr_parts)
        val_idx = np.concatenate(va_parts)
        return train_idx[rng.permutation(train_idx.shape[0])], val_idx[rng.permutation(val_idx.shape[0])]

    def _temporal_block_split(
        self, subject_file: str, indices: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validation = one of round(1/0.2) contiguous blocks in trial (presentation) order.

        Training trials within `source_val_block_gap` positions of the block are dropped. The
        block is drawn per participant from (source_split_seed, file name), skipping blocks
        without both classes.
        """
        order = np.argsort(indices, kind="stable")
        idx, lab = indices[order], labels[order]
        n = idx.shape[0]
        n_blocks = max(2, int(round(1.0 / SOURCE_VAL_RATIO)))
        subject_tag = sum(ord(ch) for ch in Path(subject_file).name)
        rng = np.random.default_rng([self.source_split_seed, subject_tag])
        gap = self.source_val_block_gap
        for k in rng.permutation(n_blocks):
            start, end = k * n // n_blocks, (k + 1) * n // n_blocks
            if np.unique(lab[start:end]).size < 2:
                continue
            val_idx = idx[start:end]
            keep = np.ones(n, dtype=bool)
            keep[max(0, start - gap) : min(n, end + gap)] = False
            train_idx = idx[keep]
            return train_idx[rng.permutation(train_idx.shape[0])], val_idx[rng.permutation(val_idx.shape[0])]
        raise ValueError(f"No temporal validation block with both classes for {subject_file}")

    def _downsample_background(self, subject_key: str, train_indices: np.ndarray) -> np.ndarray:
        """Keep all targets and a random subset of at most ratio x n_targets background trials."""
        bg_ratio = float(self.config.get("source_train_bg_downsample_bg_to_pos_ratio", 1.0))
        seed = int(self.config.get("source_train_bg_downsample_seed", self.seed))
        labels = np.asarray(self._labels(self.subject_file_map[subject_key])[train_indices], dtype=np.int64)
        pos_idx = train_indices[labels == 1]
        bg_idx = train_indices[labels == 0]
        if pos_idx.size == 0 or bg_idx.size == 0:
            return train_indices
        keep_bg = min(int(bg_idx.size), max(1, int(np.floor(float(pos_idx.size) * bg_ratio))))
        rng = np.random.default_rng(seed + int(subject_key[3:]))
        kept = np.concatenate([pos_idx, bg_idx[rng.permutation(bg_idx.shape[0])[:keep_bg]]])
        return kept[rng.permutation(kept.shape[0])]

    # ---------- labels and class weights ----------
    def _labels(self, subject_file: str) -> np.ndarray:
        if subject_file not in self._label_cache:
            cache_dir = ensure_npz_cache(Path(subject_file), ["y_data.npy"])
            self._label_cache[subject_file] = np.load(cache_dir / "y_data.npy", mmap_mode="r")
        return self._label_cache[subject_file]

    def source_class_weight_tensor(self, n_class: int, device: torch.device) -> torch.Tensor:
        """Inverse-frequency class weights of the source-training trials, normalized to mean 1."""
        counts = self.source_train_class_counts.astype(np.float64)[:n_class]
        weights = np.zeros(n_class, dtype=np.float32)
        nonzero = counts > 0
        weights[nonzero] = (float(np.sum(counts)) / float(n_class)) / counts[nonzero]
        weights[nonzero] = weights[nonzero] / float(np.mean(weights[nonzero]))
        return torch.tensor(weights, dtype=torch.float32).to(device)
