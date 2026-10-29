from pathlib import Path
from typing import Dict, Iterator, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from Data.npz_io import ensure_npz_cache


class SubjectDataset(Dataset):
    """Trials `indices` of one participant's .npz file, read through numpy memmaps."""

    def __init__(self, file_path: Union[str, Path], indices: np.ndarray):
        cache_dir = ensure_npz_cache(Path(file_path), ["x_data.npy", "y_data.npy"])
        self.x_data = np.load(cache_dir / "x_data.npy", mmap_mode="r")
        self.y_data = np.load(cache_dir / "y_data.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        x = np.asarray(self.x_data[real_idx], dtype=np.float32).copy()
        return torch.from_numpy(x), torch.tensor(int(self.y_data[real_idx])).long()


class SubjectBatchLoader:
    """Source-training loader: each step draws `subjects_per_batch` participants at random and
    takes the next `per_subject_batch_size` trials from each one's shuffled queue.

    Yields (x, y, participant id). An epoch has `steps_per_epoch` steps.
    """

    def __init__(
        self,
        subject_datasets: Dict[str, SubjectDataset],
        *,
        subjects_per_batch: int,
        per_subject_batch_size: int,
        steps_per_epoch: int,
        seed: int,
        pin_memory: bool = False,
    ):
        if subjects_per_batch > len(subject_datasets):
            raise ValueError(
                f"subjects_per_batch={subjects_per_batch} is larger than the source pool ({len(subject_datasets)})."
            )
        self.subject_datasets = dict(subject_datasets)
        self.subject_keys = list(self.subject_datasets)
        self.subjects_per_batch = int(subjects_per_batch)
        self.per_subject_batch_size = int(per_subject_batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)
        self._epoch = 0

    def __len__(self):
        return self.steps_per_epoch

    def _epoch_queues(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        queues = {}
        bs = self.per_subject_batch_size
        for key in self.subject_keys:
            idx = self.subject_datasets[key].indices
            n_batches = int(idx.size // bs)
            if n_batches == 0:
                queues[key] = np.empty((0, bs), dtype=np.int64)
                continue
            queues[key] = rng.permutation(idx)[: n_batches * bs].reshape(n_batches, bs)
        return queues

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        queues = self._epoch_queues(rng)
        ptr = {key: 0 for key in self.subject_keys}

        for _ in range(self.steps_per_epoch):
            available = [key for key in self.subject_keys if ptr[key] < queues[key].shape[0]]
            if len(available) < self.subjects_per_batch:
                break
            chosen = [available[int(i)] for i in rng.choice(len(available), size=self.subjects_per_batch, replace=False)]

            x_parts, y_parts, d_parts = [], [], []
            for key in chosen:
                ds = self.subject_datasets[key]
                batch_idx = queues[key][ptr[key]]
                ptr[key] += 1
                x_parts.append(np.asarray(ds.x_data[batch_idx], dtype=np.float32))
                y_parts.append(np.asarray(ds.y_data[batch_idx], dtype=np.int64))
                d_parts.append(np.full(batch_idx.shape[0], int(key[3:]), dtype=np.int64))

            x = np.concatenate(x_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
            d = np.concatenate(d_parts, axis=0)
            perm = rng.permutation(y.shape[0])
            batch = (
                torch.from_numpy(np.ascontiguousarray(x[perm], dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y[perm], dtype=np.int64)),
                torch.from_numpy(np.ascontiguousarray(d[perm], dtype=np.int64)),
            )
            if self.pin_memory and torch.cuda.is_available():
                batch = tuple(t.pin_memory() for t in batch)
            yield batch
