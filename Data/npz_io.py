import os
import zipfile
from pathlib import Path
from typing import Iterable


def ensure_npz_cache(npz_path: Path, members: Iterable[str]) -> Path:
    """Extract members of a .npz once into `<name>.__cache__/` next to it, for memmap access."""
    npz_path = Path(npz_path)
    cache_dir = npz_path.parent / (npz_path.stem + ".__cache__")
    cache_dir.mkdir(parents=True, exist_ok=True)
    members = list(members)
    if all((cache_dir / member).exists() for member in members):
        return cache_dir

    with zipfile.ZipFile(npz_path, "r") as zf:
        names = set(zf.namelist())
        for member in members:
            out_path = cache_dir / member
            if out_path.exists():
                continue
            if member not in names:
                raise KeyError(f"{npz_path} missing member {member}")
            tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
            with zf.open(member, "r") as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            os.replace(tmp_path, out_path)
    return cache_dir
