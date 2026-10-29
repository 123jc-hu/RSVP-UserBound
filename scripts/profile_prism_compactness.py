#!/usr/bin/env python3
"""Profile the final PRISM architecture without retraining.

MACs include Conv1d, Linear, the regional virtual-channel projections, and
the learned W_s projection. Batch normalization, activations, pooling,
dropout, tensor concatenation, and data movement are excluded, matching the
usual neural-network MAC convention. FLOPs are reported as two per MAC.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.PRISM import (
    PRISM,
    RegionalChannelProjector,
    SpatialProjector,
    module_parameter_summary,
)


def final_config() -> dict:
    """Return the paper-facing default THU configuration declared in Table II."""
    return {
        "n_channels": 62,
        "fs": 128,
        "n_class": 2,
        "erp_net_n_components": 16,
        "erp_net_token_dim": 8,
        "erp_net_prews_fusion": "concat",
        "erp_net_dropout": 0.3,
        "erp_net_activation": "gelu",
        "erp_net_prews_spatial_dropout_prob": 0.10,
        "erp_net_input_region_projector": "sagittal_lmr",
        "erp_net_input_region_projector_dim": 8,
        "erp_net_input_region_projector_fusion": "concat",
        "erp_net_max_norm": 1.0,
    }


def count_macs(model: PRISM, sample: torch.Tensor) -> tuple[int, dict[str, int]]:
    """Count per-sample multiply-accumulate operations on the actual graph."""
    counts: dict[str, int] = {}
    handles = []

    def add(name: str, value: int) -> None:
        counts[name] = counts.get(name, 0) + int(value)

    def conv_hook(name: str):
        def hook(module: nn.Conv1d, inputs, output) -> None:
            batch = int(output.shape[0])
            out_elements = int(output.numel() // batch)
            kernel_work = int(module.kernel_size[0]) * (
                int(module.in_channels) // int(module.groups)
            )
            add(name, out_elements * kernel_work)

        return hook

    def linear_hook(name: str):
        def hook(module: nn.Linear, inputs, output) -> None:
            batch = int(output.shape[0]) if output.ndim > 1 else 1
            out_elements = int(output.numel() // batch)
            add(name, out_elements * int(module.in_features))

        return hook

    def regional_hook(name: str):
        def hook(module: RegionalChannelProjector, inputs, output) -> None:
            time_points = int(inputs[0].shape[-1])
            add(name, time_points * sum(int(weight.numel()) for weight in module.projectors))

        return hook

    def spatial_hook(name: str):
        def hook(module: SpatialProjector, inputs, output) -> None:
            time_points = int(inputs[0].shape[-1])
            add(name, time_points * int(module.weight.numel()))

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(conv_hook(name)))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook(name)))
        elif isinstance(module, RegionalChannelProjector):
            handles.append(module.register_forward_hook(regional_hook(name)))
        elif isinstance(module, SpatialProjector):
            handles.append(module.register_forward_hook(spatial_hook(name)))

    model.eval()
    with torch.inference_mode():
        model(sample)
    for handle in handles:
        handle.remove()
    return sum(counts.values()), counts


def cuda_latency(
    model: PRISM,
    batch_size: int,
    *,
    warmup: int,
    blocks: int,
    repeats_per_block: int,
) -> dict[str, float]:
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    x = torch.randn(batch_size, 1, 62, 128, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize(device)

        samples_ms = []
        for _ in range(blocks):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats_per_block):
                model(x)
            end.record()
            torch.cuda.synchronize(device)
            samples_ms.append(float(start.elapsed_time(end)) / repeats_per_block)

    median_ms = statistics.median(samples_ms)
    return {
        "batch_size": batch_size,
        "median_batch_ms": median_ms,
        "min_batch_ms": min(samples_ms),
        "max_batch_ms": max(samples_ms),
        "median_per_trial_ms": median_ms / batch_size,
        "throughput_trials_per_s": 1000.0 * batch_size / median_ms,
        "block_samples_ms": samples_ms,
    }


def cpu_name() -> str:
    try:
        output = subprocess.check_output(["lscpu"], text=True)
        for line in output.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def nvidia_driver_version() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def write_report(output_dir: Path, summary: dict) -> None:
    latency = summary["latency"]
    lines = [
        "# Q2-08: PRISM Compactness and Inference Cost",
        "",
        "## Protocol",
        "",
        "- Final paper-facing default THU model: C=62, T=128, D=16, M=8.",
        "- FP32 eager-mode inference with `model.eval()` and `torch.inference_mode()`.",
        f"- GPU: {summary['hardware']['gpu']} (driver {summary['hardware']['driver']}).",
        f"- PyTorch {summary['hardware']['torch']}; CUDA {summary['hardware']['cuda']}.",
        f"- Latency: {summary['benchmark']['warmup']} warm-up passes, "
        f"{summary['benchmark']['blocks']} blocks x "
        f"{summary['benchmark']['repeats_per_block']} timed passes, CUDA events.",
        "- MACs include convolution, linear projection, regional projection, and W_s.",
        "- BN, activations, pooling, dropout, concatenation, and data movement are excluded.",
        "- FLOPs use the explicit convention 1 MAC = 2 FLOPs.",
        "",
        "## Results",
        "",
        f"- Trainable parameters: {summary['trainable_parameters']:,}.",
        f"- MACs per trial: {summary['macs_per_trial']:,} "
        f"({summary['macs_per_trial']/1e6:.4f} M).",
        f"- FLOPs per trial: {summary['flops_per_trial']:,} "
        f"({summary['flops_per_trial']/1e6:.4f} M).",
    ]
    for row in latency:
        lines.append(
            f"- Batch {row['batch_size']}: median {row['median_batch_ms']:.4f} ms/batch, "
            f"block range {row['min_batch_ms']:.4f}-{row['max_batch_ms']:.4f} ms, "
            f"{row['median_per_trial_ms']:.4f} ms/trial, "
            f"{row['throughput_trials_per_s']:.1f} trials/s."
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These measurements characterize model-side forward-pass cost on one desktop GPU. "
            "They exclude data loading, preprocessing, acquisition, and transfer overhead, and "
            "therefore do not establish end-to-end real-time deployment.",
            "",
        ]
    )
    (output_dir / "Q2_08_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Experiments/analysis/prism_compactness"),
    )
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--repeats-per-block", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")

    model = PRISM(final_config()).eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, mac_breakdown = count_macs(model, torch.randn(1, 1, 62, 128))

    if not torch.cuda.is_available():
        raise RuntimeError("Q2-08 latency benchmark requires an available CUDA GPU.")
    gpu_index = torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(gpu_index)
    latency = [
        cuda_latency(
            PRISM(final_config()),
            batch_size,
            warmup=args.warmup,
            blocks=args.blocks,
            repeats_per_block=args.repeats_per_block,
        )
        for batch_size in (1, 32)
    ]

    summary = {
        "configuration": final_config(),
        "input_shape": [1, 1, 62, 128],
        "trainable_parameters": params,
        "parameter_breakdown": module_parameter_summary(model),
        "macs_per_trial": macs,
        "flops_per_trial": 2 * macs,
        "mac_breakdown": mac_breakdown,
        "latency": latency,
        "hardware": {
            "gpu": gpu_props.name,
            "gpu_memory_bytes": int(gpu_props.total_memory),
            "driver": nvidia_driver_version(),
            "cpu": cpu_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "benchmark": {
            "dtype": "float32",
            "execution": "PyTorch eager",
            "warmup": args.warmup,
            "blocks": args.blocks,
            "repeats_per_block": args.repeats_per_block,
            "timing": "CUDA events with synchronization after each block",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
    }

    (args.output_dir / "compactness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        latency_fields = [key for key in latency[0] if key != "block_samples_ms"]
        writer = csv.DictWriter(handle, fieldnames=latency_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(latency)
    with (args.output_dir / "latency_blocks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_size", "block", "mean_batch_ms"])
        for row in latency:
            for block, value in enumerate(row["block_samples_ms"], start=1):
                writer.writerow([row["batch_size"], block, value])
    with (args.output_dir / "mac_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module", "macs_per_trial"])
        writer.writerows(sorted(mac_breakdown.items()))
    with (args.output_dir / "parameter_breakdown.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["module", "trainable_parameters"])
        writer.writerows(summary["parameter_breakdown"].items())
    write_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
