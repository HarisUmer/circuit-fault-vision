"""Pick CPU vs GPU so the same train/infer scripts work here and on a later NVIDIA box."""
from __future__ import annotations

import os

import torch


def resolve_device(requested: str = "auto") -> str:
    req = (requested or "auto").strip().lower()
    if req in {"auto", ""}:
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


def is_cpu(device: str) -> bool:
    return str(device).lower() in {"cpu", ""}


def train_runtime(device: str, batch: int | None, imgsz: int | None, workers: int | None) -> dict:
    cpu = is_cpu(device)
    return {
        "device": device,
        "batch": batch if batch is not None else (8 if cpu else 16),
        "imgsz": imgsz if imgsz is not None else (320 if cpu else 640),
        "workers": workers if workers is not None else (0 if cpu or os.name == "nt" else 4),
        "amp": not cpu,
    }
