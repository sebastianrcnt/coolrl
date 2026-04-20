from __future__ import annotations

import torch


def configure_device(requested: str = "auto") -> str:
    name = requested.strip().upper()
    if name == "AUTO":
        if torch.cuda.is_available():
            return "CUDA"
        if torch.backends.mps.is_available():
            return "METAL"
        return "CPU"
    if name == "CUDA":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is unavailable")
        return "CUDA"
    if name in {"METAL", "GPU"}:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Metal/MPS requested but torch.backends.mps is unavailable")
        return "METAL"
    if name == "CPU":
        return "CPU"
    raise ValueError(f"unsupported torch device: {requested!r}")
