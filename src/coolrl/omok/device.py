from __future__ import annotations

from tinygrad import Device


def configure_device(requested: str = "auto") -> str:
    name = requested.strip().upper()
    if name == "AUTO":
        return str(Device.DEFAULT)
    if name in {"CPU", "METAL", "CUDA", "GPU"}:
        Device[name]
        Device.DEFAULT = name
        return name
    raise ValueError(f"unsupported tinygrad device: {requested!r}")

