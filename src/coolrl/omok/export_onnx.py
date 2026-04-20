"""
Export safetensors checkpoints to ONNX for browser inference.

Requires PyTorch (not a normal project dependency):

    uv run --with torch python -m coolrl.omok.export_onnx \\
        --checkpoint checkpoints/omokai_converted/best \\
        --output exports/best.onnx

Export an entire directory:

    uv run --with torch python -m coolrl.omok.export_onnx \\
        --checkpoint checkpoints/omokai_converted \\
        --output exports/omokai
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "PyTorch is required but not installed.\n"
            "Run with: uv run --with torch python -m coolrl.omok.export_onnx ...",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# PyTorch mirror of tinygrad PolicyValueNet
# ---------------------------------------------------------------------------
# Attribute names match tinygrad exactly so state dict keys are 1:1.

def _build_torch_model(board_size: int, channels: int, blocks: int,
                        input_planes: int, value_hidden: int, se_reduction: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SEBlock(nn.Module):
        def __init__(self, ch: int, red: int):
            super().__init__()
            hidden = max(8, ch // max(1, red))
            self.fc1 = nn.Linear(ch, hidden)
            self.fc2 = nn.Linear(hidden, ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, _, _ = x.shape
            w = x.mean(dim=(2, 3))
            w = self.fc2(self.fc1(w).relu()).sigmoid()
            return x * w.view(b, c, 1, 1)

    class ResidualBlock(nn.Module):
        def __init__(self, ch: int, se_red: int):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)
            self.se = SEBlock(ch, se_red)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.bn1(self.conv1(x)).relu()
            x = self.bn2(self.conv2(x))
            x = self.se(x)
            return (x + residual).relu()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            action_size = board_size * board_size
            self.stem_conv = nn.Conv2d(input_planes, channels, 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm2d(channels)
            self.tower = nn.ModuleList([ResidualBlock(channels, se_reduction) for _ in range(blocks)])
            self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_fc = nn.Linear(2 * action_size, action_size)
            self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(action_size, value_hidden)
            self.value_fc2 = nn.Linear(value_hidden, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.stem_bn(self.stem_conv(x)).relu()
            for block in self.tower:
                x = block(x)
            p = self.policy_bn(self.policy_conv(x)).relu()
            policy_logits = self.policy_fc(p.reshape(p.shape[0], -1))
            v = self.value_bn(self.value_conv(x)).relu()
            v = self.value_fc2(self.value_fc1(v.reshape(v.shape[0], -1)).relu()).tanh()
            return policy_logits, v.reshape(v.shape[0])

    return Net()


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def export_one(ckpt_path: Path, out_path: Path, *, verbose: bool = True) -> None:
    import torch
    from tinygrad.nn.state import safe_load

    from .checkpoint import checkpoint_path, checkpoint_metadata_path, load_checkpoint
    from .config import config_from_dict

    import json

    target = checkpoint_path(ckpt_path)
    meta_path = checkpoint_metadata_path(target)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    config = config_from_dict(payload["config"])
    net_cfg = config.network

    # Load tinygrad weights as numpy
    tg_state = safe_load(target)
    np_state = {k: v.numpy() for k, v in tg_state.items()}

    # Build PyTorch model and load weights
    torch_model = _build_torch_model(
        board_size=config.rules.board_size,
        channels=net_cfg.channels,
        blocks=net_cfg.blocks,
        input_planes=net_cfg.input_planes,
        value_hidden=net_cfg.value_hidden,
        se_reduction=net_cfg.se_reduction,
    )
    torch_state = {k: torch.tensor(v) for k, v in np_state.items()}
    torch_model.load_state_dict(torch_state, strict=True)
    torch_model.eval()

    # Export to ONNX
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, net_cfg.input_planes, config.rules.board_size, config.rules.board_size)
    torch.onnx.export(
        torch_model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    if verbose:
        size_kb = out_path.stat().st_size / 1024
        iteration = payload.get("metadata", {}).get("iteration", "?")
        print(f"  {target.name} -> {out_path.name}  ({size_kb:.0f} KB, iter={iteration})")


def export_directory(ckpt_dir: Path, out_dir: Path) -> None:
    from .checkpoint import list_checkpoints

    checkpoints = list_checkpoints(ckpt_dir)
    if not checkpoints:
        print(f"No safetensors checkpoints found in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(checkpoints)} checkpoint(s) to {out_dir} ...")

    for ckpt in checkpoints:
        out_path = out_dir / ckpt.with_suffix(".onnx").name
        try:
            export_one(ckpt, out_path)
        except Exception as exc:
            print(f"  [skip] {ckpt.name}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export safetensors checkpoints to ONNX.",
        epilog="Run with: uv run --with torch python -m coolrl.omok.export_onnx ...",
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .safetensors file or a directory of checkpoints")
    p.add_argument("--output", required=True,
                   help="Output .onnx file path or directory")
    return p


def main() -> None:
    _require_torch()
    args = _build_argparser().parse_args()
    src = Path(args.checkpoint)
    dst = Path(args.output)

    if src.is_dir():
        export_directory(src, dst)
    else:
        if dst.suffix != ".onnx":
            dst = dst.with_suffix(".onnx")
        export_one(src, dst)
        print("Done.")


if __name__ == "__main__":
    main()
