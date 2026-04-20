"""
Export checkpoints to ONNX for browser inference.

Usage:

    uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
        --checkpoint checkpoints/omok_full_cuda/latest.pt \
        --output exports/best.onnx

    uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
        --checkpoint checkpoints/omok_full_cuda \
        --output exports/omok_full_cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "PyTorch is required but not installed.\n"
            "Run with: uv run --with torch --with onnx python -m coolrl.omok.export_onnx ...",
            file=sys.stderr,
        )
        sys.exit(1)


def export_one(ckpt_path: Path, out_path: Path, *, verbose: bool = True) -> None:
    import torch

    from .checkpoint import checkpoint_metadata_path, load_checkpoint
    from .config import config_from_dict

    model, config, metadata, _, _ = load_checkpoint(ckpt_path)
    model.eval()
    model.cpu()

    target = checkpoint_metadata_path(ckpt_path)
    sidecar = {}
    if target.exists():
        sidecar = json.loads(target.read_text(encoding="utf-8"))
    out_config = sidecar.get("config", None)
    if not out_config:
        out_config = config.to_dict()
    cfg = config_from_dict(out_config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, cfg.network.input_planes, cfg.rules.board_size, cfg.rules.board_size)
    torch.onnx.export(
        model,
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
        iteration = metadata.get("iteration", "?")
        print(f"  {ckpt_path.name} -> {out_path.name}  ({size_kb:.0f} KB, iter={iteration})")


def export_directory(ckpt_dir: Path, out_dir: Path) -> None:
    from .checkpoint import list_checkpoints

    checkpoints = list_checkpoints(ckpt_dir)
    if not checkpoints:
        print(f"No checkpoints found in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(checkpoints)} checkpoint(s) to {out_dir} ...")

    for ckpt in checkpoints:
        out_path = out_dir / ckpt.with_suffix(".onnx").name
        try:
            export_one(ckpt, out_path)
        except Exception as exc:
            print(f"  [skip] {ckpt.name}: {exc}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export checkpoints to ONNX.",
        epilog="Run with: uv run --with torch --with onnx python -m coolrl.omok.export_onnx ...",
    )
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint file or directory")
    p.add_argument("--output", required=True, help="Output .onnx file path or directory")
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
