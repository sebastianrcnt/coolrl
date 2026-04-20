"""
Convert ONNX omok checkpoints to safetensors format for use with coolrl.omok.gui.

Requires the ``onnx`` package (not a normal project dependency):

    uv run --with onnx python -m coolrl.omok.convert_onnx \\
        --source /path/to/omokai/web/models \\
        --output checkpoints/omokai_converted

Background
----------
PyTorch fuses Conv + BatchNorm into a single Conv node during ONNX export,
so the ONNX file contains fused conv weights (with BN scale absorbed) and
no separate BN tensors. The Omok model applies Conv then BN
separately, so we reconstruct identity BN parameters:

    running_mean  = 0
    running_var   = 1 - eps           (so sqrt(running_var + eps) == 1 exactly)
    weight        = 1
    bias          = fused_conv_bias   (the only contribution that remains)
    num_batches_tracked = 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# ONNX extraction helpers
# ---------------------------------------------------------------------------

def _require_onnx():
    try:
        import onnx  # noqa: F401
    except ImportError:
        print(
            "The 'onnx' package is required but not installed.\n"
            "Run the script with:\n"
            "  uv run --with onnx python -m coolrl.omok.convert_onnx ...",
            file=sys.stderr,
        )
        sys.exit(1)


def _load_onnx_initializers(onnx_path: Path) -> dict[str, np.ndarray]:
    import onnx
    import onnx.numpy_helper

    model = onnx.load(str(onnx_path))
    return {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}


# ---------------------------------------------------------------------------
# State dict construction
# ---------------------------------------------------------------------------

_BN_EPS = 1e-5  # default eps for BatchNorm2d


def _identity_bn(out_channels: int, fused_bias: np.ndarray) -> dict[str, np.ndarray]:
    """Identity BN parameters that simply add the fused conv bias."""
    return {
        "running_mean": np.zeros(out_channels, dtype=np.float32),
        "running_var": np.full(out_channels, 1.0 - _BN_EPS, dtype=np.float32),
        "weight": np.ones(out_channels, dtype=np.float32),
        "bias": fused_bias.astype(np.float32),
        "num_batches_tracked": np.array(0, dtype=np.int64),
    }


def _build_state_dict(
    raw: dict[str, np.ndarray],
    channels: int,
    blocks: int,
) -> dict[str, np.ndarray]:
    """Map ONNX initializer names to Omok PolicyValueNet state dict keys.

    Fused conv initializers have opaque ``onnx::Conv_NNN`` names.  They appear
    in graph order: stem, tower[0].conv1, tower[0].conv2, ..., policy, value.
    Within each pair, the higher-dimensional tensor is the weight and the 1-D
    tensor is the bias.  Some ONNX exports omit zero biases (e.g. at iteration
    0), so we fall back to zeros when a bias is missing.
    """
    # --- pair (weight, bias) from sorted onnx::Conv_* entries -------------
    fused_keys = sorted(
        [k for k in raw if k.startswith("onnx::Conv_")],
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )

    fused: list[tuple[np.ndarray, np.ndarray]] = []
    i = 0
    while i < len(fused_keys):
        w = raw[fused_keys[i]]
        if w.ndim < 2:
            raise ValueError(f"Unexpected 1-D initializer before a weight: {fused_keys[i]}")
        out_channels = w.shape[0]
        # Check if the next entry is the matching bias (ndim==1)
        if i + 1 < len(fused_keys) and raw[fused_keys[i + 1]].ndim == 1:
            b = raw[fused_keys[i + 1]]
            i += 2
        else:
            b = np.zeros(out_channels, dtype=np.float32)
            i += 1
        fused.append((w, b))

    expected_fused = 1 + blocks * 2 + 2  # stem + tower + policy + value
    if len(fused) != expected_fused:
        raise ValueError(
            f"Expected {expected_fused} fused conv pairs, got {len(fused)}."
        )

    state: dict[str, np.ndarray] = {}

    def _place_conv_bn(w: np.ndarray, b: np.ndarray, conv_key: str, bn_prefix: str, out_ch: int) -> None:
        state[conv_key] = w.astype(np.float32)
        for suffix, arr in _identity_bn(out_ch, b).items():
            state[f"{bn_prefix}.{suffix}"] = arr

    idx = 0

    # stem
    _place_conv_bn(fused[idx][0], fused[idx][1], "stem_conv.weight", "stem_bn", channels)
    idx += 1

    # residual tower
    for i in range(blocks):
        _place_conv_bn(fused[idx][0], fused[idx][1], f"tower.{i}.conv1.weight", f"tower.{i}.bn1", channels)
        idx += 1
        _place_conv_bn(fused[idx][0], fused[idx][1], f"tower.{i}.conv2.weight", f"tower.{i}.bn2", channels)
        idx += 1

    # policy head conv
    _place_conv_bn(fused[idx][0], fused[idx][1], "policy_conv.weight", "policy_bn", 2)
    idx += 1

    # value head conv
    _place_conv_bn(fused[idx][0], fused[idx][1], "value_conv.weight", "value_bn", 1)
    idx += 1

    # --- SE blocks (named initializers) ------------------------------------
    for i in range(blocks):
        for src, dst in [
            (f"tower.{i}.se.fc.1.weight", f"tower.{i}.se.fc1.weight"),
            (f"tower.{i}.se.fc.1.bias",   f"tower.{i}.se.fc1.bias"),
            (f"tower.{i}.se.fc.3.weight", f"tower.{i}.se.fc2.weight"),
            (f"tower.{i}.se.fc.3.bias",   f"tower.{i}.se.fc2.bias"),
        ]:
            state[dst] = raw[src].astype(np.float32)

    # --- FC heads (named initializers) ------------------------------------
    for src, dst in [
        ("policy_head.4.weight", "policy_fc.weight"),
        ("policy_head.4.bias",   "policy_fc.bias"),
        ("value_head.4.weight",  "value_fc1.weight"),
        ("value_head.4.bias",    "value_fc1.bias"),
        ("value_head.6.weight",  "value_fc2.weight"),
        ("value_head.6.bias",    "value_fc2.bias"),
    ]:
        state[dst] = raw[src].astype(np.float32)

    return state


# ---------------------------------------------------------------------------
# Config / sidecar helpers
# ---------------------------------------------------------------------------

def _infer_arch(raw: dict[str, np.ndarray]) -> tuple[int, int]:
    """Infer value_hidden and se_reduction from tensor shapes."""
    value_hidden = int(raw["value_head.4.weight"].shape[0])

    # SE fc1 shape: [se_hidden, channels]
    se_fc1_key = next(k for k in raw if re.match(r"tower\.\d+\.se\.fc\.1\.weight", k))
    se_hidden = int(raw[se_fc1_key].shape[0])
    channels = int(raw[se_fc1_key].shape[1])
    se_reduction = channels // se_hidden

    return value_hidden, se_reduction


def _make_sidecar(entry: dict, value_hidden: int, se_reduction: int) -> dict:
    return {
        "config": {
            "experiment_name": "omokai_converted",
            "seed": 0,
            "device": "auto",
            "rules": {
                "board_size": entry["board_size"],
                "exactly_five": entry["exactly_five"],
            },
            "network": {
                "input_planes": entry["input_planes"],
                "channels": entry["channels"],
                "blocks": entry["blocks"],
                "value_hidden": value_hidden,
                "se_reduction": se_reduction,
            },
        },
        "metadata": {**entry.get("metadata", {}), "source": "omokai_onnx"},
    }


# ---------------------------------------------------------------------------
# Per-checkpoint conversion
# ---------------------------------------------------------------------------

def _convert_one(
    onnx_path: Path,
    out_path: Path,
    entry: dict,
    value_hidden: int,
    se_reduction: int,
    *,
    verbose: bool = True,
) -> None:
    from safetensors.numpy import save_file

    raw = _load_onnx_initializers(onnx_path)
    numpy_state = _build_state_dict(raw, entry["channels"], entry["blocks"])

    save_file(numpy_state, str(out_path), metadata={"format": "coolrl.omok.legacy_weights.v1"})

    sidecar = _make_sidecar(entry, value_hidden, se_reduction)
    out_path.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if verbose:
        iteration = entry.get("metadata", {}).get("iteration", "?")
        print(f"  {onnx_path.name} -> {out_path.name}  (iter={iteration})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(source: Path, output: Path) -> None:
    _require_onnx()

    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest.json not found in {source}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[dict] = manifest["checkpoints"]

    if not entries:
        print("No checkpoints in manifest.", file=sys.stderr)
        sys.exit(1)

    output.mkdir(parents=True, exist_ok=True)

    # Infer architecture parameters from the first available ONNX file.
    first_onnx = source / entries[0]["file"]
    raw0 = _load_onnx_initializers(first_onnx)
    value_hidden, se_reduction = _infer_arch(raw0)
    print(f"Inferred: value_hidden={value_hidden}, se_reduction={se_reduction}")
    print(f"Converting {len(entries)} checkpoint(s) to {output} ...")

    for entry in entries:
        onnx_path = source / entry["file"]
        if not onnx_path.exists():
            print(f"  [skip] {onnx_path.name} not found")
            continue
        out_name = entry["name"] + ".safetensors"
        _convert_one(onnx_path, output / out_name, entry, value_hidden, se_reduction)

    print("Done.")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert ONNX omok checkpoints to safetensors.",
        epilog="Run with: uv run --with onnx python -m coolrl.omok.convert_onnx ...",
    )
    p.add_argument("--source", required=True, help="Directory containing .onnx files and manifest.json")
    p.add_argument("--output", required=True, help="Destination directory for .safetensors files")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    convert(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()
