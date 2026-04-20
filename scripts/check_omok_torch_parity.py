from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from coolrl.omok.config import load_config
from coolrl.omok.torch_network import PolicyValueNet as TorchPolicyValueNet, load_tinygrad_state_dict

try:
    from tinygrad import Device, Tensor
    from tinygrad.nn.state import load_state_dict, safe_load
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "tinygrad is required for parity checks because this script compares both backends.\n"
        "Run with `uv run --extra omok`."
    ) from exc


def _parse_batch_sizes(raw: str) -> list[int]:
    values = [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("--batches must contain at least one positive integer")
    return [max(1, v) for v in values]


def _feature_batch(cfg, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(batch_size, cfg.network.input_planes, cfg.rules.board_size, cfg.rules.board_size)).astype(
        np.float32
    )


def _tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "realize"):
        return np.array(value.realize().numpy())
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.array(value)


def _run_tinygrad_forward(model, features: np.ndarray, train: bool, device: str) -> tuple[np.ndarray, np.ndarray]:
    tensor = Tensor(features, device=device)
    with Tensor.train(train):
        logits, values = model(tensor)
        probs = logits.softmax(axis=1)
    return _tensor_to_numpy(probs), _tensor_to_numpy(values)


def _run_torch_forward(
    model: torch.nn.Module,
    features: np.ndarray,
    train: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if train:
        model.train()
        grad_ctx = torch.enable_grad()
    else:
        model.eval()
        grad_ctx = torch.inference_mode()

    with grad_ctx:
        logits, values = model(torch.as_tensor(features, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy(), values.detach().cpu().numpy()


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _load_tinygrad_weights(path: str | None, config) -> dict[str, np.ndarray]:
    if path is None:
        from coolrl.omok.network import PolicyValueNet

        model = PolicyValueNet(config.rules.board_size, config.network)
        return {key: _tensor_to_numpy(value) for key, value in model.state_dict().items()}

    state = safe_load(path)
    return {key: _tensor_to_numpy(value) for key, value in state.items()}


def _build_tinygrad_model(config, state: dict[str, np.ndarray], device: str):
    from coolrl.omok.network import PolicyValueNet

    tiny_model = PolicyValueNet(config.rules.board_size, config.network)
    tensor_state = {key: Tensor(np.asarray(value), device=device) for key, value in state.items()}
    load_state_dict(tiny_model, tensor_state, strict=True, verbose=False)
    return tiny_model


def _setup_tinygrad_device(requested: str | None) -> str:
    token = (requested or "auto").upper()
    if token == "CPU":
        Device.DEFAULT = "CPU"
        return "CPU"
    if token == "CUDA":
        Device.DEFAULT = "CUDA"
        return "CUDA"
    if token == "METAL":
        Device.DEFAULT = "METAL"
        return "METAL"
    if token == "GPU":
        Device.DEFAULT = "GPU"
        return "GPU"
    Device.DEFAULT = "AUTO"
    return "AUTO"


def _check_running_stats(tiny_model, torch_model: torch.nn.Module) -> None:
    tiny_state = tiny_model.state_dict()
    torch_state = torch_model.state_dict()

    tiny_bn = {
        key: _tensor_to_numpy(value)
        for key, value in tiny_state.items()
        if key.endswith("running_mean") or key.endswith("running_var")
    }
    torch_bn = {
        key: value
        for key, value in torch_state.items()
        if key.endswith("running_mean") or key.endswith("running_var")
    }
    if set(tiny_bn) != set(torch_bn):
        tiny_only = sorted(set(tiny_bn) - set(torch_bn))
        torch_only = sorted(set(torch_bn) - set(tiny_bn))
        raise SystemExit(
            "running-stat key mismatch:"
            f" tinygrad-only={tiny_only or []}, torch-only={torch_only or []}"
        )

    print("BatchNorm running stats (shape,dtype):")
    for key in sorted(tiny_bn):
        tiny_arr = tiny_bn[key]
        torch_arr = torch_bn[key].detach().cpu().numpy()
        if tiny_arr.shape != torch_arr.shape:
            raise SystemExit(f"{key}: running-stat shape mismatch {tiny_arr.shape} vs {torch_arr.shape}")
        print(
            f"  {key}: shape={tiny_arr.shape} dtype={tiny_arr.dtype} "
            f"(torch dtype={torch_bn[key].dtype})"
        )


def _torch_device(name: str | None) -> str:
    requested = (name or "auto").upper()
    if requested == "CUDA":
        return "cuda"
    if requested in {"METAL", "MPS"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check parity between tinygrad and torch Omok networks.")
    parser.add_argument("--config", default="configs/omok_full_cuda.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional legacy tinygrad .safetensors checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batches", default="7,64,93")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-batch", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _setup_tinygrad_device(args.device)
    batches = _parse_batch_sizes(args.batches)
    rng = np.random.default_rng(args.seed)

    tiny_state = _load_tinygrad_weights(args.checkpoint, cfg)
    tiny_model = _build_tinygrad_model(cfg, tiny_state, device)
    torch_model = TorchPolicyValueNet(cfg.rules.board_size, cfg.network)
    load_tinygrad_state_dict(torch_model, tiny_state)
    torch_device = _torch_device(device)
    torch_model.to(torch_device)

    print(
        f"config={Path(args.config).name} device={device} "
        f"batch_sizes={batches} channels={cfg.network.channels} blocks={cfg.network.blocks}"
    )
    print("Eval-mode parity:")
    for batch in batches:
        features = _feature_batch(cfg, batch, rng)
        t_probs, t_values = _run_tinygrad_forward(tiny_model, features, train=False, device=device)
        p_probs, p_values = _run_torch_forward(torch_model, features, train=False, device=torch_device)
        print(
            f"  batch={batch}: "
            f"policy_max_abs_diff={_max_abs(t_probs, p_probs):.6g} "
            f"value_max_abs_diff={_max_abs(t_values, p_values):.6g}"
        )

    train_batch = args.train_batch or batches[0]
    print(f"Train-mode parity (batch={train_batch}):")
    train_features = _feature_batch(cfg, train_batch, rng)
    t_probs, t_values = _run_tinygrad_forward(tiny_model, train_features, train=True, device=device)
    p_probs, p_values = _run_torch_forward(torch_model, train_features, train=True, device=torch_device)
    print(
        f"  policy_max_abs_diff={_max_abs(t_probs, p_probs):.6g} "
        f"value_max_abs_diff={_max_abs(t_values, p_values):.6g}"
    )

    _check_running_stats(tiny_model, torch_model)
    print("Done.")


if __name__ == "__main__":
    main()
