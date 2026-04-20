from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig, config_from_dict
from .torch_network import PolicyValueNet, load_legacy_state_dict


def _coerce_cpu(state: Any) -> Any:
    if isinstance(state, dict):
        return {key: _coerce_cpu(value) for key, value in state.items()}
    if torch.is_tensor(state):
        return state.detach().cpu()
    if isinstance(state, (list, tuple)):
        return type(state)(_coerce_cpu(item) for item in state)
    return state


def checkpoint_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix:
        return target
    pt = target.with_suffix(".pt")
    if pt.exists():
        return pt
    legacy = target.with_suffix(".safetensors")
    if legacy.exists():
        return legacy
    return pt


def checkpoint_metadata_path(path: str | Path) -> Path:
    return checkpoint_path(path).with_suffix(".json")


def save_checkpoint(
    path: str | Path,
    model: PolicyValueNet,
    config: RunConfig,
    metadata: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> Path:
    target = checkpoint_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint_format": "coolrl.omok.torch.v1",
        "model": _coerce_cpu(model.state_dict()),
        "optimizer": _coerce_cpu(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler": _coerce_cpu(scheduler.state_dict()) if scheduler is not None else None,
        "metadata": {
            "torch_version": torch.__version__,
            "config": config.to_dict(),
            "iteration": 0,
            **(metadata or {}),
        },
    }
    torch.save(payload, target)

    checkpoint_metadata_path(target).write_text(
        json.dumps(payload["metadata"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _load_legacy_model(path: Path) -> tuple[PolicyValueNet, RunConfig]:
    from safetensors.numpy import load_file

    raw_metadata_path = checkpoint_metadata_path(path)
    payload = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    config_payload = payload.get("config", payload.get("metadata", {}).get("config"))
    if config_payload is None:
        raise ValueError(f"legacy checkpoint metadata does not include config: {raw_metadata_path}")
    config = config_from_dict(config_payload)
    model = PolicyValueNet(config.rules.board_size, config.network)
    load_legacy_state_dict(model, load_file(path))
    return model, config


def load_checkpoint(
    path: str | Path,
) -> tuple[PolicyValueNet, RunConfig, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    target = checkpoint_path(path)
    if target.suffix == ".pt":
        payload = torch.load(target, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        config = config_from_dict(metadata["config"])
        model = PolicyValueNet(config.rules.board_size, config.network)
        model.load_state_dict(payload["model"])
        return (
            model,
            config,
            metadata,
            payload.get("optimizer"),
            payload.get("scheduler"),
        )

    if target.suffix == ".safetensors":
        model, config = _load_legacy_model(target)
        metadata = {"checkpoint_format": "coolrl.omok.legacy_weights.v1"}
        raw_metadata_path = checkpoint_metadata_path(target)
        if raw_metadata_path.exists():
            raw = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
            metadata.update(raw.get("metadata", {}))
            if "config" in raw:
                config = config_from_dict(raw["config"])
        metadata["torch_version"] = torch.__version__
        return model, config, metadata, None, None

    raise ValueError(f"unsupported checkpoint extension: {target.suffix!r}")


def list_checkpoints(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    checkpoint_files = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix in {".pt", ".safetensors"}
        and path.name not in {"trainer_state.pt", "optimizer.safetensors"}
    )
    preferred = []
    for name in ("best.pt", "latest.pt", "best.safetensors", "latest.safetensors"):
        for path in checkpoint_files:
            if path.name == name:
                preferred.append(path)
                break
    remainder = [path for path in checkpoint_files if path not in preferred]
    return remainder + preferred


def save_trainer_state(
    directory: str | Path,
    metadata: dict[str, Any],
    replay_state: dict[str, Any],
) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / "trainer_state.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (root / "replay.pkl").open("wb") as fh:
        pickle.dump(replay_state, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_trainer_state(directory: str | Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    root = Path(directory)
    metadata_path = root / "trainer_state.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    replay_path = root / "replay.pkl"
    replay_state = None
    if replay_path.exists():
        with replay_path.open("rb") as fh:
            replay_state = pickle.load(fh)
    return metadata, replay_state
