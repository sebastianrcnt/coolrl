from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save

from .config import RunConfig, config_from_dict
from .network import PolicyValueNet


def checkpoint_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix:
        return target
    return target.with_suffix(".safetensors")


def checkpoint_metadata_path(path: str | Path) -> Path:
    return checkpoint_path(path).with_suffix(".json")


def save_checkpoint(
    path: str | Path,
    model: PolicyValueNet,
    config: RunConfig,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = checkpoint_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_save(model.state_dict(), str(target), metadata={"format": "coolrl.omok.v1"})
    checkpoint_metadata_path(target).write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "metadata": metadata or {},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def load_checkpoint(path: str | Path) -> tuple[PolicyValueNet, RunConfig, dict[str, Any]]:
    target = checkpoint_path(path)
    payload = json.loads(checkpoint_metadata_path(target).read_text(encoding="utf-8"))
    config = config_from_dict(payload["config"])
    model = PolicyValueNet(config.rules.board_size, config.network)
    load_state_dict(model, safe_load(target), strict=True, verbose=False)
    return model, config, payload.get("metadata", {})


def list_checkpoints(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    checkpoints = sorted(path for path in root.glob("*.safetensors") if path.name != "trainer_state.safetensors")
    preferred = []
    for name in ("best.safetensors", "latest.safetensors"):
        path = root / name
        if path in checkpoints:
            preferred.append(path)
    return [path for path in checkpoints if path not in preferred] + preferred


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


def save_optimizer_state(directory: str | Path, optimizer: Any) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    state = {
        key: value
        for key, value in get_state_dict(optimizer).items()
        if not key.startswith("params.")
    }
    safe_save(state, str(root / "optimizer.safetensors"), metadata={"format": "coolrl.omok.optimizer.v1"})


def load_optimizer_state(directory: str | Path, optimizer: Any) -> bool:
    path = Path(directory) / "optimizer.safetensors"
    if not path.exists():
        return False
    load_state_dict(optimizer, safe_load(path), strict=False, verbose=False)
    return True


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
