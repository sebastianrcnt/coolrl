from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

import numpy as np

from coolrl.omok.board import GameState
from coolrl.omok.features import states_to_feature_planes


_PRELOADED_LIBS: list[object] = []


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _provider_candidates(device: str) -> list[str]:
    token = device.strip().lower()
    if token in {"tensorrt", "trt"}:
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    if token == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if token in {"coreml", "metal", "mps"}:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if token == "auto":
        return ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _select_providers(device: str, available_providers: list[str]) -> list[str]:
    token = device.strip().lower()
    available = set(available_providers)
    if token == "auto":
        providers = [provider for provider in _provider_candidates(token) if provider in available]
        if providers:
            return providers
        raise RuntimeError("ONNX Runtime has no usable execution providers")

    required = {
        "tensorrt": "TensorrtExecutionProvider",
        "trt": "TensorrtExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "metal": "CoreMLExecutionProvider",
        "mps": "CoreMLExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }.get(token, "CPUExecutionProvider")

    if required not in available:
        installed = ", ".join(available_providers) if available_providers else "none"
        if required == "CUDAExecutionProvider":
            hint = (
                "Install the CUDA ONNX Runtime build, for example "
                "`uv sync --extra omok-tui-cuda`, then rerun with `--device cuda`."
            )
        elif required == "TensorrtExecutionProvider":
            hint = (
                "Install the TensorRT TUI runtime, for example "
                "`uv sync --extra omok-tui-tensorrt`, then rerun with `--device tensorrt`."
            )
        else:
            hint = "Use `--device auto` or install an ONNX Runtime build with that provider."
        raise RuntimeError(
            f"{required} was requested but is not available. "
            f"Installed ONNX Runtime providers: {installed}. {hint}"
        )

    return [provider for provider in _provider_candidates(token) if provider in available]


def _preload_cuda_dependencies(ort: object) -> None:
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    preload(directory="")


def _preload_tensorrt_dependencies() -> None:
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None or not spec.submodule_search_locations:
        return
    lib_dir = Path(next(iter(spec.submodule_search_locations)))
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        path = lib_dir / name
        if path.exists():
            _PRELOADED_LIBS.append(ctypes.CDLL(str(path), mode=mode))


def _require_session_provider(device: str, session_providers: list[str]) -> None:
    token = device.strip().lower()
    required = {
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "trt": "TensorrtExecutionProvider",
    }.get(token)
    if required is None or required in session_providers:
        return
    active = ", ".join(session_providers) if session_providers else "none"
    hint = (
        "Check CUDA 12.x/cuDNN 9.x runtime libraries."
        if required == "CUDAExecutionProvider"
        else "Check TensorRT 10.x runtime libraries and CUDA dependencies."
    )
    raise RuntimeError(
        f"{required} was requested but the ONNX Runtime session did not activate it. "
        f"Active providers: {active}. {hint}"
    )


class OnnxModelEvaluator:
    def __init__(self, model_path: Path, *, device: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional env.
            raise RuntimeError(
                "onnxruntime is required for the Omok TUI. "
                "Install it with `uv sync --extra omok-tui` or run with "
                "`uv run --extra omok-tui ...`."
            ) from exc

        self.model_path = Path(model_path)
        providers = _select_providers(device, ort.get_available_providers())
        if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
            _preload_cuda_dependencies(ort)
        if "TensorrtExecutionProvider" in providers:
            _preload_tensorrt_dependencies()

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        _require_session_provider(device, self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.provider = self.session.get_providers()[0]

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        if not states:
            raise ValueError("states must not be empty")
        features = np.ascontiguousarray(states_to_feature_planes(states))
        outputs = self.session.run(None, {self.input_name: features})
        if len(outputs) < 2:
            raise ValueError(
                f"{self.model_path} must produce policy logits and value outputs; "
                f"got {len(outputs)} output(s)"
            )

        logits = np.asarray(outputs[0], dtype=np.float32)
        values = np.asarray(outputs[1], dtype=np.float32).reshape(len(states), -1)[:, 0]
        expected_actions = states[0].action_size
        if logits.shape != (len(states), expected_actions):
            raise ValueError(
                f"{self.model_path.name} policy shape {logits.shape} does not match "
                f"{states[0].board_size}x{states[0].board_size} board "
                f"({expected_actions} actions)"
            )
        return _softmax(logits).astype(np.float32, copy=False), values.astype(np.float32, copy=False)
