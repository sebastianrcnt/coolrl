from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .board import GameState
from .evaluator import Evaluator
from .features import states_to_feature_planes
from .torch_evaluator import _coerce_torch_model, _require_torch, _torch_device, torch

_TorchModuleBase = object if torch is None else torch.nn.Module


def tensorrt_is_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def _require_tensorrt() -> Any:
    if not tensorrt_is_available():
        raise RuntimeError(
            "TensorRT evaluator requires the optional NVIDIA TensorRT Python package. "
            "Install TensorRT on a CUDA-capable NVIDIA system, or use "
            "`selfplay.evaluator_backend: torch`."
        )
    import tensorrt as trt

    return trt


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _engine_cache_dir() -> Path:
    raw = os.environ.get("COOLRL_TENSORRT_CACHE")
    if raw and raw.strip() not in {"0", "false", "no", "off"}:
        return Path(raw).expanduser()
    if raw and raw.strip().lower() in {"0", "false", "no", "off"}:
        return Path(tempfile.mkdtemp(prefix="coolrl-trt-"))
    return Path.home() / ".cache" / "coolrl" / "tensorrt"


def _hash_model_state(model: object) -> str:
    digest = hashlib.sha256()
    state_dict = model.state_dict()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _torch_dtype_for_trt(trt: Any, dtype: Any) -> "torch.dtype":
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.INT32:
        return torch.int32
    if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
        return torch.bfloat16
    raise RuntimeError(f"unsupported TensorRT tensor dtype: {dtype}")


class _TensorRTExportModule(_TorchModuleBase):
    def __init__(self, model: "torch.nn.Module") -> None:
        super().__init__()
        self.model = model

    def forward(self, features: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        logits, values = self.model(features)
        return torch.softmax(logits, dim=1), values


class TensorRTModelEvaluator(Evaluator):
    """TensorRT inference backend for Omok self-play and arena evaluation.

    TensorRT is CUDA-only. This class intentionally imports TensorRT lazily so
    macOS/Apple Silicon and CPU-only environments can keep using the torch
    evaluator without installing NVIDIA packages.
    """

    def __init__(self, model: object, device: str | None = None) -> None:
        _require_torch()
        self.device = _torch_device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                "TensorRT evaluator requires CUDA on an NVIDIA GPU. "
                f"Resolved evaluator device was {self.device!s}."
            )
        self.trt = _require_tensorrt()
        self.model = _coerce_torch_model(model).to(self.device).eval()
        self.board_size = int(getattr(self.model, "board_size"))
        self.action_size = int(getattr(self.model, "action_size"))
        self.max_batch_size = _env_int("COOLRL_TENSORRT_MAX_BATCH", 4096)
        self.opt_batch_size = min(self.max_batch_size, _env_int("COOLRL_TENSORRT_OPT_BATCH", 384))
        self.fp16 = _env_bool("COOLRL_TENSORRT_FP16", True)
        self.workspace_bytes = _env_int("COOLRL_TENSORRT_WORKSPACE_MB", 2048) * 1024 * 1024
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.engine = self._load_or_build_engine()
        self.context = self.engine.create_execution_context()
        self.input_name, self.output_names = self._io_names()
        if len(self.output_names) != 2:
            raise RuntimeError(f"TensorRT engine expected 2 outputs, got {self.output_names}")
        self.priors_name = "priors" if "priors" in self.output_names else self.output_names[0]
        self.values_name = "values" if "values" in self.output_names else self.output_names[1]

    def effective_batch_size(self, batch_size: int) -> int:
        return min(int(batch_size), self.max_batch_size)

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluate_features(states_to_feature_planes(states))

    def evaluate_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if features.ndim != 4 or features.shape[1:] != (4, self.board_size, self.board_size):
            raise ValueError(
                f"TensorRT evaluator expected features [N, 4, {self.board_size}, {self.board_size}], "
                f"got {features.shape}"
            )
        if features.shape[0] > self.max_batch_size:
            raise ValueError(
                f"TensorRT evaluator batch {features.shape[0]} exceeds max profile batch {self.max_batch_size}"
            )

        with torch.inference_mode():
            tensor = torch.as_tensor(np.ascontiguousarray(features), device=self.device)
            priors_shape = (tensor.shape[0], self.action_size)
            value_shape = (tensor.shape[0],)
            priors = torch.empty(priors_shape, dtype=self._tensor_dtype(self.priors_name), device=self.device)
            values = torch.empty(value_shape, dtype=self._tensor_dtype(self.values_name), device=self.device)
            stream = torch.cuda.current_stream(self.device)

            if hasattr(self.context, "set_input_shape"):
                self.context.set_input_shape(self.input_name, tuple(tensor.shape))
            elif hasattr(self.context, "set_binding_shape"):
                self.context.set_binding_shape(self._binding_index(self.input_name), tuple(tensor.shape))

            self._set_tensor_address(self.input_name, tensor)
            self._set_tensor_address(self.priors_name, priors)
            self._set_tensor_address(self.values_name, values)
            if not self._execute(stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            stream.synchronize()

            return (
                priors.detach().float().cpu().numpy().astype(np.float32, copy=False),
                values.detach().float().cpu().numpy().astype(np.float32, copy=False),
            )

    def _load_or_build_engine(self) -> Any:
        cache_dir = _engine_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._engine_cache_key()
        engine_path = cache_dir / f"{key}.engine"
        if engine_path.exists():
            runtime = self.trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
            if engine is None:
                raise RuntimeError(f"failed to deserialize TensorRT engine cache: {engine_path}")
            return engine

        serialized = self._build_serialized_engine()
        engine_path.write_bytes(bytes(serialized))
        runtime = self.trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("failed to deserialize newly built TensorRT engine")
        return engine

    def _engine_cache_key(self) -> str:
        cc = torch.cuda.get_device_capability(self.device)
        trt_version = getattr(self.trt, "__version__", "unknown")
        payload = "|".join(
            [
                f"trt={trt_version}",
                f"cc={cc[0]}{cc[1]}",
                f"board={self.board_size}",
                f"action={self.action_size}",
                f"max={self.max_batch_size}",
                f"opt={self.opt_batch_size}",
                f"fp16={int(self.fp16)}",
                _hash_model_state(self.model),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_serialized_engine(self) -> Any:
        try:
            import onnx  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("TensorRT evaluator export requires the optional `onnx` package.") from exc

        builder = self.trt.Builder(self.logger)
        flags = 1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = self.trt.OnnxParser(network, self.logger)
        config = builder.create_builder_config()
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(self.trt.MemoryPoolType.WORKSPACE, self.workspace_bytes)
        else:
            config.max_workspace_size = self.workspace_bytes
        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(self.trt.BuilderFlag.FP16)

        onnx_bytes = self._export_onnx()
        if not parser.parse(onnx_bytes):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "features",
            (1, 4, self.board_size, self.board_size),
            (max(1, self.opt_batch_size), 4, self.board_size, self.board_size),
            (max(1, self.max_batch_size), 4, self.board_size, self.board_size),
        )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        return serialized

    def _export_onnx(self) -> bytes:
        wrapper = _TensorRTExportModule(self.model).to(self.device).eval()
        dummy = torch.zeros(1, 4, self.board_size, self.board_size, dtype=torch.float32, device=self.device)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as fh:
            torch.onnx.export(
                wrapper,
                dummy,
                fh.name,
                input_names=["features"],
                output_names=["priors", "values"],
                dynamic_axes={
                    "features": {0: "batch"},
                    "priors": {0: "batch"},
                    "values": {0: "batch"},
                },
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
            return Path(fh.name).read_bytes()

    def _io_names(self) -> tuple[str, list[str]]:
        if hasattr(self.engine, "num_io_tensors"):
            inputs: list[str] = []
            outputs: list[str] = []
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT:
                    inputs.append(name)
                else:
                    outputs.append(name)
            if len(inputs) != 1:
                raise RuntimeError(f"TensorRT engine expected 1 input, got {inputs}")
            return inputs[0], outputs

        inputs = []
        outputs = []
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            if self.engine.binding_is_input(idx):
                inputs.append(name)
            else:
                outputs.append(name)
        if len(inputs) != 1:
            raise RuntimeError(f"TensorRT engine expected 1 input, got {inputs}")
        return inputs[0], outputs

    def _tensor_dtype(self, name: str) -> "torch.dtype":
        if hasattr(self.engine, "get_tensor_dtype"):
            return _torch_dtype_for_trt(self.trt, self.engine.get_tensor_dtype(name))
        return _torch_dtype_for_trt(self.trt, self.engine.get_binding_dtype(self._binding_index(name)))

    def _binding_index(self, name: str) -> int:
        if hasattr(self.engine, "get_binding_index"):
            return int(self.engine.get_binding_index(name))
        for idx in range(self.engine.num_bindings):
            if self.engine.get_binding_name(idx) == name:
                return idx
        raise RuntimeError(f"TensorRT binding not found: {name}")

    def _set_tensor_address(self, name: str, tensor: "torch.Tensor") -> None:
        if hasattr(self.context, "set_tensor_address"):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            return
        if not hasattr(self, "_bindings"):
            self._bindings = [0] * self.engine.num_bindings
        self._bindings[self._binding_index(name)] = int(tensor.data_ptr())

    def _execute(self, stream_handle: int) -> bool:
        if hasattr(self.context, "execute_async_v3"):
            return bool(self.context.execute_async_v3(stream_handle=stream_handle))
        return bool(self.context.execute_async_v2(bindings=self._bindings, stream_handle=stream_handle))
