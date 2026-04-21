# Setup

## 1) Install uv

If you don't have `uv` yet:

```bash
pip install uv
```

## 2) Create and sync environment

From the project root:

```bash
uv sync
```

This installs the base dependencies from `[project.dependencies]`.

## 3) Optional feature dependencies

This project defines extras in `[project.optional-dependencies]`:

- `poker`: `inquirer`
- `omok`: `torch`, `numpy`, `pygame`, `pyyaml`, `safetensors`
- `all`: installs both `poker` and `omok`

Install one or more extras as needed:

```bash
uv sync --extra poker
uv sync --extra omok
uv sync --extra omok-tensorrt
uv sync --extra all
uv sync --extra poker --extra omok
uv sync --all-extras
```

TensorRT is intentionally not part of the normal `omok` extra because it is
NVIDIA CUDA-only. For CUDA inference experiments, install NVIDIA TensorRT and
ONNX through `uv sync --extra omok-tensorrt`, then select
`selfplay.evaluator_backend: tensorrt` or `auto`. The extra skips TensorRT on
macOS so Apple Silicon setups keep the normal PyTorch/MPS path.

## 4) Run examples

Use the project docs for each module:

- Kuhn Poker: see `src/coolrl/kuhn_poker/tabular_cfr.md`
- Omok: see `src/coolrl/omok/README.md`

For Omok quick smoke test:

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device CPU
```
