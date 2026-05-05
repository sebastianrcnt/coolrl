# 설정

## 1) uv 설치

`uv`가 없다면:

```bash
pip install uv
```

## 2) 환경 생성 및 동기화

프로젝트 루트에서:

```bash
uv sync
```

이는 `[project.dependencies]`에서 기본 의존성을 설치합니다.

## 3) 선택적 기능 의존성

이 프로젝트는 `[project.optional-dependencies]`에 extras를 정의합니다:

- `poker`: `inquirer`
- `omok`: `torch`, `numpy`, `pygame`, `pyyaml`, `safetensors`
- `omok-tui`: `omok` plus `onnxruntime` and `textual`
- `omok-tui-cuda`: `omok` plus CUDA ONNX Runtime and `textual`
- `omok-tui-tensorrt`: `omok` plus CUDA ONNX Runtime, TensorRT, and `textual`
- `all`: `poker`, `omok`, `omok-tui`, and `lost-cities` 설치

필요에 따라 하나 이상의 extras를 설치합니다:

```bash
uv sync --extra poker
uv sync --extra omok
uv sync --extra omok-tui
uv sync --extra omok-tui-cuda
uv sync --extra omok-tui-tensorrt
uv sync --extra omok-tensorrt
uv sync --extra all
uv sync --extra poker --extra omok
uv sync --all-extras
```

TensorRT는 NVIDIA CUDA 전용이므로 일반적인 `omok` extra에 포함되지 않습니다. CUDA inference 실험을 위해 `uv sync --extra omok-tensorrt`를 통해 NVIDIA TensorRT와 ONNX를 설치한 후, `selfplay.evaluator_backend: tensorrt` 또는 `auto`를 선택합니다. 이 extra는 macOS에서 TensorRT를 건너뛰어 Apple Silicon 설정이 일반적인 PyTorch/MPS 경로를 유지하게 합니다.

## 4) 예제 실행

각 모듈의 프로젝트 문서를 사용합니다:

- Kuhn Poker: `src/coolrl/kuhn_poker/tabular_cfr.md` 참조
- Omok: `src/coolrl/omok/README.md` 참조

Omok 빠른 테스트용:

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device CPU
```

## 5) 문서 링크 확인

문서의 로컬 링크가 깨졌는지 확인하려면 `lychee` CLI가 필요합니다. `lychee`는 Python 패키지가 아니므로 별도로 설치합니다:

```bash
cargo install lychee
```

프로젝트 루트에서 다음 명령을 실행합니다:

```bash
uv run check-doc-links
```

이 명령은 `README.md`, `docs/**/*.md`, `src/**/*.md`, `configs/**/*.md`의 Markdown 링크를 `lychee --offline`으로 검사합니다. 외부 URL은 네트워크로 확인하지 않습니다.
