# 설정

## 1) uv 설치

아직 `uv`가 없다면:
```bash
pip install uv
```
## 2) 환경 생성 및 동기화

프로젝트 루트에서:
```bash
uv sync
```
그러면 `[project.dependent]`에서 기본 종속성이 설치됩니다.

## 3) 선택적 기능 종속성

이 프로젝트는 `[project.ional-dependents]`에 추가 항목을 정의합니다.

- `포커`: `탐구자`
- `omok`: `torch`, `numpy`, `pygame`, `pyyaml`, `safetensors`
- `omok-tui`: `omok` + `onnxruntime` 및 `textual`
- `omok-tui-cuda`: `omok` + CUDA ONNX 런타임 및 `textual`
- `omok-tui-tensorrt`: `omok` + CUDA ONNX 런타임, TensorRT 및 `textual`
- `all`: `poker`, `omok`, `omok-tui`, `lost-cities`를 설치합니다.

필요에 따라 하나 이상의 추가 기능을 설치하십시오.
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
TensorRT는 의도적으로 일반적인 'omok' 추가 항목의 일부가 아닙니다.
NVIDIA CUDA 전용. CUDA 추론 실험을 위해서는 NVIDIA TensorRT를 설치하고
'uv sync --extra omok-tensorrt'를 통해 ONNX를 선택한 다음 선택하세요.
`selfplay.evaluator_backend: tensorrt` 또는 `auto`. 추가로 TensorRT를 건너뜁니다.
macOS이므로 Apple Silicon 설정은 일반적인 PyTorch/MPS 경로를 유지합니다.

## 4) 예제 실행

각 모듈에 대한 프로젝트 문서를 사용하십시오.

- Kuhn Poker: `src/coolrl/kuhn_poker/tabular_cfr.md`를 참조하세요.
- 오목: `src/coolrl/omok/README.md`를 참조하세요.

오목 빠른 연기 테스트의 경우:
```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device CPU
```
