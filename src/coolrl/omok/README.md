#오목RL

이 패키지는 PyTorch, 셀프 플레이 MCTS, 정책/가치로 Omok 에이전트를 교육합니다.
네트워크, 재생, 체크포인트, 경기장 승격, 파이게임 플레이, 브라우저 GUI 등이 있습니다.
보드 크기는 `rules.board_size`를 통해 구성됩니다. 같은 `coolrl.omok`
codepath는 네이티브 내에서 9x9, 13x13, 15x15 및 기타 정사각형 크기를 지원합니다.
백엔드 제한.

MacBook에서는 다음과 같은 경우 PyTorch MPS를 통해 Apple Silicon에서 훈련/평가를 실행할 수 있습니다.
그렇지 않으면 PyTorch CPU 대체가 사용됩니다.

## 빠른 시작

먼저 연기 구성을 사용하십시오. 이는 의도적으로 작으며 전체 파이프라인이 작동하는지 확인하기 위해 존재합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device CPU
```

짧은 로컬 교육 세션을 실행하세요.

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --device CPU
```

동일한 트레이너를 통해 15x15 연기 또는 빠른 사전 설정을 실행합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok15_smoke.yaml --device CPU
uv run python -m coolrl.omok.train --config configs/omok15_quick.yaml --device CPU
```

체크포인트 디렉터리에서 재개:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --resume checkpoints/omok_quick --device CPU
```

체크포인트를 ONNX로 내보내고 그에 대한 GUI를 엽니다.

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omok_quick \
    --output exports/omok_quick.onnx

uv run python -m coolrl.omok.gui --model exports/omok_quick.onnx

# For a 15x15 model, pass the matching board size.
uv run python -m coolrl.omok.gui --model exports/omok15_quick.onnx --board-size 15
```

ONNX 모델이 터미널 텍스트 UI에서 서로 재생되는 모습을 시청하세요.

```bash
uv run --extra omok-tui python -m coolrl.omok.tui \
    --model exports/omok_quick.onnx

uv run --extra omok-tui python -m coolrl.omok.tui \
    --black-model exports/black.onnx \
    --white-model exports/white.onnx \
    --board-size 15

# CUDA ONNX Runtime requires the CUDA extra.
uv run --extra omok-tui-cuda python -m coolrl.omok.tui \
    --model exports/omok_quick.onnx \
    --device cuda

# TensorRT requires the TensorRT TUI extra.
uv run --extra omok-tui-tensorrt python -m coolrl.omok.tui \
    --model exports/omok_quick.onnx \
    --device tensorrt

# Continuous model-vs-model run with a persistent score board.
uv run --extra omok-tui python -m coolrl.omok.tui \
    --model exports/omok_quick.onnx \
    --infinite
```

컴퓨터에 대한 전체 크기 프로필을 실행합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok_full_metal.yaml
uv run python -m coolrl.omok.train --config configs/omok_full_cuda.yaml
uv run python -m coolrl.omok.train --config configs/omok15_full_cuda.yaml
```

전체 프로필은 기본 MCTS 백엔드를 사용합니다. C 백엔드는 다음 위치에 있습니다.
`src/coolrl/omok/cmcts/`; Rust 백엔드는 아래에 있습니다.
`src/coolrl/omok/rmcts/` 및 `selfplay.mcts_backend: Rust`로 선택할 수 있습니다.
소스 체크아웃에서 컴파일된 C 확장을 찾을 수 없는 경우 해당 위치에 빌드하세요.

```bash
uv run --with setuptools python setup.py build_ext --inplace
```

유용한 GUI 키:

- 왼쪽 클릭: 돌을 놓습니다.
-`R`: 게임을 재설정합니다.
- `S`: 인간 측을 교환합니다.
- `M`: AI를 강제로 이동시킵니다.
- `O`: 현재 시드에서 결정론적 오프닝을 적용합니다.
- `[` / `]`: 오프닝 시드를 줄이거나 늘립니다.
- `Esc`: 종료합니다.

GUI CLI 옵션:

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--모델 FILE.onnx` | 없음 | ONNX 모델 파일. 2인용 또는 UI 테스트는 생략하세요. |
| `--보드 크기 N` | `9` | 플레이를 위한 보드 크기. 모델이 로드될 때 ONNX 정책 출력 길이와 일치해야 합니다. |
| `--device auto\|cpu\|cuda\|coreml` | '자동' | ONNX 런타임 실행 공급자. |
| `--시뮬레이션 N` | `64` | AI 이동당 MCTS 시뮬레이션. |
| `--인간색 검정색\|흰색` | '백인' | 인간이 어떤 색을 연주하는지. |
| `--시드 N` | `0` | 오프닝 시드(게임 내 `[`/`]`로 조정 가능). |

TUI CLI 옵션:

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--모델 FILE.onnx` | 없음 | 양면에 단일 ONNX 모델이 사용됩니다. |
| `--블랙-모델 FILE.onnx` / `--화이트-모델 FILE.onnx` | 없음 | 색상별 ONNX 모델. |
| `--보드 크기 9\|15` | `9` | 보드 크기. 모델 정책 출력과 일치해야 합니다. |
| `--device auto\|cpu\|cuda\|tensorrt\|coreml` | '자동' | ONNX 런타임 실행 공급자. |
| `--시뮬레이션 N` | `256` | 이동당 MCTS 시뮬레이션. |
| `--이동-지연 SEC` | `0.05` | 표시된 동작 사이의 지연입니다. |
| `--무한` | 떨어져 | 각 터미널 결과가 나온 후 새로운 시드 게임을 시작하세요. `--seed`를 무시합니다. |
| `--디버그 라인 N` | '1000' | 디버그 콘솔에 스크롤백이 유지됩니다. |

## 구성

`configs/omok_smoke.yaml`은 9x9 배관 검사입니다.

- 1회 반복.
- 셀프 플레이 게임 1회.
- 이동당 2개의 MCTS 시뮬레이션.
- 소규모 네트워크: `채널=16`, `블록=1`.
- 경기장이 비활성화되었습니다.

`configs/omok_quick.yaml`은 짧은 9x9 실제 실행입니다.

- 20번 반복.
- 반복당 8개의 셀프 플레이 게임.
- 배치당 8개의 활성 셀프 플레이 게임(`games_per_iteration`과 일치).
- MCTS 일정은 8~16개 시뮬레이션입니다.
- 대규모 네트워크: `채널=32`, `블록=2`.
- 반복당 16개의 최적화 업데이트.
- 작은 경기장이 활성화되었습니다.

실제 전체 실행의 경우 하드웨어별 사전 설정을 선호합니다.

- `configs/omok_full_cuda.yaml`: NVIDIA/개별 GPU 프로필. `device: CUDA`, C MCTS, `evaluator_backend: torch`, `num_workers: 0`, `batch_size: 64` 및 `leaves_per_batch: 64`를 사용하므로 셀프 플레이와 경기장 추론이 대규모 배치로 GPU에 유지됩니다.
- `configs/omok_full_metal.yaml`: Apple Silicon 프로필. `device: METAL`, C MCTS, 더 작은 셀프 플레이 청크, `num_workers: auto` 및 CPU 작업자 병렬성을 사용하므로 생성된 프로세스 전체에서 공유 Metal 컨텍스트를 피하면서 여러 게임을 동시에 생성할 수 있습니다.
- `configs/omok15_smoke.yaml`, `configs/omok15_quick.yaml` 및 `configs/omok15_full_cuda.yaml`: `rules.board_size: 15`와 함께 통합 `coolrl.omok` 트레이너를 사용하는 15x15 사전 설정입니다.

호환성 참고 사항: 전체 프로필은 `use_amp`와 같은 참조 필드를 유지합니다.
`search_threads`, `inference_batch_size`, `inference_wait_ms`, `virtual_loss` 및
구성 호환성을 위한 `grad_clip`. C 백엔드는 트리 수준에 `search_threads`를 사용합니다.
활성 게임 전체에서 병렬 수집을 수행하지만 동일한 트리를 구현하지 않습니다.
가상 손실 검색 또는 비동기 추론 대기열. 활성 셀프 플레이 처리량
노브는 `selfplay.batch_size`, `selfplay.num_workers`,
`selfplay.leaves_per_batch`, `selfplay.search_threads` 및
`selfplay.evaluator_backend`는 호환성을 위해 스키마에 남아 있지만 지원되는 런타임 평가기는 PyTorch입니다.

장기간 실행하려면 `configs/omok_quick.yaml`을 복사하고 다음을 늘리세요.

- `max_iterations`
- `selfplay.games_per_iteration`
- `selfplay.simulation_schedule[].simulations`
- `optimization.updates_per_iteration`
- `아레나.게임`

## 셀프 플레이 병렬화

여기에는 네 가지 종류의 병렬 처리가 있습니다.

GPU 커널 병렬 처리는 가능한 경우 CUDA 및 MPS에서 활성화됩니다. 이것이 가장
신경망 추론 및 훈련에 중요한 계층입니다.

학습 배치 병렬성은 다음을 통해 제어됩니다.

```yaml
optimization:
  batch_size: 64
```

배치가 클수록 최적화 단계당 가속기에 더 많은 작업이 제공되지만
더 많은 메모리. 16GB RAM을 갖춘 Apple M2에서는 '32' 또는 '64'로 시작하세요. '128'을 시도해 보세요
실행이 안정된 후에야. 개별 GPU에서는 '256'이 현재 전체 GPU입니다.
프로필 기본값.

자체 재생 검색 처리량은 주로 다음에 의해 제어됩니다.

```yaml
selfplay:
  games_per_iteration: 16
  batch_size: 4
  leaves_per_batch: 8
  simulation_schedule:
    - fraction: 0.0
      simulations: 16
```

`selfplay.batch_size`는 한 게임 내에서 활성 상태로 유지되는 게임 수를 제어합니다.
`MCTS.search_batch(...)` 호출. `selfplay.leaves_per_batch`는 얼마나 많은 수를 제어합니다.
MCTS는 각 활성 게임이 하나의 신경망 평가 전에 기여하도록 둡니다.
대략적인 추론 배치 크기를 함께 설정합니다.

```text
active games * leaves_per_batch
```

CUDA 전체 자체 플레이의 경우 대규모 평가자 호출당 '64 * 64 = 4096' 위치는 다음과 같습니다.
현재 높은 처리량 조정 목표입니다. 이 값은 다음을 수행하는 데 좋은 기준이 됩니다.
하드웨어를 다시 청소하십시오.
RTX 3090 측정에 대해서는 `docs/omok_cuda_tuning.md`를 참조하세요.

다중 프로세스 자체 재생은 다음을 통해 제어됩니다.

```yaml
selfplay:
  num_workers: auto  # or an integer like 4
```

허용되는 값:

- `auto`: 시작 시 `os.cpu_count()`로 확인되고 확인된 값을 기록합니다. `셀프 플레이 num_workers=auto가 8(os.cpu_count=8)로 해결되었습니다`. 구성을 편집하지 않고도 여러 컴퓨터에 이식 가능합니다.
- `0`: 다중 프로세스를 비활성화하고 레거시 단일 프로세스 경로를 유지합니다.
- 임의의 양의 정수: 고정된 수의 작업자 프로세스.

확인된 값이 `>= 1`이면 자체 재생 생성이
CPU 작업자의 `ProcessPoolExecutor`. 각 근로자는 현재의 사본을 받습니다.
가중치를 모델화하고(numpy 배열로) CPU에서 토치 `PolicyValueNet`을 재구성하고 실행합니다.
MCTS + 게임은 독립적입니다. 주요 프로세스는 구성된 훈련 장치를 유지합니다
따라서 가속기 컨텍스트는 프로세스 로컬로 유지됩니다. 결과는 풀을 통해 다시 수집됩니다.
메인 프로세스의 공유 재생 버퍼에 추가됩니다.

CPU 작업자가 필요한 이유: 가속기 컨텍스트를 프로세스 로컬로 유지합니다. Apple Silicon에서는 이것이 가능합니다.
CPU 워커는 계속해서 자체 플레이를 할 수 있기 때문에 합리적인 절충안이 필요합니다.
기본 프로세스는 MPS에서 업데이트를 실행합니다. 개별 NVIDIA GPU에서는 장단점이 다릅니다.
작업자 경로는 자체 재생 추론을 CUDA에서 이동하고 전체 CUDA보다 느릴 수 있습니다.
자기 플레이. CUDA 전체 실행에는 'num_workers: 0'을 사용하세요.

기본값과 장단점:

- `num_workers: 0`(기본값): 단일 프로세스 자체 재생. CUDA 전체 실행, 디버깅, 스모크 실행 및 프로파일링이 결정적이어야 하는 모든 경우에 이 옵션을 선택하세요.
- `num_workers: 1`: 작업자 프로세스 1개. 거의 유용하지 않습니다. 프로세스 격리를 특별히 원하지 않는 한 대신 '0'을 사용하세요.
- `num_workers: 2` ~ `os.cpu_count() - 1`: CPU/Metal 스타일 자체 플레이에 유용합니다. 더 많은 작업자 = 비행 중인 게임이 더 많아지지만 물리적 코어를 초과하면 수익이 감소합니다.

시작 비용: 각 작업자에게는 시작 오버헤드가 발생합니다. 소규모 구성의 경우(연기, 빠른)
이는 단일 반복을 지배할 수 있습니다. 중간 및 전체 구성의 경우 풀 비용은 다음과 같습니다.
반복당 많은 MCTS 호출에 걸쳐 상각됩니다. 셀프 플레이마다 새로운 풀이 생성됩니다.
반복당 소스(후보와 최고는 각자 자신의 것을 얻음), 작업자는
`ProcessPoolExecutor(initializer=...)`를 통해 한 번 초기화되었으므로 모델 가중치는 다음과 같습니다.
청크별로 재배송되지 않습니다.

작업 청크: 오프닝은 `selfplay.batch_size` 청크로 분할되며 각 청크는 풀에 제출된 하나의 작업입니다. 청크 내에서 `MCTS.search_batch`는 여전히 리프를 일괄 처리하므로 청크별 및 리프별 일괄 처리가 모두 활성화됩니다.

가장 중요한 손잡이는 다음과 같습니다.

- 반복당 더 많은 데이터를 얻으려면 'games_per_iteration'을 늘리세요.
- CUDA의 경우 'num_workers: 0'을 유지하고 'selfplay.batch_size'를 늘립니다. /
CUDA에 충분히 큰 추론이 제공될 때까지 `selfplay.leaves_per_batch`
배치.
- Metal/CPU 작업자의 경우 `selfplay.num_workers`를 늘려 CPU 코어를 포화시킵니다.
여러 청크를 생성할 수 있을 만큼 `selfplay.batch_size`를 작게 유지하세요.
- 더 강력한 MCTS 목표를 위해 '시뮬레이션'을 늘립니다.
- 더 많은 네트워크 학습을 위해 'optimization.batch_size' 및 'updates_per_iteration'을 늘리세요.
- 아레나 게임도 MCTS가 많기 때문에 튜닝 중에 'arena.games'를 작게 유지하세요.

## 추천 레시피

빠른 상태 점검:

```yaml
max_iterations: 1
selfplay:
  games_per_iteration: 1
  simulation_schedule:
    - fraction: 0.0
      simulations: 2
optimization:
  batch_size: 4
  updates_per_iteration: 1
arena:
  games: 0
```

MacBook 빠른 튜닝:

```yaml
max_iterations: 20
selfplay:
  mcts_backend: c
  games_per_iteration: 8
  batch_size: 8        # match games_per_iteration so all games join every batch
  leaves_per_batch: 4  # evaluate 4 leaves per MCTS step to increase METAL batch size
  simulation_schedule:
    - fraction: 0.0
      simulations: 8
    - fraction: 0.5
      simulations: 16
optimization:
  batch_size: 64
  updates_per_iteration: 16
arena:
  games: 2
  simulations: 8
```

더 무거운 밤새 MacBook 실행:

```yaml
max_iterations: 200
network:
  channels: 48
  blocks: 3
  value_hidden: 96
selfplay:
  mcts_backend: c
  games_per_iteration: 16
  batch_size: 4
  num_workers: 4      # CPU self-play workers; tune to physical cores
  leaves_per_batch: 8
  simulation_schedule:
    - fraction: 0.0
      simulations: 16
    - fraction: 0.5
      simulations: 32
optimization:
  batch_size: 64
  updates_per_iteration: 32
  replay_capacity: 50000
arena:
  games: 4
  simulations: 16
```

기계가 느려지면 먼저 '시뮬레이션'을 줄이세요. 메모리 부족이 증가하면 'optimization.batch_size' 또는 네트워크 '채널'을 줄이세요.

## 토치 디버그 연기

전체 파이프라인 검사를 위해 다음 1회 반복 연기를 사용하세요.

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --max-iterations 1 --device CPU
```

전체 GPU 문제 해결 흐름을 보려면 표준 PyTorch 도구(`torch.profiler`,
`nsys`, 벤더 프로파일러 등) smoke 명령이 안정적인지 확인한 후.

## 웹 GUI

브라우저 기반 GUI는 WASM(ONNX 런타임 웹)을 통해 클라이언트 측에서 ONNX 모델을 완전히 실행합니다.

PyTorch '.pt' 체크포인트를 ONNX로 내보내기:

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omokai_converted/best \
    --output exports/best.onnx
```

전체 디렉터리 내보내기:

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omokai_converted \
    --output exports/omokai
```

omokai에서 기존 ONNX 체크포인트를 레거시 '.safetensors' 가중치 파일로 가져옵니다.

```bash
uv run --with onnx python -m coolrl.omok.convert_onnx \
    --source /path/to/omokai/web/models \
    --output checkpoints/omokai_converted
```

웹 GUI를 제공하고 브라우저에서 엽니다.

```bash
cd src/coolrl/omok/web && python -m http.server 8080
```

그런 다음 `http://localhost:8080`을 열고 **Load .onnx**를 클릭한 후 내보낸 모델을 업로드합니다. 컨트롤: 클릭하여 돌 배치, 재설정, 실행 취소, 측면 교체, 강제 AI 이동 및 시뮬레이션 슬라이더(4–512)를 사용할 수 있습니다.
모델을 로드하기 전에 보드 크기 선택기를 사용하십시오. 웹 UI는
ONNX 정책 출력 길이는 선택한 보드 크기와 일치합니다.

## 측정항목 시각화

완료되었거나 진행 중인 실행에서 훈련 측정항목을 플롯합니다.

```bash
# Save metrics.png in the checkpoint directory
omok-plot checkpoints/omok_full_cuda

# Open an interactive window instead
omok-plot checkpoints/omok_full_cuda --show

# Custom output path
omok-plot checkpoints/omok_full_cuda -o ~/reports/run1.png
```

`metrics.jsonl` 파일을 직접 가리킬 수도 있습니다.

```bash
omok-plot checkpoints/omok_full_cuda/metrics.jsonl
```

보고서는 열차 손실, 정책/가치 손실, 경기장 승률(허용 모델 마커 포함), 자체 플레이 평균 이동, 재생 버퍼 크기 및 경과 시간 등 2×3 그리드입니다. 정책 엔트로피 기준선은 실행에 기록된 `board_size`에서 파생되므로 9x9 및 15x15 실행은 서로 다른 균일 기준선을 사용합니다.

## 출력

학습은 구성된 체크포인트 디렉터리에 기록됩니다.

- `latest.pt`: 현재 후보 모델.
- `best.pt`: 가장 승격된 모델입니다.
- `iter_XXXX.pt`: 활성화된 경우 반복 스냅샷.
- `trainer_state.json`: 반복 카운터 및 실행 메타데이터.
- `replay.pkl`: 재생 버퍼.
- `metrics.jsonl`: 반복당 하나의 JSON 측정항목 레코드입니다.
- `runtime_progress.json`: 최신 진행 상황 스냅샷.

레거시 '.safetensors' 체크포인트는 여전히 모델 가중치 전용 시드 입력으로 사용될 수 있습니다. 최적화 프로그램 상태를 복원하지 않습니다. 새로운 훈련 체크포인트는 PyTorch `.pt` 전용입니다.
