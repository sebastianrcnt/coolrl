# Omok RL

이 패키지는 PyTorch, self-play MCTS, policy/value network, replay, checkpointing, arena promotion, Pygame play, browser GUI를 사용하여 Omok agents를 훈련합니다. Board 크기는 `rules.board_size`를 통해 구성되며, 동일한 `coolrl.omok` codepath는 native backend limits 내에서 9x9, 13x13, 15x15 및 기타 정사각형 크기를 지원합니다.

MacBook에서는 PyTorch MPS를 통해 Apple Silicon에서 training/evaluation을 실행할 수 있으며, 사용 불가능한 경우 PyTorch CPU fallback을 사용합니다.

## 빠른 시작

먼저 smoke config를 사용하세요. 이는 의도적으로 작으며 전체 pipeline이 작동하는지 확인하기 위해 존재합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device CPU
```

짧은 로컬 training session을 실행합니다:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --device CPU
```

같은 trainer를 통해 15x15 smoke 또는 quick presets을 실행합니다:

```bash
uv run python -m coolrl.omok.train --config configs/omok15_smoke.yaml --device CPU
uv run python -m coolrl.omok.train --config configs/omok15_quick.yaml --device CPU
```

checkpoint 디렉토리에서 재개합니다:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --resume checkpoints/omok_quick --device CPU
```

checkpoint를 ONNX로 내보내고 GUI를 열기:

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omok_quick \
    --output exports/omok_quick.onnx

uv run python -m coolrl.omok.gui --model exports/omok_quick.onnx

# 15x15 모델의 경우 일치하는 board 크기를 전달합니다.
uv run python -m coolrl.omok.gui --model exports/omok15_quick.onnx --board-size 15
```

terminal Textual UI에서 ONNX 모델들이 서로 경기하는 것을 봅니다:

```bash
uv run --extra omok-tui python -m coolrl.omok.tui \
    --model exports/omok_quick.onnx

uv run --extra omok-tui python -m coolrl.omok.tui \
    --black-model exports/black.onnx \
    --white-model exports/white.onnx \
    --board-size 15

# CUDA ONNX Runtime은 CUDA extra가 필요합니다.
uv run --extra omok-tui-cuda python -m coolrl.omok.tui \
    --model exports/omok_quick.onnx \
    --device cuda

# TensorRT는 TensorRT TUI extra가 필요합니다.
uv run --extra omok-tui-tensorrt python -m coolrl.omok.tui \
    --model exports/omok_quick.onnx \
    --device tensorrt

# 지속적인 scoreboard를 가진 model-vs-model 실행.
uv run --extra omok-tui python -m coolrl.omok.tui \
    --model exports/omok_quick.onnx \
    --infinite
```

머신을 위한 full-sized profile 실행:

```bash
uv run python -m coolrl.omok.train --config configs/omok_full_metal.yaml
uv run python -m coolrl.omok.train --config configs/omok_full_cuda.yaml
uv run python -m coolrl.omok.train --config configs/omok15_full_cuda.yaml
```

전체 profiles은 native MCTS backends를 사용합니다. C backend는 `src/coolrl/omok/cmcts/` 아래에 있습니다. Rust backend는 `src/coolrl/omok/rmcts/` 아래에 있으며 `selfplay.mcts_backend: rust`로 선택할 수 있습니다. Source checkout이 컴파일된 C extension을 찾을 수 없으면, in-place로 빌드합니다:

```bash
uv run --with setuptools python setup.py build_ext --inplace
```

유용한 GUI 키:

- Left click: 돌을 놓음.
- `R`: 게임 초기화.
- `S`: 인간 side 교환.
- `M`: AI move 강제.
- `O`: 현재 seed에서 deterministic opening 적용.
- `[` / `]`: opening seed 감소 또는 증가.
- `Esc`: 종료.

GUI CLI 옵션:

| Flag | Default | Description |
|---|---|---|
| `--model FILE.onnx` | none | ONNX model 파일. 2-player 또는 UI testing을 위해 생략합니다. |
| `--board-size N` | `9` | Play를 위한 board 크기. 모델이 로드될 때 ONNX policy output 길이와 일치해야 합니다. |
| `--device auto\|cpu\|cuda\|coreml` | `auto` | ONNX Runtime execution provider. |
| `--simulations N` | `64` | AI move당 MCTS simulations. |
| `--human-color black\|white` | `white` | 인간이 플레이하는 색. |
| `--seed N` | `0` | Opening seed (게임 중 `[`/`]`로도 조정 가능). |

TUI CLI 옵션:

| Flag | Default | Description |
|---|---|---|
| `--model FILE.onnx` | none | 양쪽 모두에 사용되는 단일 ONNX model. |
| `--black-model FILE.onnx` / `--white-model FILE.onnx` | none | Color-specific ONNX models. |
| `--board-size 9\|15` | `9` | Board 크기. 모델 policy output과 일치해야 합니다. |
| `--device auto\|cpu\|cuda\|tensorrt\|coreml` | `auto` | ONNX Runtime execution provider. |
| `--simulations N` | `256` | Move당 MCTS simulations. |
| `--move-delay SEC` | `0.05` | 표시된 moves 사이의 delay. |
| `--infinite` | off | 각 terminal result 후 새로운 seeded game 시작. `--seed`를 무시합니다. |
| `--debug-lines N` | `1000` | Debug console에 보관되는 scrollback. |

## Configs

`configs/omok_smoke.yaml`는 9x9 plumbing check입니다:

- 1 iteration.
- 1 self-play game.
- Move당 2 MCTS simulations.
- small network: `channels=16`, `blocks=1`.
- arena disabled.

`configs/omok_quick.yaml`는 짧은 9x9 real run입니다:

- 20 iterations.
- Iteration당 8 self-play games.
- Batch당 8 active self-play games (`games_per_iteration`과 일치).
- 8에서 16 simulations로의 MCTS schedule.
- larger network: `channels=32`, `blocks=2`.
- Iteration당 16 optimizer updates.
- Small arena enabled.

실제 full runs의 경우, hardware-specific presets을 선호합니다:

- `configs/omok_full_cuda.yaml`: NVIDIA/discrete GPU profile. `device: CUDA`, C MCTS, `evaluator_backend: torch`, `num_workers: 0`, `batch_size: 64`, `leaves_per_batch: 64`를 사용하므로 self-play 및 arena inference이 큰 batches에서 GPU에 유지됩니다.
- `configs/omok_full_metal.yaml`: Apple Silicon profile. `device: METAL`, C MCTS, 더 작은 self-play chunks, `num_workers: auto`, CPU worker parallelism을 사용하므로 spawned processes 간 shared Metal contexts를 회피하면서 여러 게임을 동시에 생성할 수 있습니다.
- `configs/omok15_smoke.yaml`, `configs/omok15_quick.yaml`, `configs/omok15_full_cuda.yaml`: `rules.board_size: 15`와 함께 unified `coolrl.omok` trainer를 사용하는 15x15 presets.

호환성 참고: full profiles는 config 호환성을 위해 `use_amp`, `search_threads`, `inference_batch_size`, `inference_wait_ms`, `virtual_loss`, `grad_clip` 등의 reference 필드를 유지합니다. C backend는 active games 간 tree-level parallel collection을 위해 `search_threads`를 사용하지만, same-tree virtual-loss search나 async inference queues는 구현하지 않습니다. Active self-play throughput knobs는 `selfplay.batch_size`, `selfplay.num_workers`, `selfplay.leaves_per_batch`, `selfplay.search_threads`이고, `selfplay.evaluator_backend`는 호환성을 위해 schema에 유지되지만, 지원되는 runtime evaluator는 PyTorch입니다.

더 긴 run의 경우, `configs/omok_quick.yaml`을 복사하고 증가시킵니다:

- `max_iterations`
- `selfplay.games_per_iteration`
- `selfplay.simulation_schedule[].simulations`
- `optimization.updates_per_iteration`
- `arena.games`

## Self-Play Parallelization

여기에는 4가지 다른 종류의 parallelism이 있습니다.

GPU kernel parallelism은 사용 가능할 때 CUDA와 MPS에서 활성화됩니다. 이는 neural network inference와 training을 위한 가장 중요한 레이어입니다.

Training batch parallelism은 다음과 같이 제어됩니다:

```yaml
optimization:
  batch_size: 64
```

더 큰 batches는 optimizer step당 더 많은 작업을 accelerator에 공급하지만 더 많은 메모리를 사용합니다. 16GB RAM이 있는 Apple M2에서는 `32` 또는 `64`로 시작합니다. run이 안정적인 후에만 `128`을 시도하세요. 이산 GPU에서는 `256`이 현재 full profile 기본값입니다.

Self-play search throughput은 주로 다음과 같이 제어됩니다:

```yaml
selfplay:
  games_per_iteration: 16
  batch_size: 4
  leaves_per_batch: 8
  simulation_schedule:
    - fraction: 0.0
      simulations: 16
```

`selfplay.batch_size`는 하나의 `MCTS.search_batch(...)` 호출 내에 활성화된 게임의 수를 제어합니다. `selfplay.leaves_per_batch`는 각 active game이 한 번의 neural network evaluation 전에 기여하는 MCTS leaves의 수를 제어합니다. 함께 그들은 대략적인 inference batch 크기를 설정합니다:

```text
active games * leaves_per_batch
```

CUDA full self-play의 경우, `64 * 64 = 4096` positions per large evaluator call이 현재 high-throughput tuning 목표입니다. 이 값은 당신의 hardware에 대해 다시 sweep하기에 좋은 baseline입니다.
RTX 3090 measurements는 `docs/omok_cuda_tuning.md`를 참조하세요.

Multi-process self-play은 다음과 같이 제어됩니다:

```yaml
selfplay:
  num_workers: auto  # or an integer like 4
```

수락된 값:

- `auto`: startup에서 `os.cpu_count()`로 해석되고 해석된 값을 로그하거나, 예: `Self-play num_workers=auto resolved to 8 (os.cpu_count=8)`. Config 편집 없이 머신 간 이식성.
- `0`: multi-process를 비활성화하고 legacy single-process path를 유지합니다.
- 모든 양수: fixed number of worker processes.

해석된 값이 `>= 1`이면, self-play generation은 CPU workers의 `ProcessPoolExecutor`에 dispatch됩니다. 각 worker는 현재 model weights의 복사본(numpy arrays로)을 받고, CPU에서 torch `PolicyValueNet`을 재구성하고, MCTS + games을 독립적으로 실행합니다. Main process는 구성된 training device를 유지하므로 accelerator contexts는 process-local로 유지됩니다. 결과는 pool을 통해 수집되어 main process의 shared replay buffer에 추가됩니다.

CPU workers인 이유: 이는 accelerator contexts를 process-local로 유지합니다. Apple Silicon에서는 CPU workers가 self-play을 진행할 수 있으므로 합리적인 trade-off가 될 수 있고, main process는 MPS에서 updates를 실행합니다. 이산 NVIDIA GPU에서는 trade-off가 다릅니다: worker path는 self-play inference을 CUDA에서 이동시키고 full CUDA self-play보다 느릴 수 있습니다. CUDA full runs의 경우 `num_workers: 0`를 사용하세요.

Defaults 및 trade-offs:

- `num_workers: 0` (default): single-process self-play. CUDA full runs, debugging, smoke runs, profiling이 deterministic이어야 하는 모든 경우에 선택합니다.
- `num_workers: 1`: one worker process. 거의 유용하지 않음 — process isolation을 특별히 원하지 않는 한 대신 `0`를 사용하세요.
- `num_workers: 2` to `os.cpu_count() - 1`: CPU/Metal-style self-play에 유용합니다. 더 많은 workers = 더 많은 games in flight, 하지만 physical cores를 초과하면 diminishing returns.

Startup cost: 각 worker는 startup overhead를 발생시킵니다. Small configs(smoke, quick)의 경우 이는 단일 iteration을 주도할 수 있습니다. Medium 및 full configs의 경우 pool cost는 iteration당 많은 MCTS 호출에 걸쳐 amortized됩니다. 새로운 pool은 iteration당 self-play source마다 생성되고(candidate와 best는 각각 고유), workers는 `ProcessPoolExecutor(initializer=...)`를 통해 한 번만 초기화되므로 model weights는 chunk마다 재배송되지 않습니다.

Work chunking: openings은 `selfplay.batch_size`의 chunks로 split되고 각 chunk는 pool에 제출된 하나의 task입니다. Chunk 내에서 `MCTS.search_batch`는 여전히 leaves을 함께 batch하므로 per-chunk 및 per-leaf batching 모두 활성화됩니다.

가장 중요한 knobs:

- Iteration당 더 많은 데이터를 위해 `games_per_iteration`을 증가시킵니다.
- CUDA의 경우, `num_workers: 0`을 유지하고 `selfplay.batch_size` / `selfplay.leaves_per_batch`를 CUDA가 충분히 큰 inference batches로 공급될 때까지 증가시킵니다.
- Metal/CPU workers의 경우, CPU cores를 saturate하기 위해 `selfplay.num_workers`를 증가시키고 여러 chunks를 생성할 수 있을 정도로 `selfplay.batch_size`를 작게 유지합니다.
- 더 강한 MCTS targets를 위해 `simulations`을 증가시킵니다.
- 더 많은 network learning을 위해 `optimization.batch_size` 및 `updates_per_iteration`을 증가시킵니다.
- Tuning 중에는 `arena.games`을 작게 유지합니다. arena games도 MCTS-heavy이기 때문입니다.

## 추천 레시피

빠른 sanity check:

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

MacBook quick tuning:

```yaml
max_iterations: 20
selfplay:
  mcts_backend: c
  games_per_iteration: 8
  batch_size: 8        # 모든 게임이 모든 batch에 참여하도록 games_per_iteration과 일치
  leaves_per_batch: 4  # METAL batch 크기를 증가시키기 위해 MCTS step당 4 leaves 평가
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

더 무거운 overnight MacBook run:

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
  num_workers: 4      # CPU self-play workers; physical cores로 tune
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

머신이 느려지면 먼저 `simulations`을 감소시킵니다. 메모리 압력이 증가하면 `optimization.batch_size` 또는 network `channels`를 감소시킵니다.

## Torch Debug Smoke

전체 pipeline check를 위해 이 one-iteration smoke를 사용합니다:

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --max-iterations 1 --device CPU
```

전체 GPU troubleshooting flow의 경우, smoke 명령이 안정적인지 확인한 후 표준 PyTorch tooling(`torch.profiler`, `nsys`, vendor profiler 등)을 사용합니다.

## Web GUI

Browser-based GUI는 ONNX Runtime Web(WASM)을 통해 ONNX 모델을 완전히 client-side에서 실행합니다.

PyTorch `.pt` checkpoints를 ONNX로 내보냅니다:

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omokai_converted/best \
    --output exports/best.onnx
```

전체 디렉토리를 내보냅니다:

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omokai_converted \
    --output exports/omokai
```

omokai에서 legacy `.safetensors` weight 파일로 기존 ONNX checkpoints를 import합니다:

```bash
uv run --with onnx python -m coolrl.omok.convert_onnx \
    --source /path/to/omokai/web/models \
    --output checkpoints/omokai_converted
```

Web GUI를 serve하고 browser에서 열기:

```bash
cd src/coolrl/omok/web && python -m http.server 8080
```

그 다음 `http://localhost:8080`을 열고, **Load .onnx**를 클릭하고, 내보낸 모델을 업로드합니다. Controls: click to place a stone, Reset, Undo, Swap side, force AI Move, simulations slider(4–512).
모델을 로드하기 전에 board-size selector를 사용하세요. Web UI는 ONNX policy output 길이가 선택된 board 크기와 일치하는지 확인합니다.

## 메트릭 시각화

완료되었거나 진행 중인 run에서 training metrics을 plot합니다:

```bash
# checkpoint 디렉토리에 metrics.png를 저장
omok-plot checkpoints/omok_full_cuda

# 대신 interactive window를 열기
omok-plot checkpoints/omok_full_cuda --show

# Custom output path
omok-plot checkpoints/omok_full_cuda -o ~/reports/run1.png
```

직접 `metrics.jsonl` 파일을 가리킬 수도 있습니다:

```bash
omok-plot checkpoints/omok_full_cuda/metrics.jsonl
```

Report는 2×3 grid입니다: train loss, policy/value loss, arena win rate(accepted-model markers 포함), selfplay average moves, replay buffer size, elapsed hours. Policy entropy reference 라인은 run의 기록된 `board_size`에서 파생되므로, 9x9 및 15x15 runs은 다른 uniform baselines을 사용합니다.

## 출력

Configured checkpoint 디렉토리 아래에 training을 씁니다:

- `latest.pt`: 현재 candidate model.
- `best.pt`: 최고 promoted model.
- `iter_XXXX.pt`: enabled일 때 iteration snapshots.
- `trainer_state.json`: iteration counters 및 run metadata.
- `replay.pkl`: replay buffer.
- `metrics.jsonl`: iteration당 하나의 JSON metrics record.
- `runtime_progress.json`: 최신 progress snapshot.

Legacy `.safetensors` checkpoints는 여전히 model-weight-only seed input로 사용될 수 있습니다. 이들은 optimizer state를 restore하지 않습니다. 새로운 training checkpoints는 PyTorch `.pt`만 사용합니다.
