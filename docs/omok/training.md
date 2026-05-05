# Omok 학습 구조

이 문서는 Omok training run을 키우거나 config를 조정하려는 사람을 위한 설명입니다.

## 전체 흐름

Omok training loop는 대략 다음 순서로 움직입니다.

1. 현재 candidate/best model을 준비합니다.
2. MCTS self-play로 replay samples를 생성합니다.
3. replay buffer에서 batch를 뽑아 policy/value network를 업데이트합니다.
4. arena에서 candidate와 best를 비교합니다.
5. 기준을 넘으면 candidate를 best로 promote합니다.
6. checkpoint와 metrics를 저장합니다.

## 기본 config

9x9 presets:

- `configs/omok_smoke.yaml`: pipeline 확인용.
- `configs/omok_quick.yaml`: 짧은 로컬 run.
- `configs/omok_full_cuda.yaml`: NVIDIA/discrete GPU profile.
- `configs/omok_full_metal.yaml`: Apple Silicon profile.

15x15 presets:

- `configs/omok15_smoke.yaml`
- `configs/omok15_quick.yaml`
- `configs/omok15_full_cuda.yaml`
- `configs/omok15_full_cuda_hdd.yaml`

보존 중인 `omok15_full_cuda_hdd` checkpoint의 위치와 run 요약은 [`omok15-full-cuda-hdd-run.md`](omok15-full-cuda-hdd-run.md)에 기록합니다.

Smoke/quick에서 시작한 뒤 `max_iterations`, `games_per_iteration`, `simulations`, `updates_per_iteration`, `arena.games`를 키우는 방식이 안전합니다.

## Self-play knobs

Self-play throughput은 주로 아래 값들로 정해집니다.

```yaml
selfplay:
  games_per_iteration: 16
  batch_size: 4
  leaves_per_batch: 8
  simulation_schedule:
    - fraction: 0.0
      simulations: 16
```

- `games_per_iteration`: iteration마다 생성할 self-play games 수.
- `batch_size`: 한 번의 `MCTS.search_batch(...)`에 같이 넣는 active games 수.
- `leaves_per_batch`: evaluator call 전에 각 game이 제공하는 MCTS leaves 수.
- `simulation_schedule`: 학습 진행도에 따라 move당 MCTS simulations를 조정.

대략적인 evaluator batch 크기는 다음과 같습니다.

```text
active games * leaves_per_batch
```

CUDA profile에서는 큰 evaluator batch가 유리할 수 있습니다. CPU/Metal에서는 worker 수와 batch 크기를 함께 조정해야 합니다.

## Multi-process self-play

```yaml
selfplay:
  num_workers: auto
```

값의 의미:

- `0`: single-process self-play. CUDA full runs, debugging, deterministic profiling에 적합.
- `auto`: `os.cpu_count()` 기반으로 worker 수를 정함.
- 양수 integer: 고정 worker 수.

CUDA full profile에서는 보통 `num_workers: 0`을 유지합니다. GPU inference를 main process에 두고 큰 batch로 처리하는 편이 유리하기 때문입니다.

Apple Silicon/CPU-style profile에서는 CPU workers가 self-play를 병렬로 진행하고 main process가 training device를 유지하는 구조가 유용할 수 있습니다.

## Optimization knobs

```yaml
optimization:
  batch_size: 64
  updates_per_iteration: 16
```

- `batch_size`: optimizer step당 replay samples 수.
- `updates_per_iteration`: iteration마다 수행할 optimizer updates 수.

더 큰 batch는 accelerator 활용을 높일 수 있지만 메모리 사용량도 늘립니다.

## Arena

Arena는 candidate와 best model을 비교합니다.

```yaml
arena:
  games: 4
  simulations: 16
```

Tuning 중에는 arena games를 작게 유지하는 편이 좋습니다. Arena도 MCTS-heavy phase라서 전체 iteration time을 크게 늘릴 수 있습니다.

## Checkpoint 출력

Configured checkpoint 디렉토리 아래에 다음 파일들이 저장됩니다.

- `latest.pt`: 현재 candidate model.
- `best.pt`: 최고 promoted model.
- `iter_XXXX.pt`: enabled일 때 iteration snapshots.
- `trainer_state.json`: iteration counters와 run metadata.
- `replay.pkl`: replay buffer.
- `metrics.jsonl`: iteration당 하나의 JSON metrics record.
- `runtime_progress.json`: 최신 progress snapshot.

Legacy `.safetensors` checkpoints는 model-weight-only seed input으로 사용할 수 있지만 optimizer state는 복원하지 않습니다. 새로운 training checkpoints는 PyTorch `.pt`를 사용합니다.

## Metrics 보기

완료되었거나 진행 중인 run에서 training metrics를 plot합니다.

```bash
omok-plot checkpoints/omok_full_cuda
```

Interactive window를 열려면:

```bash
omok-plot checkpoints/omok_full_cuda --show
```

직접 `metrics.jsonl` 파일을 가리킬 수도 있습니다.

```bash
omok-plot checkpoints/omok_full_cuda/metrics.jsonl
```

Report는 train loss, policy/value loss, arena win rate, self-play average moves, replay buffer size, elapsed hours를 보여줍니다. Policy entropy reference line은 기록된 `board_size`에서 파생됩니다.

## 추천 시작점

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
  batch_size: 8
  leaves_per_batch: 4
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

더 무거운 run에서는 먼저 `games_per_iteration`, `simulations`, `updates_per_iteration`를 조금씩 키우세요. 머신이 느려지면 `simulations`를 먼저 줄이고, 메모리 압력이 생기면 `optimization.batch_size` 또는 network `channels`를 줄입니다.
