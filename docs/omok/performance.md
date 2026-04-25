# Omok 성능 설정

이 문서는 Omok training을 어느 backend와 config로 돌릴지 고르는 데 필요한 현재 기준의 요약입니다.

과거 CUDA tuning log와 benchmark 세부 기록은 [`../omok_cuda_tuning.md`](../omok_cuda_tuning.md)에 남겨두었습니다. 이 문서는 먼저 실사용 결정을 돕는 것을 목표로 합니다.

## 한 줄 선택지

| 환경 | 추천 시작점 |
|---|---|
| CPU에서 기능 확인 | `configs/omok_smoke.yaml`, `configs/omok_quick.yaml` |
| Apple Silicon | `configs/omok_full_metal.yaml` |
| NVIDIA CUDA GPU | `configs/omok_full_cuda.yaml` |
| 15x15 CUDA | `configs/omok15_full_cuda.yaml` |
| GUI/TUI inference | ONNX export 후 `play.md` 참고 |

## CUDA profile

NVIDIA/discrete GPU에서는 full CUDA preset으로 시작합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok_full_cuda.yaml
```

CUDA self-play의 기본 방향은 GPU에 충분히 큰 inference batch를 공급하는 것입니다.

```yaml
selfplay:
  mcts_backend: c
  evaluator_backend: torch
  num_workers: 0
  batch_size: 64
  leaves_per_batch: 64
  search_threads: auto
```

CUDA에서는 보통 `num_workers: 0`을 유지합니다. Multi-process worker path는 GPU batching 이점을 줄일 수 있습니다.

PyTorch evaluator를 쓰려면 PyTorch가 설치되어 있어야 합니다. 필요한 extra와 설치 환경은 project dependency 설정을 확인하세요.

## Apple Silicon / Metal profile

Apple Silicon에서는 Metal/MPS training과 CPU worker self-play를 조합하는 profile로 시작합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok_full_metal.yaml
```

Metal/CPU-style runs에서는 `selfplay.num_workers`를 늘려 여러 CPU workers가 self-play를 만들고, main process가 training device를 유지하는 식의 trade-off가 유용할 수 있습니다.

## CPU / smoke / debugging

디버깅이나 pipeline 확인에는 CPU smoke run을 사용하세요.

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device CPU
```

Deterministic profiling이나 작은 재현에는 `selfplay.num_workers: 0`이 다루기 쉽습니다.

## TensorRT evaluator

TensorRT evaluator는 self-play와 arena MCTS 안의 neural network inference를 가속화하기 위한 선택지입니다. PyTorch training, optimizer updates, replay sampling, MCTS tree traversal 자체를 대체하지 않습니다.

명시적으로 켜려면:

```yaml
selfplay:
  evaluator_backend: tensorrt
```

또는 CUDA에서만 자동 선택을 허용하려면:

```yaml
selfplay:
  evaluator_backend: auto
```

설치:

```bash
uv sync --extra omok-tensorrt
```

TensorRT는 NVIDIA CUDA systems용입니다. Apple Silicon과 Metal/MPS runs에서는 torch evaluator를 사용하세요.

## 가장 먼저 조정할 knobs

Throughput을 늘리고 싶을 때:

- `selfplay.games_per_iteration`: iteration당 더 많은 self-play data.
- `selfplay.batch_size`: 동시에 search하는 active games 수.
- `selfplay.leaves_per_batch`: evaluator call당 모으는 leaves 수.
- `optimization.batch_size`: optimizer step당 sample 수.
- `optimization.updates_per_iteration`: iteration당 학습 update 수.

Search quality를 키우고 싶을 때:

- `selfplay.simulation_schedule[].simulations`
- `arena.simulations`

Run이 너무 느릴 때는 `simulations`를 먼저 줄이는 편이 효과가 큽니다. 메모리가 부족하면 `optimization.batch_size`, network `channels`, replay capacity를 줄입니다.

## Metrics로 병목 보기

Normal training run은 `metrics.jsonl`에 phase timing을 씁니다.

주요 필드:

| Field | Meaning |
|---|---|
| `duration_seconds` | checkpoint save를 포함한 full iteration wall time |
| `selfplay_seconds` | self-play phase wall time |
| `train_seconds` | optimizer update phase wall time |
| `arena_seconds` | arena phase wall time |
| `checkpoint_seconds` | checkpoint save time |

Synthetic evaluator latency만 따로 보고 싶을 때는 benchmark script를 사용합니다.

```bash
uv run --extra omok python scripts/bench_omok_evaluator.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --backend torch \
  --batches 128,256,512,1024,2048 \
  --warmup 5 \
  --iters 20
```

일반적인 tuning은 먼저 `metrics.jsonl`의 phase timings를 보고 판단하세요. Microbenchmark는 evaluator 자체를 분리해서 보고 싶을 때만 사용합니다.

## 더 깊은 기록

- CUDA tuning, historical benchmark, TensorRT knobs: [`../omok_cuda_tuning.md`](../omok_cuda_tuning.md)
- 15x15 MCTS memory incident: [`../omok-mcts-memory.md`](../omok-mcts-memory.md)
