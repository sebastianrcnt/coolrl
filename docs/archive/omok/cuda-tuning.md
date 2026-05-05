> 역사적 참고: 이 문서는 초기 tinygrad-to-PyTorch transition에서의 CUDA tuning measurements를 기록합니다. 현재 Omok runtime은 PyTorch-only입니다: training checkpoints는 `.pt`, 일반 evaluator는 PyTorch, tinygrad는 더 이상 runtime dependency가 아닙니다. 아래의 tinygrad-specific 섹션을 현재 설정 지침이 아닌 역사적 baseline context로 취급하세요.

# Omok CUDA Self-Play Tuning Notes

이 note들은 C MCTS backend 추가 후 RTX 3090 CUDA profile(`configs/omok_full_cuda.yaml`)의 measurements를 캡처합니다.

## 현재 권장사항

이산 NVIDIA GPU의 CUDA self-play:

```yaml
selfplay:
  mcts_backend: c
  evaluator_backend: torch
  num_workers: 0
  batch_size: 64
  leaves_per_batch: 64
  search_threads: auto
```

CUDA profile의 경우 `num_workers: 0`을 유지합니다. Multi-process worker path는 `selfplay_worker.py`에서 tinygrad inference를 CPU에 고정하므로, `num_workers: auto`는 self-play neural network inference를 GPU에서 이동시킵니다.

Ryzen 5 5600X 테스트 머신에서 `search_threads: auto`는 12개의 logical CPUs로 resolve됩니다. 이는 C leaf-collection phase의 최대값이지, 전체 training process가 12개의 CPU cores를 계속 사용한다는 약속은 아닙니다.

`evaluator_backend: torch`는 self-play와 arena inference에만 영향을 미칩니다. Training, optimizer state, checkpoints는 여전히 tinygrad `PolicyValueNet`을 사용합니다. PyTorch evaluator는 optimizer updates 후와 best-model promotion 후 현재 tinygrad weights에서 rebuild되므로, 전체 training migration이 아닌 drop-in inference backend로 유지됩니다.

## `leaves_per_batch: 64`인 이유

C MCTS를 사용하면 tree traversal이 기존 Python tree walk보다 빠릅니다. 이전 `leaves_per_batch: 8`은 최대 self-play inference batch를 다음과 같이 생성했습니다:

```text
64 active games * 8 leaves = 512 positions
```

이를 16으로 높이면 CUDA가 다음을 볼 수 있습니다:

```text
64 active games * 16 leaves = 1024 positions
```

이를 더 64로 높이면 evaluator calls 수를 줄입니다:

```text
simulations=256, leaves_per_batch=16 -> search_batch당 16 eval rounds
simulations=256, leaves_per_batch=64 ->  search_batch당 4 eval rounds
```

현재 trainer의 mixed self-play에서 각 source는 보통 global batch의 절반을 소유하므로, 관찰된 가장 큰 초기 bucket은 일반적으로:

```text
32 active games * 64 leaves = 2048 positions
```

전체 CUDA profile에서 관찰된 self-play 타이밍:

| Setting | Iteration | Self-play time | Notes |
|---|---:|---:|---|
| `leaves_per_batch: 8` | 2 | ~44s | no self-play JIT repeats |
| `leaves_per_batch: 8` | 3 | ~53s | longer games |
| `leaves_per_batch: 16` | 2 | ~31s | max bucket 1024 |
| `leaves_per_batch: 16` | 3 | ~37s | longer games |
| `leaves_per_batch: 64` | 2 | ~20s | max bucket 2048, fresh 4-iteration run |
| `leaves_per_batch: 64` | 3 | ~47s | simulations=160, mixed source |
| `leaves_per_batch: 64` | 4 | ~47s | simulations=160, mixed source |

8에서 16으로의 이동은 초기 runs에서 self-play 시간을 대략 30-40% 줄였습니다. 16에서 64로의 이동도 throughput을 개선했지만, 전체 iteration time은 이제 self-play, training, arena에 걸쳐 분할되므로, 추가적인 self-play-only tuning은 whole-iteration time에 제한적인 영향을 미칩니다.

`leaves_per_batch: 64`를 ROCm 또는 Metal profiles에 맹목적으로 일반화하지 마세요. Backend kernel behavior, JIT behavior, search-quality tradeoffs는 다릅니다. Non-CUDA profiles의 경우 8/16/32/64를 sweep하고 duration 및 arena quality metrics을 비교하세요.

## Optional TensorRT Evaluator

TensorRT evaluator는 self-play와 arena MCTS 내 neural network inference을 가속화하기만 합니다. 이는 PyTorch training, optimizer updates, replay sampling, MCTS tree traversal을 대체하지 않습니다.

NVIDIA CUDA systems에서 명시적으로 활성화하려면:

```yaml
selfplay:
  evaluator_backend: tensorrt
```

또는 TensorRT가 설치되었을 때 CUDA-only auto-selection을 허용하려면:

```yaml
selfplay:
  evaluator_backend: auto
```

`auto`는 CUDA 또는 TensorRT를 사용할 수 없을 때 torch evaluator로 fallback합니다. Apple Silicon과 Metal/MPS runs는 torch evaluator를 계속 사용해야 합니다. TensorRT는 Metal backend가 아니고 non-CUDA paths에서 imported되지 않습니다.

다음으로 optional dependencies를 설치하세요:

```bash
uv sync --extra omok-tensorrt
```

NVIDIA의 pip package는 TensorRT가 지원하는 최신 CUDA major variant로 기본값을 설정합니다. 머신에 특정 CUDA major version이 필요한 경우 matching NVIDIA package를 수동으로 설치하세요. 예를 들어 `tensorrt-cu12` 또는 `tensorrt-cu13`.

유용한 environment knobs:

| Variable | Default | Meaning |
|---|---:|---|
| `COOLRL_TENSORRT_MAX_BATCH` | `4096` | maximum dynamic profile batch |
| `COOLRL_TENSORRT_OPT_BATCH` | `384` | optimization profile batch |
| `COOLRL_TENSORRT_FP16` | `1` | enable FP16 tactics when the GPU supports them |
| `COOLRL_TENSORRT_WORKSPACE_MB` | `2048` | TensorRT builder workspace limit |
| `COOLRL_TENSORRT_CACHE` | `~/.cache/coolrl/tensorrt` | engine cache directory; set `0` for a temp cache |

Candidate-model engines는 비용이 많이 들 수 있습니다. candidate weights가 모든 optimizer phase 후에 변경되기 때문입니다. Best-model engines는 best model이 promotion 시에만 변경되므로 더 잘 분산됩니다.

## `num_workers: auto`인 아닌 이유

짧은 benchmark configs는 동일한 network size, 16 games, 32 simulations을 사용하여 CUDA single-process self-play를 ProcessPool workers와 비교했습니다:

| Path | `num_workers` | Inference device | Duration | Samples |
|---|---:|---|---:|---:|
| CUDA single process | `0` | CUDA | 14.924s | 787 |
| ProcessPool workers | `auto` | CPU | 44.060s | 690 |

worker path는 전체적으로 약 3배 더 느리고 replay sample당 약 3.4배 더 느렸습니다. 더 많은 CPU cores를 사용하지만, 각 worker는 tinygrad CPU inference를 수행합니다. 이는 3090에서 batching inference보다 느립니다.

shared GPU contexts를 피하는 것이 목표인 CPU/Metal-style profiles에 worker parallelism을 사용하세요. CUDA profile의 경우 self-play를 main process에 유지하세요.

## Threaded C MCTS

`search_threads`는 C backend 내 tree-level parallelism을 제어합니다. 작업은 active game/tree로 분할되므로, collection round 중에 각 `MctsTree`는 여전히 하나의 thread에서만 mutated됩니다. 이는 두 경로 모두 동일한 `MCTS.search_batch(...)` implementation을 호출하기 때문에 self-play와 arena에 적용됩니다.

이는 의도적으로 same-tree parallel MCTS가 아닙니다. virtual loss, atomics, locks를 tree 내에서 사용하지 않습니다. Utilization은 batch에서 많은 games이 active할 때 최고입니다.

tree-level threading과 C nodes에 대한 arena allocation을 추가한 후, CPU tree traversal은 더 이상 관찰되는 CUDA-profile bottleneck이 아닙니다. `checkpoints/omok_full_cuda`에서 resumed short run:

```bash
COOLRL_MCTS_TIMING=1 COOLRL_MCTS_TIMING_MAX_CALLS=40 \
uv run python -m coolrl.omok.train \
  --config configs/omok_full_cuda.yaml \
  --resume checkpoints/omok_full_cuda \
  --max-iterations 5
```

representative `cmcts_wrapper.MCTS.search_batch(...)` timings을 보였습니다:

| Segment | Example Time | Meaning | Current Status |
|---|---:|---|---|
| `root` / `noise` | ~0.0004s | root prep and Dirichlet noise | not a bottleneck |
| `collect` | ~0.006s | C MCTS leaf collection, using 12 threads | not a bottleneck |
| `eval` | ~1.15s | `ModelEvaluator.evaluate_features(...)` on CUDA | dominant bottleneck |
| `feed` | ~0.009s | C expand/backprop after NN eval | not a bottleneck |
| `extract` / `sample` | <0.001s | policy extraction and action sampling | not a bottleneck |

일반적인 line의 경우:

```text
states=32 sims=256 leaves_per_batch=16 threads=12 rounds=16 leaves=8192
collect=0.006s eval=1.15s feed=0.009s total=1.17s
```

`collect`는 측정된 `search_batch` 시간의 대략 0.5%인 반면, `eval`은 98% 이상입니다. 이것은 `search_threads: auto`에서도 CPU utilization이 한두 개의 busy cores 주위에 머물 수 있는 이유를 설명합니다. threaded C section은 매우 짧고, 대부분의 wall time은 CUDA/tinygrad evaluation에 소비됩니다.

## Iteration-Level Bottleneck Map

| Phase | Code Path | Main Resource | Current Bottleneck? |
|---|---|---|---|
| startup/resume | `Trainer.__init__`, `_restore_from_checkpoint` | disk/CPU | no |
| self-play C leaf collection | `cmcts_wrapper` -> C `mcts_batch_collect_leaves_threaded` | CPU threads | no |
| self-play NN evaluation | `ModelEvaluator.evaluate_features` | CUDA/tinygrad | yes |
| self-play feed/backup | C `mcts_batch_feed_leaves` | CPU | no |
| replay insert | `ReplayBuffer.add_game` | CPU/RAM | no |
| optimizer updates | `Trainer.train_model` | CUDA/tinygrad | not measured here; possible secondary bottleneck |
| arena MCTS | `Arena._advance_games` -> backend `search_batch` | CPU + CUDA | likely same eval bottleneck |
| checkpoint save | `save_model_checkpoints`, `save_runtime_state` | disk | no |

이러한 measurements를 고려하면, persistent C worker pool은 CUDA profile에 대해 낮은 ROI를 가질 것입니다. 모든 C collection overhead를 제거해도 측정된 `search_batch` 시간의 약 0.5%만 절약할 수 있습니다. 다음 tuning target은 `ModelEvaluator.evaluate_features`, tinygrad CUDA bucket behavior, evaluator calls의 수/shape이어야 합니다.

## Evaluator Microbenchmark

tinygrad evaluator latency를 격리하려면 `scripts/bench_omok_evaluator.py`를 사용하세요:

```bash
uv run python scripts/bench_omok_evaluator.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --backend tinygrad \
  --batches 128,256,512,1024,2048 \
  --warmup 2 \
  --iters 5
```

RTX 3090 profile에서의 representative results:

| Batch | Total Avg | `priors_numpy` Avg | Notes |
|---:|---:|---:|---|
| 128 | ~0.059s | ~0.017s | small late-game bucket |
| 256 | ~0.064s | ~0.022s | small/mid bucket |
| 512 | ~0.071s | ~0.030s | `leaves_per_batch=16`과 common |
| 1024 | ~0.101s | ~0.047s | mid bucket |
| 2048 | ~0.115s | ~0.073s | `leaves_per_batch=64`과 common early bucket |

tinygrad는 lazy이므로, benchmark의 `forward_lazy`는 mostly graph construction이고 full GPU completion이 아닙니다. main synchronization point는 `priors_numpy`에 나타나며, 여기서 policy logits이 materialized되고 C MCTS를 위해 다시 copied됩니다. 이는 더 큰 batches가 도움이 될 수 있는 이유를 설명합니다: batch 2048은 batch 512보다만 약 1.6배 느린 반면, `leaves_per_batch=64`은 256 simulations에서 16 상대로 eval rounds를 4배 줄입니다.

## PyTorch Evaluator Microbenchmark

동일한 Omok network shape (`channels=64`, `blocks=6`)을 사용한 one-off PyTorch eager benchmark를 `uv run --with torch`를 통해 실행했습니다. PyTorch를 project dependency로 추가하지 않았습니다. Environment:

```text
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.11.0+cu130
CUDA: available
```

benchmark는 equivalent PyTorch module을 구성했고 explicit synchronization으로 CUDA forward, softmax, numpy materialization을 측정했습니다. Weights는 random이었습니다. 이는 saved checkpoint에 대한 parity test가 아닌 latency comparison이었습니다.

integrated benchmark script는 이제 torch evaluator를 직접 exercise할 수 있습니다:

```bash
uv run --extra omok python scripts/bench_omok_evaluator.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --backend torch \
  --batches 128,256,512,1024,2048 \
  --warmup 5 \
  --iters 20
```

| Batch | tinygrad Total Avg | PyTorch Eager Total Avg | Speedup |
|---:|---:|---:|---:|
| 128 | ~0.0592s | ~0.0017s | ~35x |
| 256 | ~0.0631s | ~0.0019s | ~33x |
| 512 | ~0.0711s | ~0.0035s | ~20x |
| 1024 | ~0.0998s | ~0.0060s | ~17x |
| 2048 | ~0.1162s | ~0.0116s | ~10x |

이는 초기 2-4x rough estimate보다 훨씬 더 큰 gap입니다. 이전 C MCTS timing이 `ModelEvaluator.evaluate_features(...)`가 `search_batch(...)`의 98% 이상이었기 때문에, evaluator만 대체하면 training에 상대적으로 self-play 및 arena eval costs가 거의 사라질 수 있습니다.

업데이트된 rough iteration estimates:

| Scenario | Self-play | Training | Arena | Iteration | 200 Iterations |
|---|---:|---:|---:|---:|---:|
| current tinygrad | ~47s | ~48s | ~48s | ~135-143s | ~7.5-8.0h |
| PyTorch evaluator only | ~3-5s | ~48s | ~3-5s | ~55-60s | ~3.0-3.4h |

정확한 numbers는 여전히 full-loop validation이 필요합니다. microbenchmark는 checkpoint weight conversion, C MCTS feed/collect overhead after eval is reduced, Python loop overhead, game length의 변화를 포함하지 않습니다. 그러나 충분히 강력해서 PyTorch evaluator backend를 다음 highest-ROI change로 만듭니다.

integrated PyTorch evaluator는 여전히 power-of-two batch buckets로 pads합니다. unpadded smoke run은 많은 unique CUDA shapes를 생성했고 iteration 1에서 self-play eval에 `13.3s`를 소비했습니다. Power-of-two padding은 shape set을 collapsed하고 같은 run을 `3.29s` eval time으로 줄였습니다.

PyTorch eval이 integrated되면, `leaves_per_batch`를 다시 sweep해야 합니다. 현재 `64` value는 주로 expensive tinygrad evaluator calls를 줄이기 위해 선택되었습니다. PyTorch eval이 calls를 cheap하게 만든다면, `8`, `16`, `32`와 같은 더 낮은 values는 wall time을 많이 소비하지 않으면서 neural evaluations를 더 자주 feeding/backing up하여 search quality를 recover할 수 있습니다.

Implementation status: CUDA profile은 이제 `selfplay.evaluator_backend: torch`를 사용합니다. tinygrad evaluator에 대한 CPU 및 CUDA parity checks는 normal floating-point tolerance 내에 있었습니다:

| Device | Policy Max Abs Diff | Value Max Abs Diff |
|---|---:|---:|
| CPU, batch 7 | ~2.8e-9 | ~1.0e-7 |
| CUDA, batch 64 | ~4.0e-7 | ~1.1e-5 |

torch evaluator로 완료된 8 games 및 8 simulations를 가진 small C MCTS + CUDA smoke run:

```text
selfplay_seconds=1.184s
eval_selfplay_candidate_seconds=1.141s
eval_selfplay_candidate_calls=68
eval_selfplay_candidate_avg_seconds=0.0168s
```

64 games 및 96 simulations를 사용하지만 training/arena가 없는 full CUDA iteration-1 warmup run은 이전 tinygrad baseline of `25.42s` self-play time에서 다음으로 dropped:

```text
selfplay_seconds=3.591s
eval_selfplay_candidate_seconds=3.290s
eval_selfplay_candidate_calls=158
eval_selfplay_candidate_avg_seconds=0.0208s
eval_selfplay_candidate_max_bucket=4096
eval_selfplay_candidate_pad_ratio=1.1685
```

CUDA profile을 실행하기 전에 `uv run --extra omok ...`을 사용하거나 PyTorch를 설치하세요. PyTorch가 설치되지 않으면, `evaluator_backend: torch`는 installation hint로 빠르게 실패합니다.

## 4-Iteration Phase Benchmark

`metrics.jsonl`에 phase timing fields를 추가한 후, empty `checkpoints/omok_full_cuda`에서 `leaves_per_batch: 64` 및 `search_threads: auto`를 사용한 fresh run을 produced:

```bash
rm -rf checkpoints/omok_full_cuda
uv run python -m coolrl.omok.train \
  --config configs/omok_full_cuda.yaml \
  --max-iterations 4
```

| Iter | Sims | Total | Self-play | Train | Arena | Checkpoint | Avg Moves | Arena WR | Accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 96 | 25.42s | 25.42s | 0.00s | 0.00s | 0.84s | 56.23 | - | - |
| 2 | 96 | 118.47s | 20.47s | 57.95s | 39.56s | 1.03s | 53.81 | 0.5000 | true |
| 3 | 160 | 143.83s | 47.14s | 42.59s | 54.09s | 1.03s | 54.36 | 0.4167 | false |
| 4 | 160 | 141.25s | 47.04s | 43.44s | 50.77s | 1.03s | 54.41 | 0.4167 | false |

trained iterations 2-4의 경우, average phase times는:

| Phase | Average |
|---|---:|
| total | 134.52s |
| self-play | 38.22s |
| training | 47.99s |
| arena | 48.14s |
| checkpoint | 1.03s |

현재 CUDA profile은 따라서 더 이상 하나의 CPU MCTS section에 의해 dominated되지 않습니다. 160 simulations에서 self-play, training updates, arena는 모두 material contributors입니다. 미래의 최적화는 self-play MCTS traversal에만 focusing하는 것을 피해야 합니다.

## Built-In Training Metrics

trainer는 이제 별도의 microbenchmark를 먼저 실행하지 않고 common Omok CUDA bottlenecks를 진단하기 위해 충분한 per-iteration fields를 작성합니다.

Top-level phase timings:

| Field | Meaning |
|---|---|
| `duration_seconds` | checkpoint save를 포함한 full iteration wall time |
| `selfplay_seconds` | full self-play phase wall time |
| `train_seconds` | optimizer update phase wall time |
| `arena_seconds` | candidate-vs-best arena wall time |
| `checkpoint_seconds` | checkpoint 및 runtime state save time |

MCTS search timings은 phase로 grouped됩니다. 예를 들어 `search_selfplay_candidate_*`, `search_selfplay_best_*`, `search_arena_candidate_*`, `search_arena_best_*`:

| Suffix | Meaning |
|---|---|
| `_calls` | `MCTS.search_batch(...)` calls의 number |
| `_seconds` | evaluator calls를 포함한 total search wall time |
| `_avg_seconds` | average search call latency |
| `_states` / `_avg_states` | search calls로 본 active game states |
| `_requested_leaves` | requested MCTS leaf visits, roughly states * simulations |
| `_max_states` | that phase로 본 largest active batch |
| `_max_simulations` | that phase로 사용한 largest simulation count |
| `_max_leaves_per_batch` | that phase로 사용한 largest configured leaf batch |

Evaluator timings은 `eval_*` fields로 같은 phase names로 grouped됩니다:

| Suffix | Meaning |
|---|---|
| `_calls` | neural evaluator calls의 number |
| `_seconds` | total evaluator wall time |
| `_avg_seconds` | average evaluator call latency |
| `_positions` | unpadded board positions evaluated |
| `_padded_positions` | power-of-two tinygrad bucket padding 후 positions |
| `_pad_ratio` | padded_positions / positions |
| `_avg_batch` / `_max_batch` | actual evaluator batch sizes |
| `_max_bucket` | largest padded tinygrad bucket |
| `_bucket_counts` | padded bucket size당 calls count |

Training update timings:

| Field | Meaning |
|---|---|
| `train_metric_updates` | measured optimizer updates |
| `train_metric_samples` | total sampled replay rows |
| `train_sample_seconds` | replay sample 및 tensor creation time |
| `train_forward_seconds` | model forward graph construction time |
| `train_loss_seconds` | policy/value loss graph construction time |
| `train_backward_seconds` | backward graph construction time |
| `train_optimizer_seconds` | optimizer step graph construction/execution time |
| `train_sync_seconds` | loss materialization 및 synchronization time |
| `train_measured_seconds` | measured training substeps의 sum |

evaluator latency를 synthetic batch size로 isolate하거나 tinygrad를 다른 inference backend와 비교할 때만 separate `scripts/bench_omok_evaluator.py`를 사용하세요. Normal training runs은 먼저 `metrics.jsonl`에 rely해야 합니다.

## Current Handoff Notes

- tree-level threading 및 node arena allocation 후 C MCTS traversal은 빠릅니다.
- `search_threads: auto`는 ceiling로서 유용하지만, C collection은 wall time의 작은 부분이기 때문에 높은 CPU utilization을 기대할 수 없습니다.
- `leaves_per_batch: 64`는 tinygrad에서 evaluator calls를 줄여 초기 CUDA throughput을 개선했습니다. PyTorch eval이 integrated된 후 `8/16/32/64`를 re-sweep하세요. `64`는 더 이상 best quality/speed tradeoff가 아닐 수 있습니다.
- 가장 promising한 다음 experiments는:
  - self-play 및 arena를 위한 PyTorch evaluator backend를 추가합니다.
  - training runs를 위해 사용하기 전에 tinygrad-to-PyTorch checkpoint weight parity를 verify합니다.
  - PyTorch eval 및 `leaves_per_batch: 64`로 full-loop CUDA benchmark를 실행합니다.
  - PyTorch eval 후 `leaves_per_batch`를 sweep합니다.
  - eval이 더 이상 dominant하지 않을 때 training updates이 tinygrad-bound인지 measure합니다.
  - training이 expected to become the dominant remaining bottleneck이므로 PyTorch eval lands 후에만 ONNX Runtime 또는 TensorRT를 고려합니다.
  - `collect`가 다시 `search_batch` 시간의 meaningful share가 될 경우에만 persistent C worker pools을 revisit합니다.

## Currently Unused Reference Fields

이 fields는 reference configs와의 호환성을 위해 parsed되지만, 현재 C backend는 async inference queues를 구현하지 않습니다:

```yaml
selfplay:
  inference_batch_size: 512
  inference_wait_ms: 2.0
```

Useful future directions:

- `inference_batch_size`: evaluator를 호출하기 전에 여러 C leaf batches를 collecting하기 위한 target/cap이 될 수 있습니다.
- `inference_wait_ms`: async inference server/queue에서만 makes sense합니다.
