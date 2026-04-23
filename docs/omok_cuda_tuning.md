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

On the Ryzen 5 5600X test machine, `search_threads: auto` resolves to 12
logical CPUs. This is a maximum for the C leaf-collection phase, not a promise
that the whole training process will keep 12 CPU cores busy.

`evaluator_backend: torch` affects only self-play and arena inference. Training,
optimizer state, and checkpoints still use the tinygrad `PolicyValueNet`. The
PyTorch evaluator is rebuilt from the current tinygrad weights after optimizer
updates and after best-model promotion, so it remains a drop-in inference
backend rather than a full training migration.

## Why `leaves_per_batch: 64`

With C MCTS, tree traversal is faster than the old Python tree walk. The
previous `leaves_per_batch: 8` produced a maximum self-play inference batch of:

```text
64 active games * 8 leaves = 512 positions
```

Raising it to 16 lets CUDA see:

```text
64 active games * 16 leaves = 1024 positions
```

Raising it further to 64 reduces the number of evaluator calls:

```text
simulations=256, leaves_per_batch=16 -> 16 eval rounds per search_batch
simulations=256, leaves_per_batch=64 ->  4 eval rounds per search_batch
```

With mixed self-play in the current trainer, each source usually owns half the
global batch, so the largest observed early bucket is typically:

```text
32 active games * 64 leaves = 2048 positions
```

Observed self-play timings on the full CUDA profile:

| Setting | Iteration | Self-play time | Notes |
|---|---:|---:|---|
| `leaves_per_batch: 8` | 2 | ~44s | no self-play JIT repeats |
| `leaves_per_batch: 8` | 3 | ~53s | longer games |
| `leaves_per_batch: 16` | 2 | ~31s | max bucket 1024 |
| `leaves_per_batch: 16` | 3 | ~37s | longer games |
| `leaves_per_batch: 64` | 2 | ~20s | max bucket 2048, fresh 4-iteration run |
| `leaves_per_batch: 64` | 3 | ~47s | simulations=160, mixed source |
| `leaves_per_batch: 64` | 4 | ~47s | simulations=160, mixed source |

The move from 8 to 16 reduced self-play time by roughly 30-40% in the earlier
runs. The move from 16 to 64 also improved throughput, but total iteration time
is now split across self-play, training, and arena, so further self-play-only
tuning has limited impact on whole-iteration time.

Do not blindly generalize `leaves_per_batch: 64` to ROCm or Metal profiles.
Backend kernel behavior, JIT behavior, and search-quality tradeoffs differ.
For non-CUDA profiles, sweep 8/16/32/64 and compare duration plus arena
quality metrics.

## Optional TensorRT Evaluator

The TensorRT evaluator accelerates only neural network inference inside
self-play and arena MCTS. It does not replace PyTorch training, optimizer
updates, replay sampling, or MCTS tree traversal.

Enable it explicitly on NVIDIA CUDA systems with:

```yaml
selfplay:
  evaluator_backend: tensorrt
```

Or allow CUDA-only auto-selection when TensorRT is installed:

```yaml
selfplay:
  evaluator_backend: auto
```

`auto` falls back to the torch evaluator when CUDA or TensorRT is unavailable.
Apple Silicon and Metal/MPS runs should keep using the torch evaluator; TensorRT
is not a Metal backend and is not imported on non-CUDA paths.

Install the optional dependencies with:

```bash
uv sync --extra omok-tensorrt
```

NVIDIA's pip package defaults to the latest CUDA major variant supported by
TensorRT. If a machine needs a specific CUDA major version, install the matching
NVIDIA package manually, for example `tensorrt-cu12` or `tensorrt-cu13`.

Useful environment knobs:

| Variable | Default | Meaning |
|---|---:|---|
| `COOLRL_TENSORRT_MAX_BATCH` | `4096` | maximum dynamic profile batch |
| `COOLRL_TENSORRT_OPT_BATCH` | `384` | optimization profile batch |
| `COOLRL_TENSORRT_FP16` | `1` | enable FP16 tactics when the GPU supports them |
| `COOLRL_TENSORRT_WORKSPACE_MB` | `2048` | TensorRT builder workspace limit |
| `COOLRL_TENSORRT_CACHE` | `~/.cache/coolrl/tensorrt` | engine cache directory; set `0` for a temp cache |

Candidate-model engines can be expensive because the candidate weights change
after every optimizer phase. Best-model engines amortize better because the best
model changes only on promotion.

## Why Not `num_workers: auto`

Short benchmark configs compared CUDA single-process self-play against
ProcessPool workers using the same network size, 16 games, and 32 simulations:

| Path | `num_workers` | Inference device | Duration | Samples |
|---|---:|---|---:|---:|
| CUDA single process | `0` | CUDA | 14.924s | 787 |
| ProcessPool workers | `auto` | CPU | 44.060s | 690 |

The worker path was about 3x slower overall and about 3.4x slower per replay
sample. It uses more CPU cores, but each worker does tinygrad CPU inference,
which is slower than batching inference on the 3090.

Use worker parallelism for CPU/Metal-style profiles where avoiding shared GPU
contexts is the goal. For the CUDA profile, keep self-play in the main process.

## Threaded C MCTS

`search_threads` controls tree-level parallelism inside the C backend. The work
is split by active game/tree, so each `MctsTree` is still mutated by only one
thread during a collection round. This applies to self-play and arena because
both paths call the same `MCTS.search_batch(...)` implementation.

This is intentionally not same-tree parallel MCTS: it does not use virtual
loss, atomics, or locks inside one tree. Utilization is best while many games
are active in the batch.

After adding tree-level threading and arena allocation for C nodes, CPU tree
traversal is no longer the observed CUDA-profile bottleneck. A short resumed
run from `checkpoints/omok_full_cuda` with:

```bash
COOLRL_MCTS_TIMING=1 COOLRL_MCTS_TIMING_MAX_CALLS=40 \
uv run python -m coolrl.omok.train \
  --config configs/omok_full_cuda.yaml \
  --resume checkpoints/omok_full_cuda \
  --max-iterations 5
```

showed representative `cmcts_wrapper.MCTS.search_batch(...)` timings:

| Segment | Example Time | Meaning | Current Status |
|---|---:|---|---|
| `root` / `noise` | ~0.0004s | root prep and Dirichlet noise | not a bottleneck |
| `collect` | ~0.006s | C MCTS leaf collection, using 12 threads | not a bottleneck |
| `eval` | ~1.15s | `ModelEvaluator.evaluate_features(...)` on CUDA | dominant bottleneck |
| `feed` | ~0.009s | C expand/backprop after NN eval | not a bottleneck |
| `extract` / `sample` | <0.001s | policy extraction and action sampling | not a bottleneck |

For a typical line:

```text
states=32 sims=256 leaves_per_batch=16 threads=12 rounds=16 leaves=8192
collect=0.006s eval=1.15s feed=0.009s total=1.17s
```

`collect` is roughly 0.5% of the measured `search_batch` time, while `eval` is
more than 98%. This explains why CPU utilization may appear to sit around one
or two busy cores even with `search_threads: auto`: the threaded C section is
very short, and most wall time is spent in CUDA/tinygrad evaluation.

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

Given these measurements, a persistent C worker pool would have low ROI for the
CUDA profile. Even eliminating all C collection overhead would only save around
0.5% of measured `search_batch` time. The next tuning target should be
`ModelEvaluator.evaluate_features`, tinygrad CUDA bucket behavior, and the
number/shape of evaluator calls.

## Evaluator Microbenchmark

Use `scripts/bench_omok_evaluator.py` to isolate tinygrad evaluator latency:

```bash
uv run python scripts/bench_omok_evaluator.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --backend tinygrad \
  --batches 128,256,512,1024,2048 \
  --warmup 2 \
  --iters 5
```

Representative results on the RTX 3090 profile:

| Batch | Total Avg | `priors_numpy` Avg | Notes |
|---:|---:|---:|---|
| 128 | ~0.059s | ~0.017s | small late-game bucket |
| 256 | ~0.064s | ~0.022s | small/mid bucket |
| 512 | ~0.071s | ~0.030s | common with `leaves_per_batch=16` |
| 1024 | ~0.101s | ~0.047s | mid bucket |
| 2048 | ~0.115s | ~0.073s | common early bucket with `leaves_per_batch=64` |

tinygrad is lazy, so `forward_lazy` in the benchmark is mostly graph
construction, not full GPU completion. The main synchronization point appears
in `priors_numpy`, where policy logits are materialized and copied back for C
MCTS. This explains why larger batches can help: batch 2048 is only about 1.6x
slower than batch 512, while `leaves_per_batch=64` reduces eval rounds by 4x
relative to 16 at 256 simulations.

## PyTorch Evaluator Microbenchmark

A one-off PyTorch eager benchmark using the same Omok network shape
(`channels=64`, `blocks=6`) was run through `uv run --with torch`, without
adding PyTorch as a project dependency. Environment:

```text
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.11.0+cu130
CUDA: available
```

The benchmark constructed an equivalent PyTorch module and measured CUDA
forward, softmax, and numpy materialization with explicit synchronization.
Weights were random; this was a latency comparison, not a parity test against a
saved checkpoint.

The integrated benchmark script can now exercise the torch evaluator directly:

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

This is a much larger gap than the earlier 2-4x rough estimate. Because
previous C MCTS timing showed `ModelEvaluator.evaluate_features(...)` was more
than 98% of `search_batch(...)`, replacing only the evaluator can plausibly
make self-play and arena eval costs almost disappear relative to training.

Updated rough iteration estimates:

| Scenario | Self-play | Training | Arena | Iteration | 200 Iterations |
|---|---:|---:|---:|---:|---:|
| current tinygrad | ~47s | ~48s | ~48s | ~135-143s | ~7.5-8.0h |
| PyTorch evaluator only | ~3-5s | ~48s | ~3-5s | ~55-60s | ~3.0-3.4h |

The exact numbers still need full-loop validation. The microbenchmark does not
include checkpoint weight conversion, C MCTS feed/collect overhead after eval is
reduced, Python loop overhead, or changes in game length. It is strong enough,
however, to make a PyTorch evaluator backend the next highest-ROI change.

The integrated PyTorch evaluator still pads to power-of-two batch buckets. An
unpadded smoke run produced many unique CUDA shapes and spent `13.3s` in
self-play eval for iteration 1. Power-of-two padding collapsed the shape set and
reduced the same run to `3.29s` of eval time.

Once PyTorch eval is integrated, `leaves_per_batch` should be swept again. The
current `64` value was chosen mainly to reduce expensive tinygrad evaluator
calls. If PyTorch eval makes calls cheap, lower values such as `8`, `16`, or
`32` may recover search quality by feeding/backing up neural evaluations more
often, without costing much wall time.

Implementation status: the CUDA profile now uses `selfplay.evaluator_backend:
torch`. CPU and CUDA parity checks against the tinygrad evaluator were within
normal floating-point tolerance:

| Device | Policy Max Abs Diff | Value Max Abs Diff |
|---|---:|---:|
| CPU, batch 7 | ~2.8e-9 | ~1.0e-7 |
| CUDA, batch 64 | ~4.0e-7 | ~1.1e-5 |

A small C MCTS + CUDA smoke run with 8 games and 8 simulations completed with
the torch evaluator:

```text
selfplay_seconds=1.184s
eval_selfplay_candidate_seconds=1.141s
eval_selfplay_candidate_calls=68
eval_selfplay_candidate_avg_seconds=0.0168s
```

The full CUDA iteration-1 warmup run, using 64 games and 96 simulations but no
training/arena, dropped from the previous tinygrad baseline of `25.42s`
self-play time to:

```text
selfplay_seconds=3.591s
eval_selfplay_candidate_seconds=3.290s
eval_selfplay_candidate_calls=158
eval_selfplay_candidate_avg_seconds=0.0208s
eval_selfplay_candidate_max_bucket=4096
eval_selfplay_candidate_pad_ratio=1.1685
```

Use `uv run --extra omok ...` or otherwise install PyTorch before running the
CUDA profile. Without PyTorch installed, `evaluator_backend: torch` fails fast
with an installation hint.

## 4-Iteration Phase Benchmark

After adding phase timing fields to `metrics.jsonl`, a fresh run from an empty
`checkpoints/omok_full_cuda` with `leaves_per_batch: 64` and
`search_threads: auto` produced:

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

For trained iterations 2-4, average phase times were:

| Phase | Average |
|---|---:|
| total | 134.52s |
| self-play | 38.22s |
| training | 47.99s |
| arena | 48.14s |
| checkpoint | 1.03s |

The current CUDA profile is therefore no longer dominated by one CPU MCTS
section. At 160 simulations, self-play, training updates, and arena are all
material contributors. Future optimization should avoid focusing only on
self-play MCTS traversal.

## Built-In Training Metrics

The trainer now writes enough per-iteration fields to diagnose the common Omok
CUDA bottlenecks without running a separate microbenchmark first.

Top-level phase timings:

| Field | Meaning |
|---|---|
| `duration_seconds` | full iteration wall time, including checkpoint save |
| `selfplay_seconds` | full self-play phase wall time |
| `train_seconds` | optimizer update phase wall time |
| `arena_seconds` | candidate-vs-best arena wall time |
| `checkpoint_seconds` | checkpoint and runtime state save time |

MCTS search timings are grouped by phase, for example
`search_selfplay_candidate_*`, `search_selfplay_best_*`,
`search_arena_candidate_*`, and `search_arena_best_*`:

| Suffix | Meaning |
|---|---|
| `_calls` | number of `MCTS.search_batch(...)` calls |
| `_seconds` | total search wall time, including evaluator calls |
| `_avg_seconds` | average search call latency |
| `_states` / `_avg_states` | active game states seen by search calls |
| `_requested_leaves` | requested MCTS leaf visits, roughly states * simulations |
| `_max_states` | largest active batch seen by that phase |
| `_max_simulations` | largest simulation count used by that phase |
| `_max_leaves_per_batch` | largest configured leaf batch used by that phase |

Evaluator timings are grouped by the same phase names with `eval_*` fields:

| Suffix | Meaning |
|---|---|
| `_calls` | number of neural evaluator calls |
| `_seconds` | total evaluator wall time |
| `_avg_seconds` | average evaluator call latency |
| `_positions` | unpadded board positions evaluated |
| `_padded_positions` | positions after power-of-two tinygrad bucket padding |
| `_pad_ratio` | padded_positions / positions |
| `_avg_batch` / `_max_batch` | actual evaluator batch sizes |
| `_max_bucket` | largest padded tinygrad bucket |
| `_bucket_counts` | count of calls per padded bucket size |

Training update timings:

| Field | Meaning |
|---|---|
| `train_metric_updates` | measured optimizer updates |
| `train_metric_samples` | total sampled replay rows |
| `train_sample_seconds` | replay sample and tensor creation time |
| `train_forward_seconds` | model forward graph construction time |
| `train_loss_seconds` | policy/value loss graph construction time |
| `train_backward_seconds` | backward graph construction time |
| `train_optimizer_seconds` | optimizer step graph construction/execution time |
| `train_sync_seconds` | loss materialization and synchronization time |
| `train_measured_seconds` | sum of measured training substeps |

Use the separate `scripts/bench_omok_evaluator.py` only when isolating evaluator
latency by synthetic batch size or comparing tinygrad against another inference
backend. Normal training runs should rely on `metrics.jsonl` first.

## Current Handoff Notes

- C MCTS traversal is fast after tree-level threading and node arena
  allocation.
- `search_threads: auto` is useful as a ceiling, but high CPU utilization should
  not be expected because C collection is a tiny part of wall time.
- `leaves_per_batch: 64` improved early CUDA throughput by reducing evaluator
  calls in tinygrad. Re-sweep `8/16/32/64` after PyTorch eval is integrated;
  `64` may no longer be the best quality/speed tradeoff.
- The most promising next experiments are:
  - add a PyTorch evaluator backend for self-play and arena;
  - verify tinygrad-to-PyTorch checkpoint weight parity before using it for
    training runs;
  - run a full-loop CUDA benchmark with PyTorch eval and `leaves_per_batch: 64`;
  - sweep `leaves_per_batch` after PyTorch eval;
  - measure whether training updates are tinygrad-bound once eval is no longer
    dominant;
  - consider ONNX Runtime or TensorRT only after PyTorch eval lands, because
    training is expected to become the dominant remaining bottleneck;
  - only revisit persistent C worker pools if `collect` becomes a meaningful
    share of `search_batch` time again.

## Currently Unused Reference Fields

These fields are parsed for compatibility with reference configs, but the
current C backend does not implement async inference queues:

```yaml
selfplay:
  inference_batch_size: 512
  inference_wait_ms: 2.0
```

Useful future directions:

- `inference_batch_size`: could become a target/cap for collecting multiple C
  leaf batches before calling the evaluator.
- `inference_wait_ms`: only makes sense with an async inference server/queue.
