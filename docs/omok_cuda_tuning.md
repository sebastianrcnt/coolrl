# Omok CUDA Self-Play Tuning Notes

These notes capture measurements from the RTX 3090 CUDA profile
(`configs/omok_full_cuda.yaml`) after adding the C MCTS backend.

## Current Recommendation

For CUDA self-play on a discrete NVIDIA GPU:

```yaml
selfplay:
  mcts_backend: c
  num_workers: 0
  batch_size: 64
  leaves_per_batch: 16
  search_threads: 4
```

Keep `num_workers: 0` for the CUDA profile. The multi-process worker path pins
tinygrad inference to CPU in `selfplay_worker.py`, so `num_workers: auto` moves
self-play neural network inference off the GPU.

## Why `leaves_per_batch: 16`

With C MCTS, tree traversal is faster than the old Python tree walk. The
previous `leaves_per_batch: 8` produced a maximum self-play inference batch of:

```text
64 active games * 8 leaves = 512 positions
```

Raising it to 16 lets CUDA see:

```text
64 active games * 16 leaves = 1024 positions
```

Observed self-play timings on the full CUDA profile:

| Setting | Iteration | Self-play time | Notes |
|---|---:|---:|---|
| `leaves_per_batch: 8` | 2 | ~44s | no self-play JIT repeats |
| `leaves_per_batch: 8` | 3 | ~53s | longer games |
| `leaves_per_batch: 16` | 2 | ~31s | max bucket 1024 |
| `leaves_per_batch: 16` | 3 | ~37s | longer games |

The larger leaf batch reduced self-play time by roughly 30-40% in these runs.

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
