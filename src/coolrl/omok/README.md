# 9x9 Omok RL

This package trains a 9x9 Omok agent with PyTorch, self-play MCTS, a
policy/value network, replay, checkpointing, arena promotion, and a Pygame GUI.

On a MacBook, training/evaluation can run on Apple Silicon via PyTorch MPS when
available, otherwise PyTorch CPU fallback is used.

## Quick Start

Use the smoke config first. It is intentionally tiny and exists to verify that the full pipeline works.

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device CPU
```

Run a short local training session:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --device CPU
```

Resume from a checkpoint directory:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --resume checkpoints/omok_quick --device CPU
```

Export a checkpoint to ONNX and open the GUI against it:

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omok_quick \
    --output exports/omok_quick.onnx

uv run python -m coolrl.omok.gui --model exports/omok_quick.onnx
```

Run a full-sized profile for your machine:

```bash
uv run python -m coolrl.omok.train --config configs/omok_full_metal.yaml
uv run python -m coolrl.omok.train --config configs/omok_full_cuda.yaml
```

The full profiles use the C MCTS backend. An experimental Rust implementation
also lives at `src/coolrl/omok/rmcts/`, and can be selected with
`selfplay.mcts_backend: rust` (currently routed through a Python shim while native
integration is being finalized). If a source checkout cannot
find the compiled extension, build it in place:

```bash
uv run --with setuptools python setup.py build_ext --inplace
```

Useful GUI keys:

- Left click: place a stone.
- `R`: reset the game.
- `S`: swap human side.
- `M`: force an AI move.
- `O`: apply a deterministic opening from the current seed.
- `[` / `]`: decrease or increase opening seed.
- `Esc`: quit.

GUI CLI options:

| Flag | Default | Description |
|---|---|---|
| `--model FILE.onnx` | none | ONNX model file. Omit for two-player or UI testing. |
| `--device auto\|cpu\|cuda\|coreml` | `auto` | ONNX Runtime execution provider. |
| `--simulations N` | `64` | MCTS simulations per AI move. |
| `--human-color black\|white` | `white` | Which color the human plays. |
| `--seed N` | `0` | Opening seed (also adjustable with `[`/`]` in-game). |

## Configs

`configs/omok_smoke.yaml` is for debugging the plumbing:

- 1 iteration.
- 1 self-play game.
- 2 MCTS simulations per move.
- small network: `channels=16`, `blocks=1`.
- arena disabled.

`configs/omok_quick.yaml` is a short real run:

- 20 iterations.
- 8 self-play games per iteration.
- 8 active self-play games per batch (matches `games_per_iteration`).
- MCTS schedule from 8 to 16 simulations.
- larger network: `channels=32`, `blocks=2`.
- 16 optimizer updates per iteration.
- small arena enabled.

For actual full runs, prefer the hardware-specific presets:

- `configs/omok_full_cuda.yaml`: NVIDIA/discrete GPU profile. It uses `device: CUDA`, C MCTS, `evaluator_backend: torch`, `num_workers: 0`, `batch_size: 64`, and `leaves_per_batch: 64` so self-play and arena inference stay on the GPU in large batches.
- `configs/omok_full_metal.yaml`: Apple Silicon profile. It uses `device: METAL`, C MCTS, smaller self-play chunks, `num_workers: auto`, and CPU worker parallelism so several games can be generated concurrently while avoiding shared Metal contexts across spawned processes.

Compatibility note: the full profiles keep reference fields such as `use_amp`,
`search_threads`, `inference_batch_size`, `inference_wait_ms`, `virtual_loss`, and
`grad_clip` for config compatibility. The C backend uses `search_threads` for tree-level
parallel collection across active games, but it does not implement same-tree
virtual-loss search or async inference queues. The active self-play throughput
knobs are `selfplay.batch_size`, `selfplay.num_workers`,
`selfplay.leaves_per_batch`, `selfplay.search_threads`, and
`selfplay.evaluator_backend` remains in the schema for compatibility, but the supported runtime evaluator is PyTorch.

For a longer run, copy `configs/omok_quick.yaml` and increase:

- `max_iterations`
- `selfplay.games_per_iteration`
- `selfplay.simulation_schedule[].simulations`
- `optimization.updates_per_iteration`
- `arena.games`

## Self-Play Parallelization

There are four different kinds of parallelism here.

GPU kernel parallelism is active on CUDA and MPS when available. This is the most
important layer for neural network inference and training.

Training batch parallelism is controlled by:

```yaml
optimization:
  batch_size: 64
```

Larger batches feed more work to the accelerator per optimizer step, but use
more memory. On an Apple M2 with 16GB RAM, start with `32` or `64`; try `128`
only after the run is stable. On a discrete GPU, `256` is the current full
profile default.

Self-play search throughput is controlled mostly by:

```yaml
selfplay:
  games_per_iteration: 16
  batch_size: 4
  leaves_per_batch: 8
  simulation_schedule:
    - fraction: 0.0
      simulations: 16
```

`selfplay.batch_size` controls how many games stay active inside one
`MCTS.search_batch(...)` call. `selfplay.leaves_per_batch` controls how many
MCTS leaves each active game contributes before one neural network evaluation.
Together they set the approximate inference batch size:

```text
active games * leaves_per_batch
```

For CUDA full self-play, `64 * 64 = 4096` positions per large evaluator call is
the current high-throughput tuning target. This value is a good baseline to
sweep again for your hardware.
See `docs/omok_cuda_tuning.md` for the RTX 3090 measurements.

Multi-process self-play is controlled by:

```yaml
selfplay:
  num_workers: auto  # or an integer like 4
```

Accepted values:

- `auto`: resolves to `os.cpu_count()` at startup and logs the resolved value, e.g. `Self-play num_workers=auto resolved to 8 (os.cpu_count=8)`. Portable across machines without editing the config.
- `0`: disables multi-process and keeps the legacy single-process path.
- any positive integer: fixed number of worker processes.

When the resolved value is `>= 1`, self-play generation is dispatched to a
`ProcessPoolExecutor` of CPU workers. Each worker receives a copy of the current
model weights (as numpy arrays), reconstructs a torch `PolicyValueNet` on CPU, and runs
MCTS + games independently. The main process keeps the configured training device
so accelerator contexts stay process-local. Results are collected back through the pool
and appended to the shared replay buffer in the main process.

Why CPU workers: this keeps accelerator contexts process-local. On Apple Silicon this can
be a reasonable trade-off because CPU workers can keep self-play moving while the
main process runs updates on MPS. On a discrete NVIDIA GPU, the trade-off is different:
the worker path moves self-play inference off CUDA and can be slower than full CUDA
self-play. Use `num_workers: 0` for CUDA full runs.

Defaults and trade-offs:

- `num_workers: 0` (default): single-process self-play. Pick this for CUDA full runs, debugging, smoke runs, and anything where profiling needs to be deterministic.
- `num_workers: 1`: one worker process. Rarely useful — use `0` instead unless you specifically want process isolation.
- `num_workers: 2` to `os.cpu_count() - 1`: useful for CPU/Metal-style self-play. More workers = more games in flight, but diminishing returns once you exceed physical cores.

Startup cost: each worker incurs startup overhead. For small configs (smoke, quick)
this can dominate a single iteration. For medium and full configs the pool cost is
amortized across many MCTS calls per iteration. A new pool is created per self-play
source per iteration (candidate and best each get their own), and workers are
initialized once via `ProcessPoolExecutor(initializer=...)` so model weights are
not re-shipped per chunk.

Work chunking: openings are split into chunks of `selfplay.batch_size` and each chunk is one task submitted to the pool. Inside a chunk, `MCTS.search_batch` still batches leaves together, so both per-chunk and per-leaf batching are active.

The knobs that matter most are:

- Increase `games_per_iteration` for more data per iteration.
- For CUDA, keep `num_workers: 0` and increase `selfplay.batch_size` /
  `selfplay.leaves_per_batch` until CUDA is fed with large enough inference
  batches.
- For Metal/CPU workers, increase `selfplay.num_workers` to saturate CPU cores
  and keep `selfplay.batch_size` small enough to create multiple chunks.
- Increase `simulations` for stronger MCTS targets.
- Increase `optimization.batch_size` and `updates_per_iteration` for more network learning.
- Keep `arena.games` small during tuning, because arena games are also MCTS-heavy.

## Suggested Recipes

Fast sanity check:

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

Heavier overnight MacBook run:

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

If the machine gets sluggish, reduce `simulations` first. If memory pressure rises, reduce `optimization.batch_size` or network `channels`.

## Torch Debug Smoke

Use this one-iteration smoke for a full pipeline check:

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --max-iterations 1 --device CPU
```

For a full GPU troubleshooting flow, use standard PyTorch tooling (`torch.profiler`,
`nsys`, vendor profiler, etc.) after confirming the smoke command is stable.

## Web GUI

A browser-based GUI runs ONNX models entirely client-side via ONNX Runtime Web (WASM).

Export PyTorch `.pt` checkpoints to ONNX:

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omokai_converted/best \
    --output exports/best.onnx
```

Export an entire directory:

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omokai_converted \
    --output exports/omokai
```

Import existing ONNX checkpoints from omokai into legacy `.safetensors` weight files:

```bash
uv run --with onnx python -m coolrl.omok.convert_onnx \
    --source /path/to/omokai/web/models \
    --output checkpoints/omokai_converted
```

Serve the web GUI and open it in a browser:

```bash
cd src/coolrl/omok/web && python -m http.server 8080
```

Then open `http://localhost:8080`, click **Load .onnx**, and upload an exported model. Controls: click to place a stone, Reset, Undo, Swap side, force AI Move, and a simulations slider (4–512).

## Visualizing Metrics

Plot training metrics from a completed or in-progress run:

```bash
# Save metrics.png in the checkpoint directory
omok-plot checkpoints/omok_full_cuda

# Open an interactive window instead
omok-plot checkpoints/omok_full_cuda --show

# Custom output path
omok-plot checkpoints/omok_full_cuda -o ~/reports/run1.png
```

You can also point directly at the `metrics.jsonl` file:

```bash
omok-plot checkpoints/omok_full_cuda/metrics.jsonl
```

The report is a 2×3 grid: train loss, policy/value loss, arena win rate (with accepted-model markers), selfplay average moves, replay buffer size, and elapsed hours.

## Outputs

Training writes under the configured checkpoint directory:

- `latest.pt`: current candidate model.
- `best.pt`: best promoted model.
- `iter_XXXX.pt`: iteration snapshots when enabled.
- `trainer_state.json`: iteration counters and run metadata.
- `replay.pkl`: replay buffer.
- `metrics.jsonl`: one JSON metrics record per iteration.
- `runtime_progress.json`: latest progress snapshot.

Legacy `.safetensors` checkpoints can still be used as model-weight-only seed input. They do not restore optimizer state. New training checkpoints are PyTorch `.pt` only.
