# 9x9 Omok RL

This package trains a 9x9 Omok agent with tinygrad, self-play MCTS, a policy/value network, replay, checkpointing, arena promotion, and a Pygame GUI.

On a MacBook, tinygrad usually picks `METAL` automatically. You can always force it with `--device METAL`.

## Quick Start

Use the smoke config first. It is intentionally tiny and exists to verify that the full pipeline works.

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device METAL
```

Run a short local training session:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --device METAL
```

Resume from a checkpoint directory:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --resume checkpoints/omok_quick --device METAL
```

Export a checkpoint to ONNX and open the GUI against it:

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
    --checkpoint checkpoints/omok_quick \
    --output exports/omok_quick.onnx

uv run python -m coolrl.omok.gui --model exports/omok_quick.onnx
```

Run the full reference-sized profile:

```bash
uv run python -m coolrl.omok.train --config configs/omok_full.yaml --device METAL
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

`configs/omok_full.yaml` mirrors the reference `rocm_unlimited.yaml` profile, adjusted only for this repo's names and checkpoint directory:

- unlimited iterations unless stopped manually.
- 64 self-play games per iteration.
- batch self-play size 64.
- network: `channels=64`, `blocks=6`, `value_hidden=128`.
- MCTS schedule from 96 to 256 simulations.
- optimizer batch 256 and 96 updates per iteration.
- replay capacity 80k and warmup 128 games.
- arena: 48 games with 192 simulations.

Compatibility note: `omok_full.yaml` keeps reference fields such as `use_amp`, `search_threads`, `inference_batch_size`, `inference_wait_ms`, `virtual_loss`, and `grad_clip` so the profile stays easy to compare with `rocm_unlimited.yaml`. The current tinygrad trainer parses those fields, but the active parallelism path is batch self-play through `selfplay.batch_size` plus METAL batch execution, not PyTorch AMP or threaded batched evaluators.

For a longer run, copy `configs/omok_quick.yaml` and increase:

- `max_iterations`
- `selfplay.games_per_iteration`
- `selfplay.simulation_schedule[].simulations`
- `optimization.updates_per_iteration`
- `arena.games`

## Parallelization On MacBook

There are four different kinds of parallelism here.

`METAL` kernel parallelism is already active when `device: auto` resolves to `METAL`, or when you pass `--device METAL`. This is the most important layer for neural network inference and training.

Training batch parallelism is controlled by:

```yaml
optimization:
  batch_size: 64
```

Larger batches feed more work to METAL per optimizer step, but use more memory. On an Apple M2 with 16GB RAM, start with `32` or `64`; try `128` only after the run is stable.

Self-play search throughput is controlled mostly by:

```yaml
selfplay:
  games_per_iteration: 16
  batch_size: 4
  simulation_schedule:
    - fraction: 0.0
      simulations: 16
```

`selfplay.batch_size` is now active: `generate_selfplay()` keeps up to that many games alive together and sends their positions through `MCTS.search_batch(...)`. This increases model inference batch size and gives METAL more work per pass.

Multi-process self-play is controlled by:

```yaml
selfplay:
  num_workers: auto  # or an integer like 4
```

Accepted values:

- `auto` (used by all shipped `configs/omok_*.yaml`): resolves to `os.cpu_count()` at startup and logs the resolved value, e.g. `Self-play num_workers=auto resolved to 8 (os.cpu_count=8)`. Portable across machines without editing the config.
- `0`: disables multi-process and keeps the legacy single-process path.
- any positive integer: fixed number of worker processes.

When the resolved value is `>= 1`, self-play generation is dispatched to a `ProcessPoolExecutor` of CPU workers. Each worker receives a copy of the current model weights (as numpy arrays), reconstructs a tinygrad `PolicyValueNet` pinned to `Device.DEFAULT = "CPU"`, and runs MCTS + games independently. The main process keeps `METAL` for training so GPU context is never shared across processes. Results are collected back through the pool and appended to the shared replay buffer in the main process.

Why CPU workers: for the 9x9 Omok network (~10k–500k params depending on config), GPU inference is latency-bound and underutilized per MCTS step. Running N concurrent CPU self-play workers, each batching their own leaves through a CPU evaluator, is much cheaper than fighting for one GPU context and it scales close to linearly with physical cores.

Defaults and trade-offs:

- `num_workers: 0` (default): single-process self-play, unchanged legacy behavior. Pick this for debugging, smoke runs, and anything where profiling needs to be deterministic.
- `num_workers: 1`: one worker process. Rarely useful — use `0` instead unless you specifically want process isolation.
- `num_workers: 2` to `os.cpu_count() - 1`: recommended for overnight runs. More workers = more games in flight, but diminishing returns once you exceed physical cores.

Startup cost: each worker does a fresh `spawn` + tinygrad import (~1–2s on macOS). For tiny configs (smoke, quick) this startup can dominate a single iteration. For medium and full configs the pool cost is amortized across many MCTS calls per iteration. A new pool is created per self-play source per iteration (candidate and best each get their own), and workers are initialized once via `ProcessPoolExecutor(initializer=...)` so model weights are not re-shipped per chunk.

Work chunking: openings are split into chunks of `selfplay.batch_size` and each chunk is one task submitted to the pool. Inside a chunk, `MCTS.search_batch` still batches leaves together, so both per-chunk and per-leaf batching are active.

The knobs that matter most are:

- Increase `games_per_iteration` for more data per iteration.
- Increase `selfplay.batch_size` until METAL is busier, but reduce it if the UI becomes sluggish.
- Increase `selfplay.num_workers` to saturate CPU cores with self-play games in parallel. Start with 2 and scale toward your physical core count.
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
  games_per_iteration: 16
  batch_size: 4
  num_workers: 4      # CPU self-play workers; tune to physical cores
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

## Debugging tinygrad

See which device tinygrad picked:

```bash
uv run python - <<'PY'
from tinygrad import Device
print(Device.DEFAULT)
PY
```

Run with tinygrad performance logs:

```bash
DEBUG=2 uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device METAL
```

Useful tinygrad debug levels:

- `DEBUG=0`: normal quiet mode.
- `DEBUG=1`: device and scheduling summaries.
- `DEBUG=2`: per-kernel timing and throughput. Usually the best profiling starting point.
- `DEBUG=4`: generated kernel source. Very noisy.
- `DEBUG=7`: buffer allocation/free logs. Extremely noisy.

Enable profiling:

```bash
PROFILE=1 DEBUG=2 uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device METAL
```

## Web GUI

A browser-based GUI runs ONNX models entirely client-side via ONNX Runtime Web (WASM).

Export safetensors checkpoints to ONNX:

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

Import existing ONNX checkpoints from omokai into safetensors:

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

## Outputs

Training writes under the configured checkpoint directory:

- `latest.safetensors`: current candidate model.
- `best.safetensors`: best promoted model.
- `iter_XXXX.safetensors`: iteration snapshots when enabled.
- `trainer_state.json`: iteration counters and run metadata.
- `optimizer.safetensors`: tinygrad optimizer state.
- `replay.pkl`: replay buffer.
- `metrics.jsonl`: one JSON metrics record per iteration.
- `runtime_progress.json`: latest progress snapshot.
