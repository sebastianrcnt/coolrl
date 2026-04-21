# Omok Dynamic Board Size

`coolrl.omok` is the single Omok implementation for supported square board
sizes. The old duplicated `coolrl.omok15` package has been retired; 15x15 runs
now use the same trainer, feature encoder, network, checkpointing, plotting,
web UI, GUI, Python MCTS, C MCTS, and Rust MCTS paths as 9x9.

## Design

Board size comes from `rules.board_size` and is carried through each runtime
surface:

```text
action = row * board_size + col
row, col = divmod(action, board_size)
action_size = board_size * board_size
feature_stride = 4 * action_size
```

The Python game state accepts square boards with `board_size >= 5`. The native C
and Rust backends currently accept sizes from 5 through 19 inclusive, which
covers 9x9, 13x13, and 15x15 without another compile-time fork.

One MCTS `search_batch` call must use a single board size. Mixing 9x9 and 15x15
states in one batch fails early with a clear error because the policy and
feature tensor shapes differ.

## MCTS Backends

All three backends share the same Python-facing search contract:

- Python MCTS derives `action_size` from `GameState.action_size`.
- C MCTS receives `board_size` when creating each tree and allocates child,
  feature, and visit-count storage from the runtime `action_size`.
- Rust MCTS mirrors the same runtime-sized tree API and exposes board/action
  metadata through FFI getters.

The C and Rust wrappers validate:

- every state in a batch has the same board size;
- reused roots match the incoming state board size;
- evaluator priors have shape `[batch, board_size * board_size]`.

See `docs/omok-mcts-memory.md` for the 15x15 memory incident that exposed why
native MCTS node lifetime and dense child storage need extra care when scaling
from 9x9 to larger boards.

## Configs

The default 9x9 presets remain:

- `configs/omok_smoke.yaml`
- `configs/omok_quick.yaml`
- `configs/omok_full_cuda.yaml`
- `configs/omok_full_metal.yaml`

The 15x15 presets are now ordinary `coolrl.omok` configs:

- `configs/omok15_smoke.yaml`
- `configs/omok15_quick.yaml`
- `configs/omok15_full_cuda.yaml`

Run them with:

```bash
uv run python -m coolrl.omok.train --config configs/omok15_smoke.yaml --device CPU
uv run python -m coolrl.omok.train --config configs/omok15_full_cuda.yaml
```

Checkpoint directories must remain separate by board size. A 9x9 checkpoint has
a policy head of length 81, while a 15x15 checkpoint has length 225; loading a
checkpoint into a network with a different board size is expected to fail.

## Tooling

ONNX export builds the dummy input from `cfg.rules.board_size`. The Pygame GUI
accepts `--board-size`, and the browser UI has a board-size selector. Both
interfaces validate that the loaded model's policy output length matches the
selected board size.

Training metrics now record `board_size`. `omok-plot` uses that value, or the
checkpoint sidecar config when needed, to draw the correct uniform policy
entropy reference:

```python
uniform_policy_entropy = np.log(board_size * board_size)
```

## Adding Another Size

For a standard square size such as 13x13, add a config with:

```yaml
rules:
  board_size: 13

checkpoint:
  directory: checkpoints/omok13_quick
```

No new package or native backend fork should be needed as long as the size stays
within the native backend limits. Add parity coverage for the new size if it
becomes an officially maintained preset rather than an ad-hoc experiment.
