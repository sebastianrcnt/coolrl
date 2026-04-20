# 15x15 Omok RL

15x15 board variant of the Omok self-play pipeline. Code structure mirrors
`src/coolrl/omok/` (see that package's README for architectural detail and usage
patterns); this package is a sibling copy with hyperparameters and fixed
constants retuned for the larger board.

## Quick start

```bash
# Smoke (1 iter, tiny net) — confirms the pipeline boots on CPU
uv run python -m coolrl.omok15.train --config configs/omok15_smoke.yaml --device CPU

# Quick sanity (20 iter, small net)
uv run python -m coolrl.omok15.train --config configs/omok15_quick.yaml --device CPU

# Full CUDA reference run
uv run python -m coolrl.omok15.train --config configs/omok15_full_cuda.yaml
```

## Building the C MCTS backend

```bash
uv run python setup.py build_ext --inplace
```

This builds `coolrl.omok._cmcts_c` and `coolrl.omok15._cmcts_c` side by side.
The two extensions share sources but are compiled with different
`CMCTS_BOARD_SIZE`/`CMCTS_ACTION_SIZE` constants.

## Differences from `coolrl.omok`

- `RulesConfig.board_size` fixed to `15`
- `ACTION_SIZE = 225`, `FEATURE_STRIDE = 900` in `cmcts_wrapper.py`
- `cmcts/include/mcts.h` compile-time constants bumped to 15 / 225
- Hyperparameters retuned in `configs/omok15_*.yaml` (bigger network, more
  simulations, smaller dirichlet alpha, larger replay buffer)
- No GUI or web UI in this package — see `coolrl.omok` or the user's own GUI.
