# Omok MCTS Memory Incident

This note records the 2026-04-21 memory blow-up observed while starting:

```bash
uv run python -m coolrl.omok.train --config configs/omok15_full_cuda_hdd.yaml
```

The process consumed 64 GB of system RAM, filled about 8 GB of swap, and caused
heavy SSD I/O before dying. The checkpoint directory only reached startup state,
so the failure happened during the first 15x15 self-play phase rather than
during replay serialization or optimizer training.

## Symptoms

- `replay.pkl` remained effectively empty.
- `trainer_state.json` showed `iteration: 0` and `status: startup`.
- System RAM grew until swap was exhausted.
- Disk I/O spiked because the OS was paging, not because checkpoints were large.

## Root Cause

The immediate cause was C MCTS tree lifetime management after switching to
arena allocation.

Each C `MctsTree` owns arena blocks for all nodes allocated during a game. Before
the fix, `mcts_tree_advance()` moved `tree->root` to the selected child but did
not release arena blocks holding the previous root and all unselected sibling
branches. That made old search branches remain alive for the rest of the game.

This was tolerable enough to hide on some 9x9 runs but became explosive on
15x15:

```text
9x9 action_size  = 81
15x15 action_size = 225
```

Early-game expansion creates one child per legal action. A dense child-pointer
array per node makes the board-size cost roughly quadratic in the action space
for broad shallow expansion. The 15x15 full CUDA profile also starts at more
simulations than the 9x9 full CUDA profile, so the same lifetime bug became a
system-level OOM.

## Fix

The C backend now does two things:

- unexpanded nodes no longer allocate their `children` pointer array;
- `mcts_tree_advance()` clones only the selected child subtree into a fresh arena
  and then frees the old arena.

This preserves tree reuse for the chosen line while dropping unselected branches
after every move. It also keeps the existing C/Python parity behavior where
deep tree reuse affects later search results.

The Rust backend already dropped unselected branches when advancing because its
root is a `Box<TreeNode>` and the selected child is moved out with `take()`.
Rust was still changed to lazy-allocate child vectors for unexpanded nodes so
15x15 searches do not pay for dense child arrays before expansion.

## Operational Guidance

- Do not run large 15x15 profiles with the old C backend build.
- Keep 15x15 full CUDA configs on `mcts_backend: rust` until C memory behavior
  has been profiled under long self-play runs.
- If RAM grows while `replay.pkl` stays small, suspect MCTS tree lifetime or
  search-batch memory before suspecting replay persistence.
- If RSS grows mainly at iteration boundaries and `replay.pkl` is large, inspect
  replay capacity and checkpoint serialization instead.

## Validation

The fix was validated without running full training:

```bash
uv run python setup.py build_ext --inplace
cargo fmt --manifest-path src/coolrl/omok/rmcts/Cargo.toml --check
cargo test --locked --manifest-path src/coolrl/omok/rmcts/Cargo.toml
uv run --with pytest pytest tests/omok/test_mcts_backends_parity.py tests/omok/test_board_size.py
```

Expected result at the time of the fix:

```text
143 passed, 3 skipped
```
