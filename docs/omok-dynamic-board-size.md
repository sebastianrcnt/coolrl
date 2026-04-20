# Omok dynamic board size migration plan

## Goal

Make `coolrl.omok` support multiple board sizes, especially 9x9 and 15x15, without switching the whole codebase through compile-time constants.

The target result is:

- `rules.board_size: 9` continues to work.
- `rules.board_size: 15` works for training, arena, checkpoints, export, and play.
- Python MCTS and C MCTS behave consistently for any supported square board size.
- Existing 9x9 checkpoints are not silently reused for 15x15 runs.

## Non-goals

- Do not preserve compatibility with old C MCTS shared libraries. Rebuild the extension after changing the C API.
- Do not support arbitrary rectangular boards. Only square `N x N`.
- Do not mix replay buffers or checkpoints across board sizes.
- Do not optimize 15x15 strength in this migration. First make correctness and shape handling reliable.

## Current blockers

The codebase is currently 9x9-fixed in several places:

- `src/coolrl/omok/board.py`
  - `GameState.__post_init__` rejects board sizes other than 9.
- `src/coolrl/omok/config.py`
  - `RulesConfig.__post_init__` rejects board sizes other than 9.
- `src/coolrl/omok/torch_network.py`
  - `PolicyValueNet` rejects board sizes other than 9.
- `src/coolrl/omok/cmcts/include/mcts.h`
  - `CMCTS_BOARD_SIZE` is `9`.
  - `CMCTS_ACTION_SIZE` is `81`.
  - C structs use fixed-size arrays derived from these constants.
- `src/coolrl/omok/cmcts_wrapper.py`
  - `ACTION_SIZE = 81`.
  - Feature tensors are allocated as `[N, 4, 9, 9]`.
- `src/coolrl/omok/gui.py`
  - `BOARD_SIZE = 9`.
- `src/coolrl/omok/web/index.html`
  - `const BOARD_SIZE = 9`.
- `src/coolrl/omok/plot_metrics.py`
  - Uniform policy entropy reference is hardcoded to `ln(81)`.

## Recommended implementation order

### 1. Loosen Python config and game state

Files:

- `src/coolrl/omok/config.py`
- `src/coolrl/omok/board.py`

Changes:

- Replace the 9x9-only validation with a supported-size validation.
- Recommended initial support: `board_size in {9, 15}`.
- Keep square board assumption.
- Keep all action encoding as row-major integer action:

```text
action = row * board_size + col
row, col = divmod(action, board_size)
action_size = board_size * board_size
```

Acceptance criteria:

- `GameState(9)` works.
- `GameState(15)` works.
- `GameState(13)` fails unless explicitly added to supported sizes.
- `legal_moves()` returns shape `[board_size * board_size]`.
- `feature_planes()` returns shape `[4, board_size, board_size]`.

### 2. Make the PyTorch network board-size generic

File:

- `src/coolrl/omok/torch_network.py`

Changes:

- Remove the 9x9-only guard.
- Keep `action_size = board_size * board_size`.
- Ensure the policy head outputs exactly `action_size`.
- Ensure the value head works with adaptive pooling or otherwise does not assume `9 * 9`.

Important check:

- If the value head flattens spatial dimensions directly, it must be changed to avoid a fixed 9x9 feature size.
- Preferred design: use global/adaptive pooling before the value MLP.

Acceptance criteria:

- `PolicyValueNet(9, cfg)(torch.zeros(2, 4, 9, 9))` returns policy `[2, 81]`, value `[2]`.
- `PolicyValueNet(15, cfg)(torch.zeros(2, 4, 15, 15))` returns policy `[2, 225]`, value `[2]`.

### 3. Convert C MCTS from compile-time board constants to runtime board size

Files:

- `src/coolrl/omok/cmcts/include/mcts.h`
- `src/coolrl/omok/cmcts/src/internal.h`
- `src/coolrl/omok/cmcts/src/api.c`
- `src/coolrl/omok/cmcts/src/board.c`
- `src/coolrl/omok/cmcts/src/tree.c`
- `src/coolrl/omok/cmcts/src/mcts.c`

This is the most important part.

Current C design:

```c
#define CMCTS_BOARD_SIZE 9
#define CMCTS_ACTION_SIZE 81
int8_t board[CMCTS_ACTION_SIZE];
Node *children[CMCTS_ACTION_SIZE];
Node *path[CMCTS_ACTION_SIZE + 1];
```

Target C design:

```c
struct MctsTree {
  int board_size;
  int action_size;
  int feature_planes;
  int feature_stride;
  ...
};
```

Recommended API change:

```c
MctsTree *mcts_tree_new(int board_size, float c_puct, float virtual_loss, int exactly_five);
int mcts_tree_action_size(const MctsTree *tree);
int mcts_tree_feature_stride(const MctsTree *tree);
```

State storage:

- Change `CmctsState.board` from fixed array to dynamically allocated `int8_t *board`, or store a pointer plus ownership flag.
- Simpler and safer: each `CmctsState` owns its board buffer.
- Add helper functions:

```c
int state_init(CmctsState *state, int board_size, const int8_t *board, ...);
void state_free(CmctsState *state);
int state_copy(CmctsState *dst, const CmctsState *src);
```

Node children:

- Change `Node *children[CMCTS_ACTION_SIZE]` to `Node **children`.
- Allocate `children` with `action_size` entries in `tree_node_new`.
- Free children when freeing node blocks or nodes.

Pending paths:

- Change `Node *path[CMCTS_ACTION_SIZE + 1]` to dynamic storage.
- Recommended low-risk option:
  - Store `Node **path` inside `PendingEval`.
  - Allocate `action_size + 1` entries when pushing pending eval.
  - Free or reuse these buffers in `tree_clear_pending_*`.
- Alternative:
  - Keep pending path capacity on `MctsTree` and allocate fixed-size per pending entry based on `tree->action_size + 1`.

Feature writing:

- Replace all `CMCTS_FEATURE_STRIDE` usage with `tree->feature_stride`.
- `feature_stride = 4 * action_size`.
- `state_write_features` should accept either `tree` or explicit `board_size/action_size`.

Selection/expansion loops:

- Replace loops over `CMCTS_ACTION_SIZE` with `tree->action_size` or `state->action_size`.
- Replace row/col math with runtime `board_size`.

Exact-five logic:

- Keep existing win logic semantics.
- Make direction scans use runtime `board_size`.

Acceptance criteria:

- C MCTS can create a 9x9 tree and a 15x15 tree in the same Python process.
- 9x9 feature stride is `324`.
- 15x15 feature stride is `900`.
- 9x9 visit counts shape is `[81]`.
- 15x15 visit counts shape is `[225]`.
- Terminal detection works on both board sizes.

### 4. Update Python C wrapper

File:

- `src/coolrl/omok/cmcts_wrapper.py`

Changes:

- Remove module-level `ACTION_SIZE = 81` as a global truth.
- Derive `board_size` and `action_size` from `states[0]`.
- Require every state in one `search_batch` call to have the same `board_size`.
- Pass `board_size` into `mcts_tree_new`.
- Allocate features dynamically:

```python
root_features = np.empty((len(states), 4, board_size, board_size), dtype=np.float32)
leaf_features = np.empty((max_leaves, 4, board_size, board_size), dtype=np.float32)
counts = np.empty((len(active_roots), action_size), dtype=np.float32)
```

- Ensure C API pointers still receive flattened contiguous arrays.
- For reused roots, ensure root board size matches incoming state board size.

Acceptance criteria:

- Python MCTS and C MCTS both return policy shape `[81]` on 9x9.
- Python MCTS and C MCTS both return policy shape `[225]` on 15x15.
- Mixed board sizes inside one `search_batch` fail with a clear error.

### 5. Keep Python MCTS board-size generic

File:

- `src/coolrl/omok/mcts.py`

Likely already mostly generic because it uses `state.action_size` and `state.legal_moves()`.

Review for:

- Any hardcoded `81`.
- Any hardcoded `9`.
- Any assumptions that roots from one board size can be reused on another.

Acceptance criteria:

- Python MCTS runs one search on 9x9.
- Python MCTS runs one search on 15x15.

### 6. Update configs and checkpoint separation

Files:

- `configs/omok_full_cuda.yaml`
- Optionally add `configs/omok_full_cuda_15x15.yaml`.

Recommended:

- Do not mutate the main 9x9 profile if ongoing experiments depend on it.
- Add a separate 15x15 profile:

```yaml
experiment_name: omok_full_cuda_15x15

rules:
  board_size: 15
  exactly_five: true

checkpoint:
  directory: checkpoints/omok_full_cuda_15x15
```

Training hyperparameters need to scale:

- Policy action space increases from 81 to 225.
- Game length increases.
- MCTS branching factor increases.
- Replay capacity and games per iteration may need adjustment.

Reasonable first 15x15 smoke profile:

```yaml
selfplay:
  games_per_iteration: 16
  batch_size: 16
  simulations: use existing schedule or start lower
  leaves_per_batch: 8

arena:
  games: 16
  simulations: 96

optimization:
  batch_size: 256
  updates_per_iteration: 64
  replay_capacity: 100000
```

Acceptance criteria:

- 9x9 and 15x15 checkpoint directories are different.
- Config saved in checkpoints records `rules.board_size`.
- Loading a checkpoint with mismatched board size fails clearly.

### 7. Update export and web/GUI paths

Files:

- `src/coolrl/omok/export_onnx.py`
- `src/coolrl/omok/convert_onnx.py`
- `src/coolrl/omok/gui.py`
- `src/coolrl/omok/web/index.html`

Export:

- ONNX dummy input already uses config board size; verify output policy shape is dynamic enough or explicitly 225 for 15x15 exports.
- Include board size metadata in exported artifacts.

GUI:

- Replace `BOARD_SIZE = 9` with config/model metadata or a CLI flag.
- Create `GameState(board_size, exactly_five)` from that value.

Web:

- Replace `const BOARD_SIZE = 9` with model metadata or a UI selector.
- Ensure ONNX policy output length matches `BOARD_SIZE * BOARD_SIZE`.

Acceptance criteria:

- 9x9 exported model plays on web.
- 15x15 exported model plays on web.
- If model policy length and board size mismatch, UI shows a clear error.

### 8. Update plotting metrics

File:

- `src/coolrl/omok/plot_metrics.py`

Changes:

- Replace `UNIFORM_POLICY_ENTROPY_9X9 = ln(81)` with a dynamic value.
- Prefer reading board size from metrics config if stored.
- If metrics do not contain board size yet, infer from config path or allow CLI override.

Formula:

```python
uniform_policy_entropy = np.log(board_size * board_size)
```

Acceptance criteria:

- 9x9 plots show `ln(81)`.
- 15x15 plots show `ln(225)`.

## Required tests or smoke checks

Do these after implementation.

### Python-only smoke

- Construct `GameState(9)` and `GameState(15)`.
- Apply legal moves.
- Verify win detection for horizontal, vertical, diagonal, and anti-diagonal five.
- Verify overline behavior with `exactly_five: true`.
- Run Python MCTS for a few simulations on both sizes using a dummy evaluator.

### C MCTS smoke

- Rebuild extension.
- Construct C MCTS roots for 9x9 and 15x15.
- Run `search_batch` with dummy evaluator on both sizes.
- Verify returned policy lengths are 81 and 225.
- Verify no memory error under repeated searches.

### Training smoke

- Run a tiny 9x9 config for 1 iteration.
- Run a tiny 15x15 config for 1 iteration.
- Confirm metrics are written.
- Confirm checkpoint config records board size.

### Parity check

With a deterministic uniform evaluator:

- Run Python MCTS and C MCTS from the empty 9x9 board with the same simulations.
- Run Python MCTS and C MCTS from the empty 15x15 board with the same simulations.
- Compare:
  - policy shape
  - nonzero visit count count
  - legal move masking
  - terminal backup behavior

Exact visit distributions do not need to match perfectly unless traversal tie-breaking is identical.

## Risks

- Dynamic allocation in C can introduce leaks or stale pointers in pending eval queues.
- Re-rooting C trees after actions becomes more fragile because child arrays are dynamic.
- 15x15 increases action space by 2.78x, so MCTS quality at the old simulation count will likely drop.
- Existing checkpoints with 9x9 policy heads cannot load into 15x15 networks.
- Web and GUI may silently assume 9x9 unless model metadata is checked.

## Suggested PR breakdown

### PR 1: Python 15x15 support without C MCTS

- Loosen config and `GameState`.
- Make network generic.
- Verify Python MCTS works for 9x9 and 15x15.
- Keep C MCTS rejecting non-9x9 with a clear error.

### PR 2: Runtime-sized C MCTS

- Change C API.
- Add dynamic board/action storage.
- Update wrapper.
- Rebuild extension.
- Add C smoke checks.

### PR 3: Tooling and UX

- Add 15x15 config.
- Update plotting.
- Update export/web/GUI.
- Add mismatch checks for checkpoint/model board size.

This split is safer than changing every surface at once.
