# Omok PyTorch Migration Plan

This document describes how to move Omok training from the current mixed
tinygrad/PyTorch state to a PyTorch-first implementation without losing
checkpoint compatibility or breaking the existing CUDA/Metal profiles.

## Current State

The codebase is intentionally mixed right now:

| Area | Current Backend | Notes |
|---|---|---|
| self-play evaluator | PyTorch on CUDA profile, tinygrad by default | selected by `selfplay.evaluator_backend` |
| arena evaluator | PyTorch on CUDA profile, tinygrad by default | same evaluator factory as self-play |
| training model | tinygrad | `PolicyValueNet` in `network.py` |
| optimizer | tinygrad | `AdamW` in `train.py` |
| replay tensors | tinygrad | `ReplayBuffer.sample_batch(...)` returns tinygrad `Tensor`s |
| checkpoints | tinygrad safetensors | model and optimizer state are tinygrad state dicts |
| ONNX export | PyTorch mirror | `export_onnx.py` has its own duplicate torch model |
| C/Python MCTS | backend-neutral evaluator interface | should not need changes for full torch migration |

The PyTorch evaluator already validates that the tinygrad model can be mirrored
accurately:

| Device | Policy Max Abs Diff | Value Max Abs Diff |
|---|---:|---:|
| CPU, batch 7 | ~2.8e-9 | ~1.0e-7 |
| CUDA, batch 64 | ~4.0e-7 | ~1.1e-5 |
| CUDA, padded batch 93 | ~7.6e-7 | ~5.4e-6 |

This means the migration risk is no longer “can PyTorch represent the network?”
It is now mostly about training semantics, checkpoint compatibility, resume
behavior, and keeping the hardware profiles understandable.

## Why Not Remove Tinygrad Immediately

Removing tinygrad in one pass would touch model definition, training,
checkpointing, optimizer resume, replay tensor creation, worker initialization,
device configuration, docs, and export paths at the same time. That would make
regressions hard to isolate.

The better path is to make PyTorch a complete training backend first, run it
side-by-side with the tinygrad path, validate metrics and checkpoints, then
remove tinygrad only after the torch path has trained successfully for enough
full iterations.

## Target Architecture

The target should be backend-selection at the trainer level, not scattered
conditionals across every file.

Recommended config shape:

```yaml
training:
  backend: torch  # tinygrad | torch

selfplay:
  evaluator_backend: torch  # tinygrad | torch | auto
```

Short term, it is acceptable to keep `selfplay.evaluator_backend` under
`selfplay` because it is already implemented. For full migration, add an
explicit training backend field rather than overloading `device` or
`evaluator_backend`.

Target module boundaries:

| Module | Responsibility |
|---|---|
| `network.py` | tinygrad model until removed |
| `torch_network.py` | canonical PyTorch `PolicyValueNet` mirror |
| `evaluator.py` | backend-neutral evaluator protocol and tinygrad evaluator |
| `torch_evaluator.py` | PyTorch evaluator wrapper around `torch_network.py` |
| `training_backend.py` | protocol for model, optimizer, train step, checkpoint IO |
| `torch_training.py` | PyTorch train step, optimizer, model clone, state conversion |
| `checkpoint.py` | format dispatch and backward-compatible checkpoint loading |
| `replay.py` | numpy replay storage plus backend-specific tensor batch builders |

Do not change MCTS interfaces for this migration. MCTS should continue to see
only `Evaluator.evaluate(...)` and `Evaluator.evaluate_features(...)`.

## Phase 1: Consolidate PyTorch Model Code

Goal: one PyTorch network definition, used by evaluator, ONNX export, parity
tests, and future training.

Work:

- Move `TorchPolicyValueNet`, `TorchResidualBlock`, and `TorchSEBlock` from
  `torch_evaluator.py` into a new `torch_network.py`.
- Replace the duplicate `_build_torch_model(...)` implementation in
  `export_onnx.py` with `torch_network.TorchPolicyValueNet`.
- Keep state dict keys exactly matched to tinygrad:
  - `stem_conv.weight`
  - `tower.N.conv1.weight`
  - `tower.N.se.fc1.weight`
  - `policy_fc.weight`
  - etc.
- Keep tinygrad-to-torch state conversion as a shared helper, not private to
  the evaluator.

Validation:

```bash
uv run --extra omok python -m py_compile \
  src/coolrl/omok/torch_network.py \
  src/coolrl/omok/torch_evaluator.py \
  src/coolrl/omok/export_onnx.py
```

Run parity checks:

```bash
uv run --extra omok python scripts/check_omok_torch_parity.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --batches 7,64,93,2048
```

The parity script does not exist yet; create it rather than relying on ad hoc
one-liners.

Acceptance criteria:

- PyTorch evaluator parity remains within the current tolerance.
- ONNX export still works.
- No behavior change in training.

## Phase 2: Add PyTorch Training Backend Side-by-Side

Goal: make torch training possible without deleting tinygrad training.

Work:

- Add `TrainingBackend` protocol with operations the trainer needs:
  - build model
  - clone model
  - build evaluator
  - build optimizer
  - train one batch
  - save/load model state
  - save/load optimizer state
- Implement `TinygradTrainingBackend` by wrapping current behavior.
- Implement `TorchTrainingBackend` using:
  - `torch.optim.AdamW`
  - `torch.nn.functional.log_softmax`
  - MSE value loss
  - current loss weights from config
- Move the current training-step timing into backend-returned metrics.
- Keep replay storage as numpy/deque. Add a torch batch path rather than
  changing storage:

```python
ReplayBuffer.sample_batch_numpy(...)
ReplayBuffer.sample_batch_tinygrad(...)
ReplayBuffer.sample_batch_torch(...)
```

The existing `ReplayBuffer.sample_batch(...)` can temporarily remain as the
tinygrad path.

Training math to match:

```text
policy_loss = -(target_policy * log_softmax(logits)).sum(axis=1).mean()
value_loss = mean((value - target_value) ** 2)
loss = policy_weight * policy_loss + value_weight * value_loss
```

Important semantic details:

- Use `model.train()` during optimizer updates.
- Use `model.eval()` for evaluator and arena.
- Preserve BatchNorm running statistics in checkpoints.
- Preserve `recency_temperature` sampling behavior exactly.
- Keep `value_discount`, symmetry augmentation, policy target shape, and value
  target shape unchanged.

Acceptance criteria:

- `training.backend: tinygrad` produces the same behavior as today.
- `training.backend: torch` can run smoke/quick configs end-to-end.
- Metrics include enough backend labels to compare runs:
  - `training_backend`
  - `evaluator_backend`
  - phase seconds
  - loss fields
  - eval/search metrics

## Phase 3: Checkpoint Compatibility

Goal: torch training can start from old tinygrad checkpoints and resume from new
torch checkpoints.

Recommended checkpoint policy:

| Format | Use | Notes |
|---|---|---|
| `coolrl.omok.v1` | existing tinygrad model checkpoints | keep readable |
| `coolrl.omok.torch.v1` | new torch model checkpoints | explicit backend metadata |
| `coolrl.omok.optimizer.v1` | existing tinygrad optimizer state | keep readable only for tinygrad |
| `coolrl.omok.torch.optimizer.v1` | new torch optimizer state | `torch.save` or safetensors-compatible numpy payload |

Do not silently load a tinygrad optimizer state into a torch optimizer. Model
weights are portable; optimizer internals are not worth pretending to be
portable.

Rules:

- Torch backend may load model weights from tinygrad checkpoint.
- Torch backend should initialize a fresh torch optimizer when resuming from a
  tinygrad-only optimizer state.
- Torch backend should fully resume optimizer state from a torch optimizer
  checkpoint.
- Metadata should record:
  - `training_backend`
  - `evaluator_backend`
  - `checkpoint_format`
  - `torch_version` when available
  - `tinygrad_version` while tinygrad is still present

Migration helper:

```bash
uv run --extra omok python -m coolrl.omok.convert_checkpoint \
  --input checkpoints/omok_full_cuda/latest.safetensors \
  --output checkpoints/omok_full_cuda_torch/latest.pt \
  --to torch
```

This converter can come after direct loading works. It is useful for cleanup
and reproducibility but should not block the side-by-side backend.

Acceptance criteria:

- Load old tinygrad `latest.safetensors` into torch model.
- Save torch `latest` and `best`.
- Resume torch run from torch runtime state.
- Resume behavior is explicit when optimizer state is not portable.

## Phase 4: Full-Loop Benchmarks

Goal: prove the torch training path is faster and stable before removing
tinygrad.

Run sequence:

1. Current mixed baseline:

```bash
rm -rf checkpoints/omok_full_cuda
uv run --extra omok python -m coolrl.omok.train \
  --config configs/omok_full_cuda.yaml \
  --max-iterations 4
```

2. Torch training backend:

```bash
rm -rf checkpoints/omok_full_cuda_torch_train
uv run --extra omok python -m coolrl.omok.train \
  --config configs/omok_full_cuda_torch.yaml \
  --max-iterations 4
```

Compare:

| Metric | Expected Direction |
|---|---|
| `selfplay_seconds` | already low from torch eval |
| `arena_seconds` | already low from torch eval |
| `train_seconds` | should drop substantially |
| `train_optimizer_seconds` | should no longer dominate |
| `duration_seconds` | should approach self-play + train + arena lower bound |
| `train_loss`, `policy_loss`, `value_loss` | should be numerically plausible |
| arena win rates | noisy, but no obvious collapse |

Current known baselines:

| Setup | Iteration 1 Self-play | Notes |
|---|---:|---|
| tinygrad evaluator | 25.42s | full CUDA warmup, 64 games, 96 sims |
| PyTorch evaluator, padded | 3.591s | same warmup shape |

Older full-iteration tinygrad-training baseline:

| Phase | Average, trained iterations 2-4 |
|---|---:|
| total | 134.52s |
| self-play | 38.22s |
| training | 47.99s |
| arena | 48.14s |
| checkpoint | 1.03s |

After torch eval, training is expected to become the dominant phase. After torch
training, re-evaluate `leaves_per_batch` and arena settings because the cost
model changes again.

## Phase 5: Re-Tune Search After Torch Training

Goal: choose quality/speed settings for the torch stack, not for the old
tinygrad stack.

Sweep:

```text
leaves_per_batch: 8, 16, 32, 64
simulation_schedule: current, maybe higher final simulations if iteration time falls
arena.games: current, maybe higher if arena becomes cheap
```

Why this matters:

- `leaves_per_batch: 64` was chosen to reduce expensive tinygrad eval calls.
- With torch eval, lower values may improve tree quality by feeding neural
  results back into MCTS more frequently.
- If torch training reduces iteration time enough, increasing simulations may
  be a better use of saved wall time than only maximizing throughput.

Compare both speed and learning quality:

- iteration wall time
- replay samples per hour
- arena promotion rate
- candidate white win rate gate
- policy entropy / visit distribution if added later
- final strength against a fixed checkpoint set

## Phase 6: Tinygrad Removal Decision

Only remove tinygrad after the torch path satisfies all of these:

- Torch evaluator and torch training are the default for CUDA.
- Quick and full CUDA configs run cleanly from scratch and from resume.
- Old tinygrad checkpoints can still be loaded or converted.
- ONNX export uses the shared torch network.
- Documentation no longer depends on tinygrad behavior for the recommended
  path.
- Metal/CPU story is decided:
  - either keep tinygrad for Metal temporarily;
  - or move Metal/CPU profiles to torch CPU/MPS;
  - or split legacy tinygrad support behind a compatibility mode.

Removal work:

- Delete tinygrad `PolicyValueNet` only after checkpoint conversion is stable.
- Remove tinygrad optimizer state handling or keep legacy loader only.
- Replace `device.py` with a torch-aware device resolver.
- Update `ReplayBuffer.sample_batch(...)` to return torch tensors by default.
- Update README and setup docs from “tinygrad trainer” to “PyTorch trainer”.
- Keep C MCTS and MCTS backend interfaces unchanged.

## Main Risks

| Risk | Mitigation |
|---|---|
| BatchNorm behavior differs | run parity in eval mode; track train-mode loss and running stats |
| optimizer state is not portable | do not promise optimizer portability across backends |
| checkpoint format confusion | explicit `checkpoint_format` and `training_backend` metadata |
| duplicated torch model definitions drift | consolidate in `torch_network.py` first |
| PyTorch dependency size | keep under `omok` extra; document `uv run --extra omok` |
| Metal profile regression | do not flip Metal to torch until tested separately |
| speedup smaller than expected | rely on full-loop metrics, not microbenchmarks only |
| learning behavior changes | compare arena gates, losses, and fixed-checkpoint matches |

## Recommended Next Commit Sequence

1. `Extract shared PyTorch Omok network`
   - create `torch_network.py`
   - update evaluator/export to use it
   - add parity script

2. `Add torch training backend skeleton`
   - add backend protocol
   - wrap existing tinygrad backend
   - no behavior change by default

3. `Implement torch optimizer updates`
   - torch replay tensor path
   - torch train step
   - smoke and quick runs

4. `Add torch checkpoint format`
   - save/load model
   - resume torch optimizer
   - load tinygrad weights into torch model with fresh optimizer

5. `Benchmark full CUDA torch training`
   - document iteration timings
   - update `docs/omok_cuda_tuning.md`

6. `Re-tune torch search settings`
   - sweep `leaves_per_batch`
   - update CUDA config only after measured results

7. `Decide tinygrad deprecation`
   - remove or mark legacy after torch path is stable

## Open Questions

- Should torch checkpoints use `.pt`, `.safetensors`, or both?
- Do we care about preserving optimizer state when converting tinygrad runs to
  torch, or is fresh optimizer acceptable?
- Should `training.backend` live at top level or under `optimization`?
- Should the Metal profile move to torch MPS, torch CPU workers, or remain
  tinygrad until separately benchmarked?
- Should the GUI/ONNX path depend on torch network directly, or remain
  checkpoint-format based?
- How much saved time should be reinvested in higher MCTS simulations versus
  more iterations?
