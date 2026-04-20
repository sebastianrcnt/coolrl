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

Config shape (final decision):

```yaml
training:
  backend: torch  # tinygrad | torch
  optimization:
    batch_size: 256
    updates_per_iteration: 96
    learning_rate: 0.0005
    weight_decay: 0.0001
    # ...remaining existing optimization fields

selfplay:
  evaluator_backend: torch  # tinygrad | torch | auto
```

Note this is a schema change: today's configs keep `optimization:` at top
level. Moving it under `training:` is done as a prerequisite commit before
Phase 2 backend work — see "Recommended Next Commit Sequence". The rationale
for a new `training:` section (instead of `optimization.backend`) is that the
backend choice affects checkpoint IO, replay tensor path, model construction,
and device handling — not just optimizer internals.

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

Run parity checks (the parity script does not exist yet; create it rather than
relying on ad hoc one-liners):

```bash
uv run --extra omok python scripts/check_omok_torch_parity.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --batches 7,64,93
```

Larger batch sizes (e.g. 2048) should only be added after a padding strategy is
decided and memory headroom is confirmed on the target GPU; don't list them as
required until then.

The parity script must cover, at minimum:

- forward parity in `eval()` mode across the batches above (existing scope)
- BatchNorm `running_mean` / `running_var` shape and dtype match after loading a
  tinygrad checkpoint into the torch model
- forward parity in `train()` mode on one fixed batch (BN uses batch stats, so
  train-mode drift is the real migration risk, not eval-mode drift)

Acceptance criteria:

- PyTorch evaluator eval-mode parity within current tolerance across listed batches.
- Train-mode parity within a documented tolerance on one fixed batch.
- BN running stats load without shape/dtype coercion warnings.
- ONNX export still works.
- No behavior change in training.

## Phase 2: Add PyTorch Training Backend Side-by-Side

Goal: make torch training possible without deleting tinygrad training.

Prerequisites landed before Phase 2 backend work begins:

- Config rename commit moves `optimization:` under `training:` and adds a
  `training.backend` field with default `tinygrad` (no behavior change).
- `use_amp: true` top-level flag audited: confirmed either active (in which
  case Phase 4 baseline must account for it) or dormant placeholder (in which
  case remove or document). Phase 4 fp32-vs-fp32 comparison depends on this.

Decisions already locked in (see "Decisions" section near the end):

- Training checkpoint format: `.pt` (PyTorch native, bundles model + optimizer
  + scheduler + metadata in one file).
- `training.backend` lives at top level under `training:`, not under
  `optimization:`.
- Metal profile stays on tinygrad for this migration.

Phase 2 must ship the minimum checkpoint save/load needed for its own smoke and
quick runs. Full checkpoint compatibility (tinygrad→torch weight import, torch
optimizer resume, format metadata) is Phase 3; Phase 2 only needs enough to run
and resume a torch iteration end-to-end.

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
- Preserve `recency_temperature` sampling behavior exactly (this path is pure
  numpy/python today; keep it that way rather than routing through a backend).
- Keep `value_discount`, symmetry augmentation, policy target shape, and value
  target shape unchanged.
- Match tinygrad `AdamW` hyperparameters exactly: learning rate, betas, eps,
  weight decay, and any gradient clipping currently applied. Don't rely on
  PyTorch defaults — read the tinygrad call site and port the values.

Explicitly out of scope for Phase 2 (deferred to Phase 2.5 / Phase 5):

- `torch.compile(model)` — enable only after baseline torch training is stable,
  because compile errors can masquerade as training divergence.
- `torch.cuda.amp.autocast` + `GradScaler` (bf16/fp16 mixed precision).
- `channels_last` memory format for conv tower.
- `torch.backends.cudnn.benchmark = True`.

These are the real reasons to migrate to torch on CUDA; skipping them in the
first landing keeps Phase 2 ↔ Phase 4 comparisons honest (torch-fp32 vs
tinygrad-fp32), then Phase 5 measures each optimization incrementally.

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

Checkpoint policy:

| Format | Use | Notes |
|---|---|---|
| `coolrl.omok.v1` | existing tinygrad model checkpoints (safetensors) | keep readable |
| `coolrl.omok.optimizer.v1` | existing tinygrad optimizer state (safetensors) | keep readable only for tinygrad |
| `coolrl.omok.torch.v1` | new torch training checkpoint (`.pt`) | bundles model state_dict, optimizer state_dict, scheduler state, training metadata |

Training checkpoint file format: `.pt` (torch native). Rationale: PyTorch
native serialization handles nested dicts (model + optimizer + scheduler +
metadata) in a single file with no custom key-flattening. `.safetensors` is
not used for training checkpoints in this migration.

Optional inference/export artifact: `.safetensors` may be added later as a
weights-only export format for ONNX conversion, external sharing, or
zero-copy inference startup. This is out of scope for Phase 3 and must not
block training checkpoint work.

Do not silently load a tinygrad optimizer state into a torch optimizer. Model
weights are portable; optimizer internals are not worth pretending to be
portable.

Rules:

- Torch backend may load model weights from tinygrad checkpoint.
- Torch backend initializes a fresh torch optimizer when resuming from a
  tinygrad-only checkpoint. This is the accepted behavior, not a fallback —
  tinygrad → torch conversion always starts optimizer state from scratch.
  Document this explicitly in resume logs.
- Torch backend should fully resume optimizer state from a torch `.pt`
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
| `train_samples_per_second` | new; report separately from wall time so AMP/compile gains in Phase 5 are visible |
| `peak_gpu_memory_mb` | new; track to catch regressions when enabling AMP or larger batches later |
| `duration_seconds` | should approach self-play + train + arena lower bound |
| `train_loss`, `policy_loss`, `value_loss` | should be numerically plausible |
| arena win rates | noisy, but no obvious collapse |

Report fp32 torch numbers first. Do not enable AMP, `torch.compile`, or
`channels_last` during the Phase 4 baseline — compare apples to apples with
tinygrad fp32, then measure each optimization's delta in Phase 5.

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

## Phase 5: Re-Tune Search and Enable Torch-Specific Optimizations

Goal: choose quality/speed settings for the torch stack, and enable the torch
optimizations that were deliberately skipped in Phase 2/4.

Trigger condition: only run this phase if Phase 4 shows the train/selfplay time
ratio shifted by more than ~20% from the tinygrad baseline, or if
`train_seconds` is no longer the dominant phase. If the ratio is roughly
unchanged, the old tuning still holds and re-tuning is churn.

Torch optimizations to layer in, measuring each independently:

1. `torch.backends.cudnn.benchmark = True` + `channels_last` memory format.
2. `torch.compile(model)` for training and evaluator (separate compiles; the
   evaluator hot path has different batch shapes).
3. `torch.cuda.amp.autocast(dtype=torch.bfloat16)` with `GradScaler` if fp16.
   Validate arena win rate does not collapse — AMP loss curves can look fine
   while policy quality silently degrades.

Only after those land, sweep search settings:

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
| BatchNorm behavior differs | run parity in both eval and train mode; verify running stats shape/dtype on load |
| AdamW hyperparameter drift | read tinygrad call site, port values explicitly; do not rely on torch defaults |
| optimizer state is not portable | do not promise optimizer portability across backends |
| checkpoint format confusion | explicit `checkpoint_format` and `training_backend` metadata |
| duplicated torch model definitions drift | consolidate in `torch_network.py` first |
| PyTorch dependency size | keep under `omok` extra; document `uv run --extra omok` |
| Metal profile regression | do not flip Metal to torch until tested separately |
| speedup smaller than expected | rely on full-loop metrics, not microbenchmarks only |
| learning behavior changes | compare arena gates, losses, and fixed-checkpoint matches |
| AMP silently hurts policy quality | enable AMP only in Phase 5 with arena gate and fixed-checkpoint match checks |
| `torch.compile` masks training bugs | land baseline torch training without compile; add compile after curves look sane |

## Recommended Next Commit Sequence

0. `Move optimization config under training`
   - rename all 4 yaml configs: `optimization:` → `training.optimization:`
   - restructure `config.py`: `OptimizationConfig` becomes a field on a new
     `TrainingConfig`
   - add `training.backend` field defaulting to `tinygrad`
   - no behavior change; prerequisite for Phase 2

0b. `Audit use_amp flag`
   - grep for actual uses of `use_amp`; confirm whether it is active in the
     tinygrad path or a dormant placeholder
   - either document its real effect or remove it before Phase 4

1. `Extract shared PyTorch Omok network`
   - create `torch_network.py`
   - update evaluator/export to use it (ONNX export may import
     `torch_network` directly — this is the documented exception)
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
   - `.pt` training checkpoint (model + optimizer + scheduler + metadata)
   - resume torch optimizer
   - load tinygrad weights into torch model with fresh optimizer

5. `Benchmark full CUDA torch training`
   - fp32 torch vs fp32 tinygrad baseline
   - document iteration timings
   - update `docs/omok_cuda_tuning.md`

6. `Re-tune torch search settings and enable torch optimizations`
   - layer in `cudnn.benchmark`, `channels_last`, `torch.compile`, AMP
     independently with measurements
   - sweep `leaves_per_batch` only after the optimizations above
   - update CUDA config only after measured results

7. `Decide tinygrad deprecation`
   - remove or mark legacy after torch path is stable

## Decisions

Final decisions driving the migration. These replace the earlier open
questions; revisit only if new data contradicts them.

**Training checkpoint format: `.pt` first, `.safetensors` deferred.**
Training checkpoints use PyTorch native `.pt`, bundling model state_dict,
optimizer state_dict, scheduler state, and training metadata in one file.
`.safetensors` may be added later as an optional weights-only artifact for
ONNX export or external inference — not as a training-loop format.

**Config location: `training.backend` at top level under a new `training:`
section.** Backend choice is cross-cutting (checkpoint IO, replay tensor
path, model construction, device handling) and does not belong under
`optimization:`. Existing `optimization:` fields move under
`training.optimization:`. This is a schema migration landed as commit 0
before Phase 2 backend work.

**Metal profile: stays on tinygrad for this migration.** PyTorch MPS has
less predictable op support and numerics than CUDA; mixing MPS migration
with CUDA migration makes cause attribution hard. CUDA torch path stabilizes
first; Metal is re-evaluated in a separate benchmark later. CPU workers can
move to torch CPU later if needed.

**Optimizer state portability (tinygrad → torch): fresh optimizer.** When
loading a tinygrad checkpoint into the torch backend, model weights transfer
but optimizer state starts fresh. The cost of reconstructing Adam moments
accurately across backends exceeds the value of a few hundred warmup steps.
Resume logs must state this explicitly.

**GUI boundary: checkpoint-format based.** GUI does not import
`torch_network` directly. It consumes checkpoint files (and later ONNX) via
a backend-agnostic loader interface. This keeps the GUI working across
backends and hardware profiles even after tinygrad removal.

**ONNX boundary: export script depends on `torch_network` directly.** ONNX
export inherently traces a torch model; the export script must instantiate
`torch_network.TorchPolicyValueNet`. The checkpoint-format boundary applies
to *consumers* of the exported ONNX file, not to the export script itself.
Phase 1 already assumes this (it removes the duplicate torch model from
`export_onnx.py`).

**Saved-time reinvestment order: iterations first, then MCTS sweep.** After
Phase 4, first verify learning curves with the same MCTS budget and more /
faster iterations. Only once curves are stable, sweep `leaves_per_batch` and
simulation counts in Phase 5. Mixing backend migration with search-budget
changes makes effects inseparable.

## Still Open

- Stabilization criterion for "curves stable" before moving from iteration
  investment to MCTS sweep (Phase 5 trigger). Candidates: N consecutive
  arena promotions within expected range, or policy loss trend within X%
  of tinygrad baseline over K iterations. Pick a concrete signal before
  Phase 5.
