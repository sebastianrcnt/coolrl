# Omok PyTorch-Only Migration Plan

이 문서는 Omok의 남은 tinygrad training path를 PyTorch로 어떻게 대체할 것인지를 설명합니다. 목표는 두 개의 training backends을 지원하는 것이 아닙니다. 목표는 추천 경로에서 tinygrad를 삭제하고 PyTorch를 model definition, training, checkpointing, export의 단일 source of truth로 만드는 것입니다.

## Migration Policy

- `training.backend`를 추가하지 마세요.
- `TrainingBackend` abstraction을 추가하지 마세요.
- tinygrad와 torch training을 나란히 유지하지 마세요.
- Training state에 PyTorch `.pt` checkpoints를 사용하세요.
- 기존 tinygrad checkpoints를 legacy input artifacts로만 취급하세요.
- Tinygrad optimizer state는 이식성이 없으며 마이그레이션되지 않을 것입니다.
- MCTS interfaces를 변경하지 마세요. MCTS는 evaluator interface에만 의존해야 합니다.
- Torch-only path가 작동한 후에만 search settings을 retune하세요.

The implementation should prefer direct replacement over compatibility layers.
Compatibility code is allowed only where it is needed to load old tinygrad model
weights during the migration.

## Current State

| Area | Current Backend | Migration Target |
|---|---|---|
| self-play evaluator | PyTorch on CUDA profile, tinygrad by default | PyTorch only |
| arena evaluator | PyTorch on CUDA profile, tinygrad by default | PyTorch only |
| training model | tinygrad `PolicyValueNet` in `network.py` | PyTorch model |
| optimizer | tinygrad `AdamW` in `train.py` | `torch.optim.AdamW` |
| replay tensors | tinygrad tensors | torch tensors |
| checkpoints | tinygrad safetensors | torch `.pt` |
| ONNX export | duplicate PyTorch mirror | shared PyTorch model |
| C/Python MCTS | backend-neutral evaluator interface | unchanged |

The PyTorch evaluator already shows that the network can be represented
accurately:

| Device | Policy Max Abs Diff | Value Max Abs Diff |
|---|---:|---:|
| CPU, batch 7 | ~2.8e-9 | ~1.0e-7 |
| CUDA, batch 64 | ~4.0e-7 | ~1.1e-5 |
| CUDA, padded batch 93 | ~7.6e-7 | ~5.4e-6 |

The remaining risk is training semantics: BatchNorm state, optimizer
hyperparameters, checkpoint resume behavior, replay tensor conversion, and
full-loop learning behavior.

## Target Architecture

Final intended module ownership:

| Module | Responsibility |
|---|---|
| `torch_network.py` | canonical PyTorch `PolicyValueNet` and residual blocks |
| `torch_evaluator.py` | evaluator wrapper around `torch_network.py` |
| `train.py` | PyTorch training loop, optimizer step, checkpoint calls |
| `checkpoint.py` | torch `.pt` save/load plus legacy tinygrad weight import |
| `replay.py` | numpy replay storage plus torch batch sampling |
| `export_onnx.py` | ONNX export from `torch_network.py` |
| `network.py` | removed after legacy weight import no longer needs it |

Config shape should stay simple:

```yaml
optimization:
  batch_size: 256
  updates_per_iteration: 96
  learning_rate: 0.0005
  weight_decay: 0.0001

selfplay:
  evaluator_backend: torch
```

Do not move `optimization:` under a new `training:` section as part of this
migration. A config schema migration adds churn without helping the torch-only
goal.

`selfplay.evaluator_backend` may temporarily remain while old configs are
updated, but the final recommended configs should use PyTorch evaluators by
default and should not offer tinygrad as a normal option.

## Phase 1: Consolidate the PyTorch Network

Goal: one PyTorch network definition used by evaluator, training, parity tests,
and ONNX export.

Work:

- Create `torch_network.py`.
- Move `TorchPolicyValueNet`, `TorchResidualBlock`, and `TorchSEBlock` from
  `torch_evaluator.py` into `torch_network.py`.
- Update `torch_evaluator.py` to import the shared model.
- Update `export_onnx.py` to import the shared model instead of maintaining a
  duplicate torch implementation.
- Add a shared helper for loading tinygrad-shaped weights into the torch model.
- Keep parameter names compatible with existing tinygrad checkpoints where
  practical:
  - `stem_conv.weight`
  - `tower.N.conv1.weight`
  - `tower.N.se.fc1.weight`
  - `policy_fc.weight`
  - `value_fc.weight`

Validation command:

```bash
uv run --extra omok python scripts/check_omok_torch_parity.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --batches 7,64,93
```

The parity script should cover:

- eval-mode forward parity on fixed batches
- train-mode forward parity on one fixed batch
- BatchNorm `running_mean` / `running_var` shape and dtype after loading a
  tinygrad checkpoint
- ONNX export still using the shared model

Acceptance criteria:

- PyTorch evaluator parity remains within documented tolerance.
- Train-mode parity has a documented tolerance.
- ONNX export still works.
- No training behavior change yet.

## Phase 2: Replace Tinygrad Training with PyTorch

Goal: make `train.py` use PyTorch directly instead of tinygrad. This is a
replacement, not a second backend.

Work:

- Replace the tinygrad model construction in `train.py` with
  `torch_network.PolicyValueNet` or the chosen class name from Phase 1.
- Replace tinygrad `AdamW` with `torch.optim.AdamW`.
- Port the exact optimizer hyperparameters from the current tinygrad call site:
  learning rate, betas, eps, weight decay, and gradient clipping if present.
- Replace replay batch tensor creation with torch tensors.
- Keep replay storage as numpy/deque; only sampled batches should become torch
  tensors.
- Use `model.train()` during optimizer updates.
- Use `model.eval()` for self-play and arena evaluation.
- Preserve the existing loss math:

```text
policy_loss = -(target_policy * log_softmax(logits)).sum(axis=1).mean()
value_loss = mean((value - target_value) ** 2)
loss = policy_weight * policy_loss + value_weight * value_loss
```

- Preserve these existing behaviors exactly:
  - `recency_temperature`
  - symmetry augmentation
  - policy target shape
  - value target shape
  - `value_discount`
  - promotion / arena gate logic
  - MCTS evaluator interface

Replay API target:

```python
ReplayBuffer.sample_batch_numpy(...)
ReplayBuffer.sample_batch_torch(...)
```

`ReplayBuffer.sample_batch(...)` may either become the torch path directly or be
removed after call sites are updated. Do not keep a tinygrad sampling path unless
it is needed by a temporary legacy import script.

Explicitly out of scope for this phase:

- `torch.compile`
- AMP / autocast / GradScaler
- `channels_last`
- `torch.backends.cudnn.benchmark = True`
- MCTS retuning
- optimizer state conversion from tinygrad

Acceptance criteria:

- Quick config runs from scratch with torch training.
- Full CUDA config starts training with torch tensors and torch optimizer.
- Metrics still include phase timings and loss fields.
- Training logs clearly identify PyTorch as the training implementation.

## Phase 3: Torch Checkpoints and Legacy Weight Import

Goal: save and resume torch training with `.pt`, while allowing old tinygrad
model weights to seed a torch run.

Torch checkpoint format:

```python
{
    "checkpoint_format": "coolrl.omok.torch.v1",
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "metadata": {
        "torch_version": "...",
        "iteration": ...,
        "config": ...,
    },
}
```

Rules:

- New training checkpoints use `.pt`.
- `latest.pt` and `best.pt` are the normal runtime checkpoints.
- Torch `.pt` checkpoints resume model, optimizer, scheduler, iteration, and
  metadata.
- Existing tinygrad `.safetensors` checkpoints may be loaded only as model
  weights.
- Loading tinygrad weights always creates a fresh torch optimizer.
- Logs must explicitly say when optimizer state was not restored.
- Do not implement tinygrad optimizer conversion.
- Do not support saving new tinygrad checkpoints.

Optional helper command:

```bash
uv run --extra omok python -m coolrl.omok.convert_checkpoint \
  --input checkpoints/omok_full_cuda/latest.safetensors \
  --output checkpoints/omok_full_cuda_torch/latest.pt
```

The helper is useful but not required if `train.py` can directly initialize
from legacy weights.

Acceptance criteria:

- Save torch `latest.pt`.
- Save torch `best.pt`.
- Resume a torch run from `latest.pt`.
- Initialize a torch run from an old tinygrad model checkpoint with fresh
  optimizer state.

## Phase 4: Remove Tinygrad from the Normal Path

Goal: delete or isolate tinygrad code after torch training and checkpointing are
working.

Work:

- Update recommended configs to use PyTorch evaluators.
- Remove tinygrad evaluator selection from recommended configs.
- Remove tinygrad training imports from `train.py`.
- Remove tinygrad replay tensor creation.
- Remove tinygrad optimizer checkpoint save/load.
- Remove duplicate tinygrad-specific docs from the recommended setup path.
- Keep only the minimal legacy code required to import old model weights.
- If legacy import still depends on `network.py`, isolate that dependency behind
  one clearly named function or script.

Metal policy:

- Do not keep tinygrad only for Metal.
- Try torch MPS if it is straightforward.
- If torch MPS is not stable, mark Metal training unsupported until separately
  fixed.
- CPU training may use torch CPU if needed.

Acceptance criteria:

- The normal training path imports no tinygrad modules.
- Recommended CUDA config uses torch evaluator and torch training.
- Existing MCTS code does not need API changes.
- Docs describe PyTorch as the trainer, not one backend among several.

## Phase 5: Benchmark and Then Retune

Goal: measure the torch-only baseline first, then decide where to reinvest saved
time.

Baseline benchmark:

```bash
rm -rf checkpoints/omok_full_cuda_torch
uv run --extra omok python -m coolrl.omok.train \
  --config configs/omok_full_cuda.yaml \
  --max-iterations 4
```

Track:

| Metric | Why |
|---|---|
| `selfplay_seconds` | evaluator cost |
| `train_seconds` | training cost |
| `train_optimizer_seconds` | optimizer bottleneck check |
| `train_samples_per_second` | throughput independent of wall-time noise |
| `arena_seconds` | promotion gate cost |
| `duration_seconds` | full-loop cost |
| `policy_loss` / `value_loss` | learning sanity |
| arena win rates | collapse detection |
| peak GPU memory if available | headroom for later optimizations |

Do not change search settings during the first torch-only benchmark.

After baseline is stable, layer torch-specific optimizations one at a time:

1. `torch.backends.cudnn.benchmark = True`
2. `channels_last`
3. `torch.compile`
4. AMP / autocast

Only after those are measured, sweep MCTS/search settings:

```text
leaves_per_batch: 8, 16, 32, 64
simulation_schedule: current, then higher final simulations if iteration time falls
arena.games: current, then higher if arena becomes cheap
```

Saved-time reinvestment order:

1. keep the same MCTS budget and verify more/faster iterations behave normally
2. enable torch optimizations incrementally
3. retune `leaves_per_batch`
4. consider higher simulations or larger arena gates

## Main Risks

| Risk | Mitigation |
|---|---|
| BatchNorm behavior changes | parity checks in eval and train mode; preserve running stats |
| AdamW drift | port tinygrad hyperparameters explicitly; do not rely on torch defaults |
| optimizer state is not portable | tinygrad checkpoints load weights only; fresh optimizer is expected |
| checkpoint confusion | use `.pt` for all new training checkpoints |
| replay target shape drift | keep replay storage unchanged and convert only sampled batches |
| learning behavior changes | compare losses, arena gates, and fixed-checkpoint matches |
| Metal regression | torch MPS or unsupported; do not preserve tinygrad just for Metal |
| premature optimization | land fp32 torch baseline before compile/AMP/channels_last |
| search retuning hides backend bugs | retune only after baseline learning behavior is sane |

## Recommended Commit Sequence

1. `Extract shared PyTorch Omok network`
   - add `torch_network.py`
   - update `torch_evaluator.py`
   - update `export_onnx.py`
   - add parity script

2. `Replace training model and optimizer with torch`
   - update `train.py`
   - add torch replay batch sampling
   - preserve current loss math and optimizer hyperparameters

3. `Add torch checkpoint save and resume`
   - write `.pt` checkpoints
   - resume model and optimizer from `.pt`
   - log checkpoint format and torch version

4. `Add legacy tinygrad weight import`
   - load old model weights into torch model
   - always start fresh torch optimizer
   - keep this path isolated

5. `Remove tinygrad from normal training path`
   - remove tinygrad imports from trainer
   - remove tinygrad replay tensors
   - update recommended configs and docs

6. `Benchmark torch-only CUDA baseline`
   - run fixed iteration benchmark
   - document timings in `docs/omok_cuda_tuning.md`
   - do not retune search yet

7. `Enable torch optimizations and retune search`
   - measure each optimization independently
   - sweep `leaves_per_batch`
   - update CUDA config only after measured results

## Decisions

**PyTorch is the only training backend.** Do not introduce backend selection or
side-by-side trainer abstractions.

**Training checkpoints use `.pt`.** `.safetensors` is legacy input or a possible
future weights-only export artifact, not the new training format.

**Tinygrad optimizer state is discarded.** Old checkpoints can seed model
weights, but torch starts with a fresh optimizer.

**Metal does not justify keeping tinygrad.** Use torch MPS if viable; otherwise
mark Metal training unsupported until it is fixed separately.

**MCTS stays backend-neutral.** The evaluator interface should be enough; do not
change MCTS as part of this migration.

**Retuning happens after migration.** First get a stable torch-only fp32
baseline, then optimize and retune.

## Still Open

- Define the exact "baseline learning is sane" signal before MCTS retuning.
  A practical default is: no loss explosion, arena behavior not obviously
  collapsed, and at least one fixed-checkpoint comparison close to the previous
  baseline over the same number of iterations.
