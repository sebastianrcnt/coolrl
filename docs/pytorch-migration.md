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

최종 의도된 module ownership:

| Module | Responsibility |
|---|---|
| `torch_network.py` | canonical PyTorch `PolicyValueNet` 및 residual blocks |
| `torch_evaluator.py` | `torch_network.py` 주변의 evaluator wrapper |
| `train.py` | PyTorch training loop, optimizer step, checkpoint calls |
| `checkpoint.py` | torch `.pt` save/load 및 legacy tinygrad weight import |
| `replay.py` | numpy replay storage 및 torch batch sampling |
| `export_onnx.py` | `torch_network.py`에서 ONNX export |
| `network.py` | legacy weight import가 더 이상 필요 없는 후 제거됨 |

Config shape은 simple하게 유지해야 합니다:

```yaml
optimization:
  batch_size: 256
  updates_per_iteration: 96
  learning_rate: 0.0005
  weight_decay: 0.0001

selfplay:
  evaluator_backend: torch
```

이 migration의 일부로 `optimization:`을 새로운 `training:` section 아래로 이동하지 마세요. config schema migration은 churn을 추가하며 torch-only goal에 도움이 되지 않습니다.

`selfplay.evaluator_backend`는 old configs이 updated되는 동안 임시로 유지될 수 있지만, 최종 recommended configs는 기본적으로 PyTorch evaluators를 사용해야 하고 tinygrad를 normal option로 제공해서는 안 됩니다.

## Phase 1: PyTorch Network 통합

Goal: evaluator, training, parity tests, ONNX export에서 사용하는 하나의 PyTorch network definition.

Work:

- `torch_network.py`를 만듭니다.
- `TorchPolicyValueNet`, `TorchResidualBlock`, `TorchSEBlock`을 `torch_evaluator.py`에서 `torch_network.py`로 이동합니다.
- `torch_evaluator.py`를 shared model을 import하도록 업데이트합니다.
- `export_onnx.py`를 duplicate torch implementation을 maintain하는 대신 shared model을 import하도록 업데이트합니다.
- tinygrad-shaped weights를 torch model로 loading하기 위한 shared helper를 추가합니다.
- practical한 경우 existing tinygrad checkpoints와 compatible한 parameter names를 유지합니다:
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

parity script는 다음을 cover해야 합니다:

- fixed batches에서 eval-mode forward parity
- one fixed batch에서 train-mode forward parity
- tinygrad checkpoint loading 후 BatchNorm `running_mean` / `running_var` shape 및 dtype
- ONNX export가 여전히 shared model을 사용

Acceptance criteria:

- PyTorch evaluator parity는 documented tolerance 내에 유지됩니다.
- Train-mode parity는 documented tolerance를 가집니다.
- ONNX export가 여전히 작동합니다.
- 아직 training behavior 변화가 없습니다.

## Phase 2: Tinygrad Training을 PyTorch로 대체

Goal: `train.py`를 tinygrad 대신 PyTorch를 직접 사용하도록 만듭니다. 이는 replacement이지, second backend가 아닙니다.

Work:

- `train.py`의 tinygrad model construction을 `torch_network.PolicyValueNet` 또는 Phase 1에서 선택한 class name으로 대체합니다.
- tinygrad `AdamW`를 `torch.optim.AdamW`로 대체합니다.
- 현재 tinygrad call site에서 정확한 optimizer hyperparameters를 port합니다: learning rate, betas, eps, weight decay, gradient clipping if present.
- replay batch tensor creation을 torch tensors로 대체합니다.
- replay storage를 numpy/deque로 유지합니다. sampled batches만 torch tensors가 되어야 합니다.
- optimizer updates 중에 `model.train()`을 사용합니다.
- self-play 및 arena evaluation을 위해 `model.eval()`을 사용합니다.
- 기존 loss math를 preserve합니다:

```text
policy_loss = -(target_policy * log_softmax(logits)).sum(axis=1).mean()
value_loss = mean((value - target_value) ** 2)
loss = policy_weight * policy_loss + value_weight * value_loss
```

- 이러한 기존 behaviors를 정확히 preserve합니다:
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

`ReplayBuffer.sample_batch(...)`는 torch path로 직접 become하거나 call sites이 updated된 후 제거될 수 있습니다. temporary legacy import script에 필요하지 않은 한 tinygrad sampling path를 유지하지 마세요.

Explicitly out of scope for this phase:

- `torch.compile`
- AMP / autocast / GradScaler
- `channels_last`
- `torch.backends.cudnn.benchmark = True`
- MCTS retuning
- tinygrad에서 optimizer state conversion

Acceptance criteria:

- torch training을 사용하여 scratch에서 quick config runs.
- Full CUDA config는 torch tensors 및 torch optimizer로 training을 시작합니다.
- Metrics는 여전히 phase timings 및 loss fields를 포함합니다.
- Training logs는 명확하게 PyTorch를 training implementation으로 identify합니다.

## Phase 3: Torch Checkpoints 및 Legacy Weight Import

Goal: `.pt`로 torch training을 save하고 resume하면서, old tinygrad model weights가 torch run을 seed하도록 허용합니다.

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

- New training checkpoints는 `.pt`를 사용합니다.
- `latest.pt`와 `best.pt`는 normal runtime checkpoints입니다.
- Torch `.pt` checkpoints는 model, optimizer, scheduler, iteration, metadata를 resume합니다.
- Existing tinygrad `.safetensors` checkpoints는 model weights로만 loaded될 수 있습니다.
- tinygrad weights을 loading하면 항상 fresh torch optimizer을 생성합니다.
- Logs는 optimizer state가 restored되지 않았을 때를 명시적으로 말해야 합니다.
- tinygrad optimizer conversion을 구현하지 마세요.
- 새로운 tinygrad checkpoints을 저장하는 것을 지원하지 마세요.

Optional helper command:

```bash
uv run --extra omok python -m coolrl.omok.convert_checkpoint \
  --input checkpoints/omok_full_cuda/latest.safetensors \
  --output checkpoints/omok_full_cuda_torch/latest.pt
```

helper는 유용하지만, `train.py`가 legacy weights에서 직접 initialize할 수 있다면 필수는 아닙니다.

Acceptance criteria:

- torch `latest.pt`를 save합니다.
- torch `best.pt`를 save합니다.
- `latest.pt`에서 torch run을 resume합니다.
- fresh optimizer state로 old tinygrad model checkpoint에서 torch run을 initialize합니다.

## Phase 4: Normal Path에서 Tinygrad 제거

Goal: torch training 및 checkpointing이 작동한 후 tinygrad code를 delete하거나 isolate합니다.

Work:

- recommended configs를 PyTorch evaluators를 사용하도록 업데이트합니다.
- recommended configs에서 tinygrad evaluator selection을 제거합니다.
- `train.py`에서 tinygrad training imports를 제거합니다.
- tinygrad replay tensor creation을 제거합니다.
- tinygrad optimizer checkpoint save/load을 제거합니다.
- recommended setup path에서 duplicate tinygrad-specific docs를 제거합니다.
- old model weights를 import하는 데 필요한 minimal legacy code만 유지합니다.
- legacy import가 여전히 `network.py`에 depends하면, one clearly named function 또는 script 뒤에 그 dependency를 isolate합니다.

Metal policy:

- Metal만을 위해 tinygrad를 유지하지 마세요.
- straightforward하면 torch MPS를 시도하세요.
- torch MPS가 stable하지 않으면, Metal training을 separately fixed될 때까지 unsupported로 mark하세요.
- CPU training은 필요하면 torch CPU를 사용할 수 있습니다.

Acceptance criteria:

- normal training path는 tinygrad modules를 import하지 않습니다.
- Recommended CUDA config는 torch evaluator 및 torch training을 사용합니다.
- Existing MCTS code는 API changes가 필요하지 않습니다.
- Docs는 PyTorch를 trainer로 describe하며, 여러 backends 중 하나가 아닙니다.

## Phase 5: Benchmark 및 Then Retune

Goal: torch-only baseline을 먼저 measure한 다음, 저장된 시간을 재투자할 위치를 결정합니다.

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
| `train_samples_per_second` | wall-time noise와 independent한 throughput |
| `arena_seconds` | promotion gate cost |
| `duration_seconds` | full-loop cost |
| `policy_loss` / `value_loss` | learning sanity |
| arena win rates | collapse detection |
| peak GPU memory if available | later optimizations를 위한 headroom |

첫 번째 torch-only benchmark 중에 search settings을 변경하지 마세요.

baseline이 stable해진 후, torch-specific optimizations을 한 번에 하나씩 layer합니다:

1. `torch.backends.cudnn.benchmark = True`
2. `channels_last`
3. `torch.compile`
4. AMP / autocast

이들이 measured 된 후에만, MCTS/search settings을 sweep합니다:

```text
leaves_per_batch: 8, 16, 32, 64
simulation_schedule: current, then higher final simulations if iteration time falls
arena.games: current, then higher if arena becomes cheap
```

Saved-time reinvestment order:

1. 같은 MCTS budget을 유지하고 more/faster iterations이 정상적으로 behave하는지 verify합니다.
2. torch optimizations를 incrementally enable합니다.
3. `leaves_per_batch`를 retune합니다.
4. higher simulations 또는 larger arena gates를 고려합니다.

## Main Risks

| Risk | Mitigation |
|---|---|
| BatchNorm behavior changes | eval 및 train mode의 parity checks; running stats를 preserve |
| AdamW drift | tinygrad hyperparameters를 명시적으로 port합니다; torch defaults에 rely하지 마세요 |
| optimizer state is not portable | tinygrad checkpoints는 weights만 load하고; fresh optimizer은 expected입니다 |
| checkpoint confusion | 모든 새로운 training checkpoints에 `.pt`를 사용하세요 |
| replay target shape drift | replay storage를 unchanged로 유지하고 sampled batches만 convert하세요 |
| learning behavior changes | losses, arena gates, fixed-checkpoint matches를 compare하세요 |
| Metal regression | torch MPS 또는 unsupported; Metal을 위해서만 tinygrad를 preserve하지 마세요 |
| premature optimization | compile/AMP/channels_last 전에 fp32 torch baseline을 land하세요 |
| search retuning hides backend bugs | baseline learning behavior이 sane인 후에만 retune하세요 |

## Recommended Commit Sequence

1. `Extract shared PyTorch Omok network`
   - `torch_network.py`를 추가합니다
   - `torch_evaluator.py`를 업데이트합니다
   - `export_onnx.py`를 업데이트합니다
   - parity script를 추가합니다

2. `Replace training model and optimizer with torch`
   - `train.py`를 업데이트합니다
   - torch replay batch sampling을 추가합니다
   - current loss math 및 optimizer hyperparameters를 preserve합니다

3. `Add torch checkpoint save and resume`
   - `.pt` checkpoints를 작성합니다
   - `.pt`에서 model 및 optimizer를 resume합니다
   - checkpoint format 및 torch version을 log합니다

4. `Add legacy tinygrad weight import`
   - old model weights를 torch model로 load합니다
   - 항상 fresh torch optimizer을 시작합니다
   - 이 path를 isolated 상태로 유지합니다

5. `Remove tinygrad from normal training path`
   - trainer에서 tinygrad imports를 제거합니다
   - tinygrad replay tensors를 제거합니다
   - recommended configs 및 docs를 업데이트합니다

6. `Benchmark torch-only CUDA baseline`
   - fixed iteration benchmark를 실행합니다
   - `docs/omok_cuda_tuning.md`에서 timings을 document합니다
   - search를 아직 retune하지 마세요

7. `Enable torch optimizations and retune search`
   - 각 optimization을 independently measure합니다
   - `leaves_per_batch`를 sweep합니다
   - measured results 후에만 CUDA config을 업데이트합니다

## Decisions

**PyTorch는 유일한 training backend입니다.** backend selection이나 side-by-side trainer abstractions을 introduce하지 마세요.

**Training checkpoints는 `.pt`를 사용합니다.** `.safetensors`는 legacy input이거나 possible future weights-only export artifact이며, new training format이 아닙니다.

**Tinygrad optimizer state는 discarded됩니다.** Old checkpoints는 model weights를 seed할 수 있지만, torch은 fresh optimizer로 시작합니다.

**Metal은 tinygrad를 유지할 정당성이 없습니다.** viable하면 torch MPS를 사용하세요. 아니면 Metal training을 separately fixed될 때까지 unsupported로 mark하세요.

**MCTS는 backend-neutral로 유지됩니다.** evaluator interface로 충분해야 합니다. 이 migration의 일부로 MCTS를 변경하지 마세요.

**Retuning은 migration 후에 발생합니다.** 먼저 stable torch-only fp32 baseline을 얻으세요. 그런 다음 optimize하고 retune하세요.

## Still Open

- MCTS retuning 전에 정확한 "baseline learning is sane" signal을 정의하세요.
  practical default는: no loss explosion, arena behavior not obviously collapsed, 같은 number of iterations에서 previous baseline에 close한 at least one fixed-checkpoint comparison입니다.
