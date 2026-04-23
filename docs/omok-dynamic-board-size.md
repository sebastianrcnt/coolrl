# Omok 동적 Board 크기

`coolrl.omok`은 지원되는 정사각형 board 크기를 위한 단일 Omok 구현입니다. 이전에 중복된 `coolrl.omok15` 패키지는 폐기되었으며, 15x15 runs는 이제 9x9와 동일한 trainer, feature encoder, network, checkpointing, plotting, web UI, GUI, Python MCTS, C MCTS, Rust MCTS 경로를 사용합니다.

## 설계

Board 크기는 `rules.board_size`에서 나오며 각 runtime surface를 통해 전달됩니다:

```text
action = row * board_size + col
row, col = divmod(action, board_size)
action_size = board_size * board_size
feature_stride = 4 * action_size
```

Python game state는 `board_size >= 5`인 정사각형 boards를 수락합니다. Native C 및 Rust backends는 현재 5에서 19까지의 크기를 수락하므로, 다른 compile-time fork 없이 9x9, 13x13, 15x15를 포함합니다.

하나의 MCTS `search_batch` 호출은 단일 board 크기를 사용해야 합니다. 한 배치에서 9x9 및 15x15 상태를 섞으면 policy 및 feature tensor 형태가 다르기 때문에 명확한 오류로 조기에 실패합니다.

## MCTS Backends

세 가지 backends 모두 동일한 Python-facing search 계약을 공유합니다:

- Python MCTS는 `GameState.action_size`에서 `action_size`를 파생합니다.
- C MCTS는 각 tree를 생성할 때 `board_size`를 받고 runtime `action_size`에서 child, feature, visit-count storage를 할당합니다.
- Rust MCTS는 동일한 runtime-sized tree API를 반영하고 FFI getters를 통해 board/action metadata를 노출합니다.

C 및 Rust wrappers는 다음을 검증합니다:

- 배치의 모든 상태가 동일한 board 크기를 가짐;
- 재사용된 roots가 들어오는 상태 board 크기와 일치함;
- evaluator priors가 `[batch, board_size * board_size]` 형태를 가짐.

9x9에서 더 큰 boards로 확장할 때 native MCTS node lifetime과 조밀한 child storage가 특별한 주의가 필요한 이유를 드러낸 15x15 메모리 사건은 `docs/omok-mcts-memory.md`를 참조하세요.

## Configs

기본 9x9 presets는 유지됩니다:

- `configs/omok_smoke.yaml`
- `configs/omok_quick.yaml`
- `configs/omok_full_cuda.yaml`
- `configs/omok_full_metal.yaml`

15x15 presets는 이제 일반적인 `coolrl.omok` configs입니다:

- `configs/omok15_smoke.yaml`
- `configs/omok15_quick.yaml`
- `configs/omok15_full_cuda.yaml`

다음과 같이 실행합니다:

```bash
uv run python -m coolrl.omok.train --config configs/omok15_smoke.yaml --device CPU
uv run python -m coolrl.omok.train --config configs/omok15_full_cuda.yaml
```

Checkpoint 디렉토리는 board 크기별로 분리된 상태로 유지되어야 합니다. 9x9 checkpoint는 길이 81의 policy head를 가지고, 15x15 checkpoint는 길이 225를 가지므로, 다른 board 크기의 network에 checkpoint를 로드하면 실패할 것입니다.

## 도구

ONNX export는 `cfg.rules.board_size`에서 dummy input을 빌드합니다. Pygame GUI는 `--board-size`를 수락하고, browser UI는 board-size selector를 가집니다. 두 인터페이스 모두 로드된 모델의 policy output 길이가 선택된 board 크기와 일치하는지 검증합니다.

Training metrics는 이제 `board_size`를 기록합니다. `omok-plot`은 그 값, 또는 필요한 경우 checkpoint sidecar config를 사용하여 올바른 uniform policy entropy reference를 그립니다:

```python
uniform_policy_entropy = np.log(board_size * board_size)
```

## 다른 크기 추가

13x13과 같은 표준 정사각형 크기의 경우, 다음과 같은 config를 추가합니다:

```yaml
rules:
  board_size: 13

checkpoint:
  directory: checkpoints/omok13_quick
```

크기가 native backend limits 내에 머무르는 한 새로운 패키지나 native backend fork가 필요하지 않습니다. 임시 실험이 아니라 공식적으로 유지되는 preset이 되면 새로운 크기에 대한 parity coverage를 추가하세요.
