# Omok board 크기

Omok은 9x9와 15x15를 별도 패키지로 나누지 않습니다. `coolrl.omok` 하나의 codepath가 `rules.board_size`에 따라 정사각형 board를 처리합니다.

## 빠른 사용법

9x9 quick run:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --device CPU
```

15x15 quick run:

```bash
uv run python -m coolrl.omok.train --config configs/omok15_quick.yaml --device CPU
```

15x15 full CUDA run:

```bash
uv run python -m coolrl.omok.train --config configs/omok15_full_cuda.yaml
```

## Config에서 board 크기 바꾸기

Board 크기는 config의 `rules.board_size`로 정합니다.

```yaml
rules:
  board_size: 13

checkpoint:
  directory: checkpoints/omok13_quick
```

공식 preset이 아니라 실험용이라면 기존 quick config를 복사해서 `board_size`와 checkpoint directory만 바꾸는 방식으로 시작하세요.

## Checkpoint는 board 크기별로 분리하기

Board 크기가 바뀌면 policy head 크기도 바뀝니다.

```text
9x9  -> action_size = 81
15x15 -> action_size = 225
```

따라서 9x9 checkpoint를 15x15 network에 로드하거나, 반대로 15x15 checkpoint를 9x9 network에 로드하면 실패합니다. Checkpoint directory는 board 크기별로 분리하세요.

## MCTS backend 지원 범위

Python game state는 `board_size >= 5`인 정사각형 board를 받습니다.

Native C/Rust backends는 현재 5에서 19까지의 크기를 지원합니다. 따라서 9x9, 13x13, 15x15 같은 크기는 별도 native fork 없이 같은 구현을 사용할 수 있습니다.

하나의 MCTS batch에는 같은 board 크기의 states만 섞어야 합니다. 9x9와 15x15를 같은 batch에 넣으면 feature tensor와 policy shape이 다르므로 조기에 실패합니다.

## GUI와 Web GUI

Pygame GUI는 `--board-size`를 받습니다.

```bash
uv run python -m coolrl.omok.gui \
  --model exports/omok15_quick.onnx \
  --board-size 15
```

Web GUI는 board-size selector를 제공합니다. ONNX 모델을 로드하기 전에 selector를 모델의 board 크기와 맞추세요.

GUI와 Web GUI 모두 ONNX policy output 길이가 선택한 board 크기와 맞는지 확인합니다.

## 더 깊은 설계 설명

Dynamic board-size migration의 세부 구현 배경은 [`../omok-dynamic-board-size.md`](../omok-dynamic-board-size.md)를 참고하세요.

15x15에서 MCTS memory behavior가 왜 중요했는지는 [`../omok-mcts-memory.md`](../omok-mcts-memory.md)에 정리되어 있습니다.
