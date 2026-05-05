# Omok RL

`coolrl.omok`은 Omok agents를 self-play로 학습하고, GUI/TUI/Web UI에서 플레이해볼 수 있는 패키지입니다.

이 README는 Omok 문서의 출발점입니다. 자세한 설명은 목적별 문서로 나누었습니다.

## 바로 실행해보기

가장 작은 smoke run으로 전체 파이프라인이 동작하는지 확인합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device CPU
```

짧은 로컬 학습을 돌립니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --device CPU
```

15x15 smoke run도 같은 trainer를 사용합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok15_smoke.yaml --device CPU
```

## 어디부터 읽을까?

- 처음 실행하거나 환경을 확인하려면: [`docs/omok/getting-started.md`](../../../docs/omok/getting-started.md)
- 학습 config, self-play, checkpoint, metrics가 궁금하면: [`docs/omok/training.md`](../../../docs/omok/training.md)
- GUI, TUI, Web GUI로 플레이하고 싶으면: [`docs/omok/play.md`](../../../docs/omok/play.md)
- 9x9, 13x13, 15x15 같은 board 크기를 다루려면: [`docs/omok/board-size.md`](../../../docs/omok/board-size.md)
- CUDA/Metal/CPU 성능 설정을 고르려면: [`docs/omok/performance.md`](../../../docs/omok/performance.md)

## 코드 위치

- Trainer: `src/coolrl/omok/train.py`
- Config schema: `src/coolrl/omok/config.py`
- Pygame GUI: `src/coolrl/omok/gui.py`
- Textual TUI: `src/coolrl/omok/tui/`
- Browser GUI: `src/coolrl/omok/web/`
- C MCTS backend: `src/coolrl/omok/cmcts/`
- Rust MCTS backend: `src/coolrl/omok/rmcts/`

## 더 깊은 기술 노트

아래 문서들은 구현 배경, 과거 성능 측정, 장애 회고에 가깝습니다. 처음 읽는 문서라기보다는 디버깅하거나 설계를 따라갈 때 참고하세요.

- [`docs/archive/omok/cuda-tuning.md`](../../../docs/archive/omok/cuda-tuning.md)
- [`docs/archive/omok/dynamic-board-size.md`](../../../docs/archive/omok/dynamic-board-size.md)
- [`docs/archive/omok/mcts-memory-incident.md`](../../../docs/archive/omok/mcts-memory-incident.md)
- [`docs/archive/omok/web-ios-memory.md`](../../../docs/archive/omok/web-ios-memory.md)
