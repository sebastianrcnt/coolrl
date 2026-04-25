# Omok 문서

Omok 문서는 사용 목적별로 나뉩니다.

## 처음 보는 경우

1. [`getting-started.md`](getting-started.md): 설치 확인, smoke run, quick run.
2. [`play.md`](play.md): 학습한 모델을 GUI/TUI/Web GUI에서 플레이.
3. [`training.md`](training.md): self-play, config, checkpoint, metrics 구조 이해.

## 특정 작업을 하는 경우

- Board 크기를 바꾸거나 15x15를 돌리려면: [`board-size.md`](board-size.md)
- CUDA, Metal, CPU, TensorRT 선택이 필요하면: [`performance.md`](performance.md)

## 기술 노트와 회고

아래 문서는 현재 사용법보다는 배경 설명과 과거 디버깅 기록입니다.

- [`../omok_cuda_tuning.md`](../omok_cuda_tuning.md): CUDA self-play tuning과 evaluator benchmark 기록.
- [`../omok-dynamic-board-size.md`](../omok-dynamic-board-size.md): dynamic board-size migration 상세 설계.
- [`../omok-mcts-memory.md`](../omok-mcts-memory.md): 15x15 MCTS memory incident 회고.
- [`../omok-web-ios-memory.md`](../omok-web-ios-memory.md): Web/iOS memory 관련 노트.
