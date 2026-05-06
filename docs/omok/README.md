# Omok 문서

Omok 문서는 사용 목적별로 나뉩니다.

## 처음 보는 경우

1. [`getting-started.md`](getting-started.md): 설치 확인, smoke run, quick run.
2. [`play.md`](play.md): 학습한 모델을 GUI/TUI/Web GUI에서 플레이.
3. [`training.md`](training.md): self-play, config, checkpoint, metrics 구조 이해.

## 특정 작업을 하는 경우

- Board 크기를 바꾸거나 15x15를 돌리려면: [`board-size.md`](board-size.md)
- CUDA, Metal, CPU, TensorRT 선택이 필요하면: [`performance.md`](performance.md)
- 보존 중인 15x15 full CUDA HDD run을 확인하려면: [`omok15-full-cuda-hdd-run.md`](omok15-full-cuda-hdd-run.md)

## 기술 노트와 회고

아래 문서는 현재 사용법보다는 배경 설명과 과거 디버깅 기록입니다.

- [`../archive/omok/cuda-tuning.md`](../archive/omok/cuda-tuning.md): CUDA self-play tuning과 evaluator benchmark 기록.
- [`../archive/omok/dynamic-board-size.md`](../archive/omok/dynamic-board-size.md): dynamic board-size migration 상세 설계.
- [`../archive/omok/mcts-memory-incident.md`](../archive/omok/mcts-memory-incident.md): 15x15 MCTS memory incident 회고.
- [`../archive/omok/web-ios-memory.md`](../archive/omok/web-ios-memory.md): Web/iOS memory 관련 노트.
