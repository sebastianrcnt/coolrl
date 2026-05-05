# Omok 15x15 Full CUDA HDD Run

이 문서는 보존 중인 15x15 Omok full CUDA HDD checkpoint를 찾기 위한 운영 기록입니다.

## 보존 대상

```text
/mnt/2tbhdd/coolrl-checkpoints/omok15_full_cuda_hdd
```

이 디렉터리는 checkpoint cleanup 대상이 아닙니다.

## Run 요약

- Config: `configs/omok15_full_cuda_hdd.yaml`
- Experiment name: `omok15_full_cuda_hdd`
- Board size: `15`
- Device: `CUDA`
- MCTS backend: `rust`
- Evaluator backend: `tensorrt`
- 마지막 checkpoint: `iter_0640.pt`, `latest.pt`
- 마지막 iteration: `640`
- Best checkpoint: `best.pt`
- Best iteration: `637`
- Best arena win rate: `0.5833`
- Total updates: `60768`
- Elapsed hours: `25.418`
- 보관 용량: 약 `23G`

## 주요 파일

- `latest.pt`: iteration 640 candidate checkpoint
- `best.pt`: iteration 637 best checkpoint
- `metrics.jsonl`: iteration별 metrics
- `runtime_progress.json`: 마지막 progress snapshot
- `replay.pkl`: replay buffer

## 관련 문서

- 15x15 board-size 운영 지침: [`board-size.md`](board-size.md)
- Omok training 구조: [`training.md`](training.md)
- 15x15 MCTS 메모리 사건 회고: [`../archive/omok/mcts-memory-incident.md`](../archive/omok/mcts-memory-incident.md)
