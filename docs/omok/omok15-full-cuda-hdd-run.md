# Omok 15x15 Full CUDA HDD Run

이 문서는 보존 중인 15x15 Omok full CUDA HDD checkpoint를 찾기 위한 운영 기록입니다.

## 보존 대상

```text
checkpoints/omok15_full_cuda_hdd
```

이 디렉터리는 checkpoint cleanup 대상이 아닙니다. 로컬에서 대용량 디스크를 사용할 때는 이 경로를 symlink로 연결합니다.

```bash
mkdir -p checkpoints /path/to/large-disk/coolrl-checkpoints
ln -s /path/to/large-disk/coolrl-checkpoints/omok15_full_cuda_hdd \
  checkpoints/omok15_full_cuda_hdd
```

Git에는 실제 디스크 경로를 기록하지 않습니다.

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

## 로컬 운영 원칙

Config와 문서에서는 portable path인 `checkpoints/omok15_full_cuda_hdd`만 사용합니다. 각 머신의 HDD, SSD, NVMe 경로는 git에 넣지 않고 로컬 symlink로만 관리합니다.

## 관련 문서

- 15x15 board-size 운영 지침: [`board-size.md`](board-size.md)
- Omok training 구조: [`training.md`](training.md)
- 15x15 MCTS 메모리 사건 회고: [`../archive/omok/mcts-memory-incident.md`](../archive/omok/mcts-memory-incident.md)
