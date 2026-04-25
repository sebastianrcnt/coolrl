# Omok 시작하기

이 문서는 Omok pipeline이 내 환경에서 돌아가는지 확인하려는 사람을 위한 최소 경로입니다.

## 1. Smoke run

먼저 가장 작은 config로 train loop, MCTS, checkpoint write가 모두 연결되어 있는지 확인합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml --device CPU
```

`configs/omok_smoke.yaml`는 의도적으로 작습니다.

- 1 iteration
- 1 self-play game
- move당 2 MCTS simulations
- 작은 network
- arena disabled

이 명령이 통과하면 기본 Python package path와 training entrypoint는 정상입니다.

## 2. 짧은 로컬 학습

Smoke run이 지나가면 quick config로 짧은 학습을 돌립니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --device CPU
```

Checkpoint 디렉토리에서 재개할 때는 `--resume`을 사용합니다.

```bash
uv run python -m coolrl.omok.train \
  --config configs/omok_quick.yaml \
  --resume checkpoints/omok_quick \
  --device CPU
```

## 3. 15x15 smoke run

15x15도 별도 패키지가 아니라 같은 `coolrl.omok` trainer를 사용합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok15_smoke.yaml --device CPU
```

짧은 15x15 run은 다음 preset을 사용합니다.

```bash
uv run python -m coolrl.omok.train --config configs/omok15_quick.yaml --device CPU
```

## 4. ONNX export

GUI나 Web GUI에서 모델을 쓰려면 checkpoint를 ONNX로 내보냅니다.

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
  --checkpoint checkpoints/omok_quick \
  --output exports/omok_quick.onnx
```

15x15 모델은 board 크기가 일치해야 합니다.

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
  --checkpoint checkpoints/omok15_quick \
  --output exports/omok15_quick.onnx
```

## 5. 다음에 읽을 문서

- 모델을 눈으로 확인하려면 [`play.md`](play.md)를 읽으세요.
- config를 키워서 제대로 학습시키려면 [`training.md`](training.md)를 읽으세요.
- GPU/Metal/CUDA 설정을 고르려면 [`performance.md`](performance.md)를 읽으세요.
