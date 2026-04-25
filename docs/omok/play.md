# Omok 플레이하기

이 문서는 학습한 Omok 모델을 사람이 보거나 플레이하는 방법을 정리합니다.

## Pygame GUI

ONNX 모델을 Pygame GUI에서 엽니다.

```bash
uv run python -m coolrl.omok.gui --model exports/omok_quick.onnx
```

15x15 모델은 board 크기를 명시합니다.

```bash
uv run python -m coolrl.omok.gui \
  --model exports/omok15_quick.onnx \
  --board-size 15
```

주요 키:

| Key | Meaning |
|---|---|
| Left click | 돌을 놓음 |
| `R` | 게임 초기화 |
| `S` | 인간 side 교환 |
| `M` | AI move 강제 |
| `O` | 현재 seed에서 deterministic opening 적용 |
| `[` / `]` | opening seed 감소/증가 |
| `Esc` | 종료 |

주요 옵션:

| Flag | Default | Description |
|---|---:|---|
| `--model FILE.onnx` | none | ONNX model 파일. 생략하면 2-player/UI testing 용도 |
| `--board-size N` | `9` | 모델 policy output 길이와 일치해야 함 |
| `--device auto|cpu|cuda|coreml` | `auto` | ONNX Runtime execution provider |
| `--simulations N` | `64` | AI move당 MCTS simulations |
| `--human-color black|white` | `white` | 인간이 플레이하는 색 |
| `--seed N` | `0` | opening seed |

## Textual TUI

Terminal에서 ONNX 모델을 관전하거나 서로 붙일 수 있습니다.

```bash
uv run --extra omok-tui python -m coolrl.omok.tui \
  --model exports/omok_quick.onnx
```

두 모델을 서로 붙입니다.

```bash
uv run --extra omok-tui python -m coolrl.omok.tui \
  --black-model exports/black.onnx \
  --white-model exports/white.onnx \
  --board-size 15
```

CUDA ONNX Runtime을 쓰려면 CUDA extra를 사용합니다.

```bash
uv run --extra omok-tui-cuda python -m coolrl.omok.tui \
  --model exports/omok_quick.onnx \
  --device cuda
```

TensorRT TUI path는 별도 extra가 필요합니다.

```bash
uv run --extra omok-tui-tensorrt python -m coolrl.omok.tui \
  --model exports/omok_quick.onnx \
  --device tensorrt
```

계속 새 게임을 돌리려면 `--infinite`를 사용합니다.

```bash
uv run --extra omok-tui python -m coolrl.omok.tui \
  --model exports/omok_quick.onnx \
  --infinite
```

주요 옵션:

| Flag | Default | Description |
|---|---:|---|
| `--model FILE.onnx` | none | 양쪽 모두에 사용되는 단일 ONNX model |
| `--black-model FILE.onnx` / `--white-model FILE.onnx` | none | 색상별 ONNX model |
| `--board-size 9|15` | `9` | 모델 policy output과 일치해야 함 |
| `--device auto|cpu|cuda|tensorrt|coreml` | `auto` | ONNX Runtime execution provider |
| `--simulations N` | `256` | move당 MCTS simulations |
| `--move-delay SEC` | `0.05` | 표시된 moves 사이 delay |
| `--infinite` | off | 게임이 끝날 때마다 새 seeded game 시작 |
| `--debug-lines N` | `1000` | debug console scrollback |

## Browser Web GUI

Browser GUI는 ONNX Runtime Web(WASM)을 통해 모델을 client-side에서 실행합니다.

먼저 checkpoint를 ONNX로 내보냅니다.

```bash
uv run --with torch --with onnx python -m coolrl.omok.export_onnx \
  --checkpoint checkpoints/omokai_converted/best \
  --output exports/best.onnx
```

Web UI를 serve합니다.

```bash
cd src/coolrl/omok/web
python -m http.server 8080
```

브라우저에서 `http://localhost:8080`을 열고 **Load .onnx**로 모델을 업로드합니다.

Web UI는 모델을 로드할 때 ONNX policy output 길이가 선택한 board 크기와 맞는지 확인합니다. 모델을 로드하기 전에 board-size selector를 먼저 맞추세요.

Controls:

- click to place a stone
- Reset
- Undo
- Swap side
- force AI Move
- simulations slider
