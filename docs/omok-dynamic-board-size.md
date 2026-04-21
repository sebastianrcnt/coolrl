# 오목 다이나믹보드 사이즈

`coolrl.omok`은 지원되는 사각형 보드에 대한 단일 Omok 구현입니다.
크기. 이전에 복제된 `coolrl.omok15` 패키지는 폐기되었습니다. 15x15 실행
이제 동일한 트레이너, 기능 인코더, 네트워크, 체크포인트, 플로팅,
웹 UI, GUI, Python MCTS, C MCTS 및 Rust MCTS 경로는 9x9입니다.

## 디자인

보드 크기는 `rules.board_size`에서 가져오며 각 런타임을 통해 전달됩니다.
표면:
```text
action = row * board_size + col
row, col = divmod(action, board_size)
action_size = board_size * board_size
feature_stride = 4 * action_size
```
Python 게임 상태는 `board_size >= 5`인 정사각형 보드를 허용합니다. 네이티브 C
Rust 백엔드는 현재 5부터 19까지의 크기를 허용합니다.
다른 컴파일 타임 포크 없이 9x9, 13x13 및 15x15를 다룹니다.

한 번의 MCTS `search_batch` 호출은 단일 보드 크기를 사용해야 합니다. 9x9와 15x15 혼합
한 배치의 상태는 정책과
기능 텐서 모양이 다릅니다.

## MCTS 백엔드

세 가지 백엔드 모두 동일한 Python 관련 검색 계약을 공유합니다.

- Python MCTS는 `GameState.action_size`에서 `action_size`를 파생합니다.
- C MCTS는 각 트리 생성 시 `board_size`를 받아 자식을 할당하고,
  기능 및 런타임 `action_size`의 방문 횟수 저장.
- Rust MCTS는 동일한 런타임 크기의 트리 API를 미러링하고 보드/액션을 노출합니다.
  FFI 게터를 통한 메타데이터.

C 및 Rust 래퍼는 다음을 검증합니다.

- 배치의 모든 상태는 동일한 보드 크기를 갖습니다.
- 재사용된 루트는 들어오는 상태 보드 크기와 일치합니다.
- 평가자 우선 순위는 `[batch,board_size *board_size]` 형태를 갖습니다.

이유를 밝힌 15x15 메모리 사고는 `docs/omok-mcts-memory.md`를 참조하세요.
기본 MCTS 노드 수명 및 밀도가 높은 하위 스토리지는 확장 시 특별한 주의가 필요합니다.
9x9에서 더 큰 보드까지.

## 구성

기본 9x9 사전 설정은 그대로 유지됩니다.

-`configs/omok_smoke.yaml`
-`configs/omok_quick.yaml`
-`configs/omok_full_cuda.yaml`
-`configs/omok_full_metal.yaml`

15x15 사전 설정은 이제 일반적인 `coolrl.omok` 구성입니다.

-`configs/omok15_smoke.yaml`
- `configs/omok15_quick.yaml`
- `configs/omok15_full_cuda.yaml`

다음을 사용하여 실행하세요.
```bash
uv run python -m coolrl.omok.train --config configs/omok15_smoke.yaml --device CPU
uv run python -m coolrl.omok.train --config configs/omok15_full_cuda.yaml
```
체크포인트 디렉토리는 보드 크기에 따라 별도로 유지되어야 합니다. 9x9 체크포인트에는
정책 헤드의 길이는 81이고, 15x15 체크포인트의 길이는 225입니다. 로드 중
보드 크기가 다른 네트워크에 대한 체크포인트는 실패할 것으로 예상됩니다.

## 툴링

ONNX 내보내기는 `cfg.rules.board_size`에서 더미 입력을 빌드합니다. 파이게임 GUI
`--board-size`를 허용하며 브라우저 UI에는 보드 크기 선택기가 있습니다. 둘 다
인터페이스는 로드된 모델의 정책 출력 길이가
선택한 보드 크기.

훈련 측정항목은 이제 'board_size'를 기록합니다. 'omok-plot'은 해당 값을 사용하거나
필요한 경우 올바른 유니폼 정책을 그리기 위한 체크포인트 사이드카 구성
엔트로피 참조:
```python
uniform_policy_entropy = np.log(board_size * board_size)
```
## 다른 크기 추가

13x13과 같은 표준 정사각형 크기의 경우 다음을 사용하여 구성을 추가하세요.
```yaml
rules:
  board_size: 13

checkpoint:
  directory: checkpoints/omok13_quick
```
크기가 유지되는 한 새 패키지나 기본 백엔드 포크가 필요하지 않습니다.
기본 백엔드 한도 내에서. 새 크기에 대한 패리티 적용 범위를 추가합니다.
임시 실험이 아닌 공식적으로 유지되는 사전 설정이 됩니다.
