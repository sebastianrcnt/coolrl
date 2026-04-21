# 잃어버린 도시

`coolrl`에 대한 구성 매개변수화된 잃어버린 도시 구현입니다.

1단계에는 다음이 포함됩니다.

- `game.py`의 순수 Python 규칙 엔진
- `env.py`의 자리 표시자 RL 래퍼
- `bots.py`의 무작위적이고 안전한 경험적 봇
-`tui.py`의 텍스트 핫시트/봇 플레이 TUI
- `tests/` 아래의 Pytest 적용 범위

다음을 사용하여 TUI를 실행합니다.
```bash
lost-cities --tier tier1
```
패키지가 `lost-cities` extra와 함께 설치되지 않은 경우 `numpy`를 설치하세요.
env 또는 TUI를 사용하기 전에 `pyyaml` 및 `textual`을 사용하세요.
