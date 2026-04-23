# Lost Cities

`coolrl`을 위한 config-parametrized Lost Cities 구현.

Step 1에는 다음이 포함됩니다:

- `game.py`의 순수 Python rules engine
- `env.py`의 placeholder RL wrapper
- `bots.py`의 random 및 safe heuristic bots
- `tui.py`의 Textual hot-seat / bot-play TUI
- `web/`의 Svelte 5 + TypeScript 웹 스타터
- `tests/` 아래의 Pytest coverage

웹 스타터는 이미 Svelte 5 runes (`$state`, `$derived`, `$effect`)를 사용합니다.
설치 및 실행 방법은 `web/README.md`를 참고하세요.

TUI는 다음과 같이 실행합니다:

```bash
lost-cities --tier tier1
```

패키지가 `lost-cities` extra로 설치되지 않은 경우, env 또는 TUI를 사용하기 전에 `numpy`, `pyyaml`, `textual`을 설치하세요.
