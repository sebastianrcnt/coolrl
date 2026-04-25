# Lost Cities

Lost Cities 룰 엔진, 환경 래퍼, 봇, GUI, 그리고 Deep CFR 학습 파이프라인을 모아둔 모듈이다.
규칙 자체에 대한 정의는 [`docs/lost_cities_spec.md`](docs/lost_cities_spec.md)를 참고한다.

## 디렉터리 맵

| 경로 | 내용 |
| --- | --- |
| `game.py`, `env.py`, `interfaces.py` | 룰 엔진 / Gym 스타일 env 래퍼 / 공통 타입 |
| `backends/` | `python`(기본), `rust` 백엔드 — `factory.py`로 선택 |
| `bots/` | `random`, `safe-heuristic` 봇과 `play.py` 헬퍼, `registry.py` |
| `deep_cfr/` | Deep CFR 학습 코드(`trainer.py`, `cli.py`, `config.py`, `evaluate.py` 등) |
| `pygame_pvp.py` | Pygame 기반 PvP / PvC GUI 진입점 |
| `web/` | Svelte 5 웹 클라이언트 (자체 [README](web/README.md) 참고) |
| `rust_core/` | Rust 룰 엔진 — `cargo`로 빌드되는 별도 크레이트 |
| `docs/lost_cities_spec.md` | 게임 규칙 스펙 |
| `tests/` | pytest 스위트 |

## 설치

루트 [설정 가이드](../../../docs/setup.md) 후 `lost-cities` extras를 함께 동기화한다:

```bash
uv sync --extra lost-cities
```

`rust` 백엔드와 `tests/test_rust_parity.py`는 `rust_core/`의 `cargo` 빌드가 필요하다 (즉 `cargo`가 설치돼 있어야 한다).

## 사람이 직접 플레이

Pygame GUI:

```bash
uv run lost-cities-gui --mode pvp
uv run lost-cities-gui --mode pvc --bot safe-heuristic --tier tier3
```

주요 옵션: `--backend {python,rust}`, `--tier {tier0,tier1,tier2,tier3}`, `--seed`, `--deep-cfr-checkpoint <path>` (학습된 정책으로 PvC).

## 봇 vs 봇

`bots/play.py`의 `play_game` / `run_series`로 두 봇을 맞붙일 수 있다. 사용 가능한 이름은 `bots/registry.py`의 `BOT_REGISTRY`에 있다 (`random`, `safe-heuristic`).

## Deep CFR 학습

CLI 진입점은 `python -m coolrl.lost_cities.deep_cfr.cli` 다.

### Config 프리셋

`configs/` 아래에 다음이 준비돼 있다:

| 파일 | 용도 |
| --- | --- |
| `lost_cities_deep_cfr_probe.yaml` | 1 iteration짜리 스모크 — 파이프라인 점검용 |
| `lost_cities_deep_cfr_small_run.yaml` | 10 iteration, hidden=128 — 빠른 sanity |
| `lost_cities_deep_cfr_tier3.yaml` | 100 iteration, hidden=256, CUDA + AMP — 표준 학습 |
| `lost_cities_deep_cfr_overnight.yaml` | 8시간 wall-clock, hidden=128, CPU — 장시간 실행 |

### 학습 시작

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_tier3.yaml
```

체크포인트는 config의 `checkpoint.directory`(기본 `checkpoints/lost_cities_deep_cfr_tier3/`)에 저장된다. 디렉터리에 이미 `metrics.jsonl`이 있는데 `--resume` 없이 train을 다시 돌리면 트레이너가 막아준다.

### 학습 이어가기

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_overnight.yaml \
  --resume checkpoints/lost_cities_deep_cfr_overnight/latest.pt
```

- `--config`는 처음 돌릴 때와 같은 YAML을 그대로 지정한다.
- `--resume`에는 보통 `latest.pt`를 준다. `checkpoint.save_latest_only=false`(기본값)면 `iteration_XXXXX.pt` 스냅샷도 같이 저장되므로 그중 하나를 골라도 된다.
- **복원되는 것**: advantage / strategy 네트워크 weight, 옵티마이저 state, `iteration` 카운터.
- **복원되지 않는 것**: reservoir 메모리(샘플 버퍼)와 RNG 상태. traversal 샘플은 처음부터 다시 쌓인다 (`trainer.py`의 `load_checkpoint` 경고 참고).

### 진행 상태 / 시각화 / 평가

```bash
# 메트릭 요약
uv run python -m coolrl.lost_cities.deep_cfr.cli status \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_overnight

# 학습 곡선 플롯
uv run python -m coolrl.lost_cities.deep_cfr.cli plot \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_overnight

# 봇과 N판 평가
uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
  --checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt \
  --games 500 --opponent safe_heuristic
```

## 테스트

```bash
uv run pytest src/coolrl/lost_cities/tests
```

`test_rust_parity.py`는 Rust 코어를 `cargo`로 호출하므로 Rust 툴체인이 없으면 실패한다. Python 백엔드만 검증하려면 해당 파일을 제외하면 된다.
