# Lost Cities

Lost Cities 룰 엔진, 환경 래퍼, 봇, GUI, 그리고 Deep CFR 학습 파이프라인을 모아둔 모듈이다.
규칙 자체에 대한 정의는 [`docs/lost_cities_spec.md`](docs/lost_cities_spec.md)를 참고한다.

## 디렉터리 맵

| 경로 | 내용 |
| --- | --- |
| `game.py`, `env.py`, `interfaces.py` | 룰 엔진 / Gym 스타일 env 래퍼 / 공통 타입 |
| `backends/` | `python`(기본), `rust` 백엔드 — `factory.py`로 선택 |
| `bots/` | `random`, `safe-heuristic`, `passive-discard` 봇과 `play.py` 헬퍼, `registry.py` |
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

`bots/play.py`의 `play_game` / `run_series`로 두 봇을 맞붙일 수 있다. 사용 가능한 이름은 `bots/registry.py`의 `BOT_REGISTRY`에 있다 (`random`, `safe-heuristic`, `passive-discard`).

이름 표면이 두 개라서 철자를 구분한다:

- GUI / 일반 bot registry 이름: `passive-discard`
- Deep CFR eval opponent 이름: `passive_discard`

## Deep CFR 학습

CLI 진입점은 `python -m coolrl.lost_cities.deep_cfr.cli` 다.

### Config 프리셋

`configs/` 아래에 다음이 준비돼 있다:

| 파일 | 용도 |
| --- | --- |
| `lost_cities_deep_cfr_probe.yaml` | 1 iteration짜리 스모크 — 파이프라인 점검용 |
| `lost_cities_deep_cfr_tier3.yaml` | 기존 score-diff cutoff 기반 tier3 baseline / CLI 기본 config |
| `lost_cities_deep_cfr_capped_rollout300.yaml` | `random_rollout` cutoff cap 300 실험 — eval 비용 절감, CPU traversal workers |
| `lost_cities_deep_cfr_safe_rollout300.yaml` | `safe_heuristic` rollout cutoff cap 300 실험 — pure random rollout의 over-opening 진단용 |
| `lost_cities_deep_cfr_safe_rollout300_safe_opponent.yaml` | `safe_heuristic` fixed-opponent best-response 실험 |
| `lost_cities_deep_cfr_safe_rollout300_eps05.yaml` | `safe_heuristic` rollout + outcome sampling epsilon 0.5 탐색 실험 |
| `lost_cities_deep_cfr_safe_rollout300_t500.yaml` | `safe_heuristic` rollout + player당 500 traversal 샘플 품질 실험 |
| `lost_cities_deep_cfr_safe_rollout300_clip500.yaml` | `safe_heuristic` rollout + outcome-sampled value clip 500 안정화 실험 |
| `lost_cities_deep_cfr_safe_rollout300_eps02_t500_clip500.yaml` | clip 500 안정화 위에서 epsilon 0.2 + player당 500 traversal를 결합한 discard spiral 탈출 실험 |
| `lost_cities_deep_cfr_safe_rollout300_zero_unsampled.yaml` | 미샘플 액션 regret를 0으로 두고 clip 500 + epsilon 0.2 + player당 500 traversal를 결합한 discard spiral 완화 실험 |
| `lost_cities_deep_cfr_safe_br_zero_unsampled.yaml` | `safe_heuristic`을 고정 opponent로 둔 best-response 목적 실험 |
| `lost_cities_deep_cfr_safe_dagger_256.yaml` | 기존 256x3 pretrain checkpoint를 초기화점으로 쓰는 aggregated imitation / DAGGER-style 후속 pretrain config |
| `lost_cities_deep_cfr_safe_dagger_512.yaml` | aggregated imitation / DAGGER-style 후속 pretrain용 512x4 config. `random`, `safe_heuristic`, variant-safe, `passive_discard` 평가를 함께 본다 |

과거 실험용 `overnight`, `small_run`, `diagnostic_depth16_nodes20k`, `cutoff_random_rollout` config는 결과가 문서화된 뒤 제거했다. 관련 기록은 루트 문서 [`docs/lost-cities-deep-cfr-training-notes.md`](../../../docs/lost-cities-deep-cfr-training-notes.md)와 [`docs/lost-cities-deep-cfr-worker-benchmark-notes.md`](../../../docs/lost-cities-deep-cfr-worker-benchmark-notes.md)를 참고한다.

### 학습 시작

기본 tier3 baseline config로 학습을 시작한다:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_tier3.yaml
```

체크포인트는 config의 `checkpoint.directory`에 저장된다. 디렉터리에 이미 `metrics.jsonl`이 있는데 `--resume` 없이 train을 다시 돌리면 트레이너가 기존 metrics를 timestamp suffix로 archive한다.

### 학습 이어가기

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_tier3.yaml \
  --resume
```

- `--config`는 처음 돌릴 때와 같은 YAML을 그대로 지정한다.
- `--resume`에 경로를 생략하면 config의 `checkpoint.directory/latest.pt`를 사용한다.
- 특정 checkpoint에서 이어가려면 `--resume checkpoints/lost_cities_deep_cfr_tier3/iteration_00050.pt`처럼 경로를 명시한다.
- **복원되는 것**: advantage / strategy 네트워크 weight, 옵티마이저 state, `iteration` 카운터.
- **복원되지 않는 것**: reservoir 메모리(샘플 버퍼)와 RNG 상태. traversal 샘플은 처음부터 다시 쌓인다 (`trainer.py`의 `load_checkpoint` 경고 참고).

### 진행 상태 / 시각화 / 평가

`train`은 콘솔 출력과 같은 로그를 checkpoint 디렉터리의 `train.log`에도 남긴다. `--resume` 없이 새 run을 시작할 때 기존 `train.log`가 있으면 timestamp suffix로 archive된다:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_tier3.yaml
```

실행 중에는 다음 명령으로 모니터링한다:

```bash
# 실시간 로그
tail -f checkpoints/lost_cities_deep_cfr_tier3/train.log

# 최근 완료 iteration 요약
watch -n 10 'uv run python -m coolrl.lost_cities.deep_cfr.cli status --checkpoint-dir checkpoints/lost_cities_deep_cfr_tier3'

# iteration별 raw metrics
tail -f checkpoints/lost_cities_deep_cfr_tier3/metrics.jsonl
```

`runtime_progress.json`과 `metrics.jsonl`은 iteration이 끝날 때 갱신된다. `status`는 이 파일들을 읽어 최신 iteration, loss, traversal 속도, eval 결과를 요약한다.

```bash
# 메트릭 요약
uv run python -m coolrl.lost_cities.deep_cfr.cli status \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_tier3

# 학습 곡선 플롯
uv run python -m coolrl.lost_cities.deep_cfr.cli plot \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_tier3

# 봇과 N판 평가
uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
  --checkpoint checkpoints/lost_cities_deep_cfr_tier3/latest.pt \
  --games 500 --opponent safe_heuristic
```

Deep CFR eval opponent는 `random`, `safe_heuristic`, `passive_discard`를 지원한다. `passive_discard`는 expedition을 열지 않는 baseline이라 random win rate만으로 passive collapse를 착각하는 문제를 잡는 데 사용한다.

추가 일반화 진단용으로 `safe_heuristic_loose`, `safe_heuristic_strict`, `noisy_safe`도 지원한다. 이 opponent들은 `safe_heuristic` 하나에만 맞춘 정책인지 빠르게 확인하기 위한 variant-safe baseline이다.

### Safe heuristic imitation / DAGGER-style pretrain

기본 safe self-play imitation checkpoint:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli pretrain-heuristic \
  --config configs/lost_cities_deep_cfr_safe_dagger_512.yaml \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_512/base.pt \
  --games 1600 --epochs 16 --batch-size 2048 --max-steps 1000
```

기존 checkpoint가 만든 상태분포까지 섞는 aggregated imitation:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli pretrain-heuristic \
  --config configs/lost_cities_deep_cfr_safe_dagger_256.yaml \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_256/aggregated.pt \
  --dataset-mode aggregated \
  --base-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --games 1600 --epochs 8 --batch-size 2048 --max-steps 1000
```

`aggregated` mode는 safe-vs-safe, policy-vs-safe, policy-vs-policy로 게임을 진행하되, target action은 항상 현재 상태에서 `SafeHeuristicBot`이 고르는 행동으로 저장한다. 목적은 fixed safe best-response가 아니라, 모델이 실제로 도달하는 상태분포에서 safe-style correction을 받는 것이다.

`safe_heuristic`을 직접 넘기기 위한 exploitation 실험은 `successful_policy_vs_safe` mode를 쓴다. 이 mode는 `--base-checkpoint` 정책이 `safe_heuristic`을 상대로 실제로 이긴 게임에서 해당 정책이 둔 행동만 imitation target으로 저장한다:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli pretrain-heuristic \
  --config configs/lost_cities_deep_cfr_safe_dagger_256.yaml \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_256/successful_policy_vs_safe.pt \
  --dataset-mode successful_policy_vs_safe \
  --base-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --games 2000 --epochs 2 --batch-size 2048 --max-steps 1000
```

더 직접적인 action-level improvement 실험은 `safe_action_rollout` mode를 쓴다. 이 mode는 base policy가 `safe_heuristic` 상대로 지거나 timeout 나는 trajectory를 우선 모으고, 각 상태의 legal action을 적용한 뒤 짧은 policy-vs-safe rollout으로 더 좋은 action을 label로 고른다:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli pretrain-heuristic \
  --config configs/lost_cities_deep_cfr_safe_dagger_256.yaml \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_256/safe_action_rollout.pt \
  --dataset-mode safe_action_rollout \
  --base-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --games 200 --epochs 2 --batch-size 1024 --max-steps 1000 \
  --improvement-rollouts 1 \
  --improvement-rollout-max-steps 300 \
  --improvement-max-examples 2000
```

현재 training caveat와 다음 실험 기준은 루트 문서 [`docs/lost-cities-deep-cfr-training-notes.md`](../../../docs/lost-cities-deep-cfr-training-notes.md)를 참고한다.

## 테스트

```bash
uv run pytest src/coolrl/lost_cities/tests
```

`test_rust_parity.py`는 Rust 코어를 `cargo`로 호출하므로 Rust 툴체인이 없으면 실패한다. Python 백엔드만 검증하려면 해당 파일을 제외하면 된다.
