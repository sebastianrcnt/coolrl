# 진행 로그

## 2026-05-05

### 준비

- Branch: `lost-cities-pure-rl-self-play`
- 진행 디렉터리: `docs/lost-cities-pure-rl`
- Run A config: `configs/lost_cities_deep_cfr_pure_self_play_a.yaml`
- Run B config: `configs/lost_cities_deep_cfr_pure_self_play_b.yaml`

### 코드 변경

- `traversal.opponent_policy: self_play_league`를 추가했다.
- self-play league는 current/recent/older advantage snapshot을 `50/30/20` 비율로 섞는다.
- 외부 bot은 training path에서 쓰지 않고, 기존 eval opponent로만 남긴다.
- evaluation 결과에 `avg_game_length`, `policy_entropy`를 추가했다.
- `train` command에 `--max-hours`, `--max-iterations`, `--checkpoint-dir` override를 추가했다.
- Run B가 pretrain optimizer/iteration을 이어받지 않도록 `train --init-checkpoint`를 추가했다.
- old/best checkpoint 상대 평가를 위해 `eval --opponent-checkpoint`를 추가했다.

### Claude 자문 반영

Claude Opus 4.7 xhigh 자문에서 다음 리스크를 받았다.

- self-play league가 node마다 opponent snapshot을 다시 뽑으면 trajectory 일관성이 깨진다.
- multiprocessing worker batch마다 league snapshot을 반복 deepcopy하면 IPC 비용이 커질 수 있다.
- Run B는 warm start semantics는 맞지만, `strategy_net`도 imitation checkpoint에서 시작하므로 초반 eval 해석에 주의해야 한다.
- 학습 중 eval에는 bot opponent만 있고 checkpoint opponent는 별도 `eval --opponent-checkpoint`로 돌려야 한다.
- 현재 older bucket은 chronological older만 지원하고 best-history 슬롯은 없다.

반영한 결정:

- 공식 A/B 비교 전 코드에서 self-play league opponent를 traversal 단위로 고정한다.
- worker batch 생성 시 league snapshot deepcopy를 iteration당 한 번만 만들도록 바꾼다.
- 2h config의 기본 `max_hours`를 `2`로 두고, worker chunk를 `32`로 늘려 snapshot IPC batch 수를 줄인다.
- README에 advantage snapshot 사용과 best slot 부재를 명시한다.

수정 후 검증:

```bash
uv run pytest \
  src/coolrl/lost_cities/tests/test_deep_cfr_config.py \
  src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py \
  src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py
```

결과:

```text
79 passed in 12.38s
```

보류한 결정:

- best-history training slot은 아직 추가하지 않는다. 요청의 `older or best` 중 chronological older path로 먼저 pure self-play를 검증한다.
- checkpoint-vs-checkpoint 평가는 학습 루프 자동 통합 대신, 우선 `eval --opponent-checkpoint`를 고정 평가 절차에 둔다.

### 아직 실행 전

장시간 학습 run은 아직 시작하지 않았다. 먼저 config/test 검증 후 2h Run A/B부터 실행한다.

### 검증

다음 테스트를 통과했다.

```bash
uv run pytest \
  src/coolrl/lost_cities/tests/test_deep_cfr_config.py \
  src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py \
  src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py
```

결과:

```text
78 passed in 12.18s
```

실제 train CLI config load도 확인했다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_a.yaml \
  --max-iterations 0 \
  --checkpoint-dir /tmp/coolrl_pure_rl_a_config_check
```

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_b.yaml \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --max-iterations 0 \
  --checkpoint-dir /tmp/coolrl_pure_rl_b_config_check
```

두 명령 모두 `device=cuda`, `input_dim=1500`, `actions=22`로 시작 조건을 로드했고, Run B는 safe checkpoint를 optimizer/iteration 없이 network weights만 초기화했다.

### Run A 2h

Run A 2h pilot을 시작했다. 이 run은 Claude 자문 반영 전 코드로 시작했으므로 공식 A/B 비교 결과가 아니라 실행 안정성 확인용으로 취급한다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_a.yaml \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h \
  --max-hours 2
```

- PID: `221133`
- Console log: `checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h/console.log`
- Training log: `checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h/train.log`
- Metrics: `checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h/metrics.jsonl`

초기 확인:

- Iteration 1 완료.
- `self_play_league_snapshots=1`로 snapshot 기록이 시작됐다.
- `cutoff_rollouts=0`, `cutoff_rollout_max_step_timeouts=0`라서 rollout label은 쓰지 않았다.

Iteration 5의 첫 eval 결과:

| Opponent | win_rate | avg_diff | timeouts | avg_game_length | policy_entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 0.74 | 17.26 | 0 | 450.9 | 1.329 |
| `passive_discard` | 0.00 | 0.00 | 0 | 134.1 | 1.037 |
| `safe_heuristic` | 0.15 | -58.02 | 65 | 790.4 | 1.489 |
| `safe_heuristic_loose` | 0.16 | -60.31 | 69 | 796.9 | 1.475 |
| `safe_heuristic_strict` | 0.17 | -50.81 | 62 | 762.8 | 1.514 |
| `noisy_safe` | 0.16 | -49.04 | 36 | 587.2 | 1.381 |

초기 상태는 아직 `safe_heuristic` 계열에 크게 밀리고, `random` 기준도 미달이다. 이는 Run A 초반 random-init baseline의 예상 범위로 보고 계속 진행한다.
