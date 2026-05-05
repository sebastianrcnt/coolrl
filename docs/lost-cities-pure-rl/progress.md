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
- 고정 평가 suite를 재현 가능하게 남기기 위해 `eval-suite` command를 추가했다.

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

`eval-suite` 추가 후 다시 검증했다.

```bash
uv run pytest \
  src/coolrl/lost_cities/tests/test_deep_cfr_config.py \
  src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py \
  src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py
```

결과:

```text
80 passed in 12.48s
```

보류한 결정:

- best-history training slot은 아직 추가하지 않는다. 요청의 `older or best` 중 chronological older path로 먼저 pure self-play를 검증한다.
- checkpoint-vs-checkpoint 평가는 학습 루프 자동 통합 대신, 우선 `eval --opponent-checkpoint`를 고정 평가 절차에 둔다.

Claude 2차 자문에서 post-run checkpoint opponent 선정 기준을 추가로 받았다.

- old checkpoint는 각 run의 총 iteration `N` 기준 `N/4`, `N/2`, `3N/4`, `N`에 가장 가까운 checkpoint로 고른다.
- best checkpoint는 6개 bot opponent의 training eval 평균 win rate가 가장 높은 eval iteration으로 고른다.
- best 선정 eval과 최종 500-game eval은 분리한다.
- Run B는 원래 실험 목적상 같은 pure self-play 조건으로 실행하되, Run A의 passive/safe timeout 실패 모드를 반드시 별도 trace한다.

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

Claude 자문 반영 전 pilot은 중단했다. 이후 공식 Run A 2h를 안정화 코드로 새로 시작했다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_a.yaml \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official \
  --max-hours 2
```

- PID: `225797`
- Console log: `checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/console.log`
- Training log: `checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/train.log`
- Metrics: `checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/metrics.jsonl`

시작 확인:

- `traversal_worker_chunk_size=32` 적용으로 batch 수가 `126`에서 `32`로 줄었다.
- Iteration 1-2가 정상 완료됐다.
- `cutoff_rollouts=0` 유지.

Iteration 5의 첫 eval 결과:

| Opponent | win_rate | avg_diff | timeouts | avg_game_length | policy_entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 0.37 | -11.49 | 0 | 516.0 | 1.685 |
| `passive_discard` | 0.01 | -26.18 | 0 | 168.4 | 1.805 |
| `safe_heuristic` | 0.13 | -59.88 | 68 | 757.9 | 1.632 |
| `safe_heuristic_loose` | 0.12 | -63.62 | 66 | 731.7 | 1.610 |
| `safe_heuristic_strict` | 0.15 | -56.89 | 67 | 727.5 | 1.624 |
| `noisy_safe` | 0.13 | -60.66 | 38 | 575.9 | 1.633 |

공식 Run A 초반은 random/passive/safe 모두에서 기준 미달이다. `cutoff_rollouts=0`과 `self_play_league_snapshots=5`는 의도대로 기록됐다.

Iteration 20 eval:

| Opponent | win_rate | avg_diff | timeouts |
| --- | ---: | ---: | ---: |
| `random` | 0.62 | 9.33 | 0 |
| `passive_discard` | 0.00 | -16.85 | 0 |
| `safe_heuristic` | 0.18 | -63.65 | 48 |
| `safe_heuristic_loose` | 0.18 | -69.54 | 47 |
| `safe_heuristic_strict` | 0.19 | -53.28 | 49 |
| `noisy_safe` | 0.17 | -52.42 | 37 |

Iteration 21 기준 `self_play_league_snapshots=20`으로 configured cap에 도달했다. `traversal_seconds`는 `4.91s`로 pilot의 `14` snapshots 시점 `13.81s`보다 낮아, chunk size 조정과 snapshot deepcopy 캐시가 IPC 병목을 줄인 것으로 보인다.

### Run A 진단 trace

Claude 2차 자문은 Run B 전 `passive_discard` 패배와 safe 계열 timeout의 원인을 trace하라고 권했다. 공식 Run A 최신 checkpoint로 1게임씩 덤프했다.

`passive_discard` 상대 losing seed:

```text
seed=610000 tracked_player=0
timed_out=False steps=132 diff=-25 final_score=-25 opponent_score=0 deck=0
opened=4 expedition_cards=8
counts={'play': 8, 'discard': 25, 'draw_deck': 11, 'draw_pile': 22}
expedition_lengths=[2, 2, 3, 1, 0]
```

해석:

- 초반에 4색 expedition을 열고 충분히 회수하지 못했다.
- 이후 `draw_pile`이 `draw_deck`보다 많아, passive 상대에서도 점수 회수보다 pile cycling 성향이 강하다.
- passive 기준 실패는 단순 timeout 문제가 아니라 bad expedition opening과 회수 실패다.

`safe_heuristic` 상대 timeout seed:

```text
seed=620000 tracked_player=0
timed_out=True steps=1000 diff=5 final_score=0 opponent_score=-5 deck=22
opened=0 expedition_cards=0
counts={'play': 0, 'discard': 250, 'draw_deck': 7, 'draw_pile': 243}
expedition_lengths=[0, 0, 0, 0, 0]
```

해석:

- expedition을 전혀 열지 않았다.
- `draw_pile` 243회, `draw_deck` 7회로 deck-out을 회피하는 루프 성향이 명확하다.
- safe 계열 timeout 악화는 policy가 game-ending pressure를 충분히 학습하지 못한 신호다.

Run B는 원래 계획상 같은 pure self-play 조건으로 실행하되, 결과 해석에서는 위 두 실패 모드를 별도 기준으로 추적한다.

Iteration 50 eval:

| Opponent | win_rate | avg_diff | timeouts |
| --- | ---: | ---: | ---: |
| `random` | 0.56 | -1.65 | 0 |
| `passive_discard` | 0.01 | -14.10 | 0 |
| `safe_heuristic` | 0.15 | -63.27 | 86 |
| `safe_heuristic_loose` | 0.13 | -74.81 | 77 |
| `safe_heuristic_strict` | 0.16 | -55.32 | 86 |
| `noisy_safe` | 0.15 | -52.56 | 49 |

Iteration 54 기준 elapsed는 18분이다. safe family timeout이 iteration 20보다 악화했으므로, Run A는 현재까지 game-ending pressure를 학습하지 못하고 있다.

Iteration 95 eval:

| Opponent | win_rate | avg_diff | timeouts |
| --- | ---: | ---: | ---: |
| `random` | 0.63 | 4.84 | 0 |
| `passive_discard` | 0.00 | -12.26 | 0 |
| `safe_heuristic` | 0.03 | -69.13 | 43 |
| `safe_heuristic_loose` | 0.07 | -81.77 | 38 |
| `safe_heuristic_strict` | 0.04 | -55.95 | 26 |
| `noisy_safe` | 0.10 | -62.43 | 20 |

Iteration 99 기준 elapsed는 34분이다. `random`은 iteration 50보다 조금 회복했지만, `passive_discard`는 여전히 0%이고 safe family 승률은 더 낮아졌다. Timeout 수는 줄었지만 score diff가 악화되어, 단순히 게임을 끝내는 방향이 강해진 것이 성능 개선으로 이어지지는 않았다.

Iteration 230 eval:

| Opponent | win_rate | avg_diff | timeouts |
| --- | ---: | ---: | ---: |
| `random` | 0.88 | 35.00 | 0 |
| `passive_discard` | 0.00 | -0.81 | 0 |
| `safe_heuristic` | 0.03 | -72.48 | 10 |
| `safe_heuristic_loose` | 0.03 | -81.74 | 8 |
| `safe_heuristic_strict` | 0.05 | -64.69 | 12 |
| `noisy_safe` | 0.11 | -47.91 | 0 |

Iteration 234 기준 elapsed는 65분이다. `random`과 `passive_discard` avg_diff는 개선됐지만, safe family 승률과 score diff는 여전히 크게 실패하고 있다. Timeout은 줄었으므로 루프는 줄었지만, safe 상대 전략 품질은 개선되지 않았다.

Iteration 355 eval:

| Opponent | win_rate | avg_diff | timeouts |
| --- | ---: | ---: | ---: |
| `random` | 0.55 | 2.41 | 0 |
| `passive_discard` | 0.00 | -22.48 | 0 |
| `safe_heuristic` | 0.06 | -93.81 | 48 |
| `safe_heuristic_loose` | 0.04 | -101.34 | 43 |
| `safe_heuristic_strict` | 0.05 | -88.13 | 53 |
| `noisy_safe` | 0.10 | -70.02 | 32 |

Iteration 358 기준 elapsed는 100분이다. 중반에 좋아졌던 `random`/`passive_discard`도 다시 악화했고, safe family score diff는 더 나빠졌다. Run A는 현재까지 pure self-play만으로 안정적인 개선 곡선을 만들지 못하고 있다.

Iteration 410 eval:

| Opponent | win_rate | avg_diff | timeouts |
| --- | ---: | ---: | ---: |
| `random` | 0.70 | 21.93 | 0 |
| `passive_discard` | 0.00 | -5.64 | 0 |
| `safe_heuristic` | 0.13 | -77.46 | 69 |
| `safe_heuristic_loose` | 0.14 | -88.96 | 60 |
| `safe_heuristic_strict` | 0.11 | -71.35 | 73 |
| `noisy_safe` | 0.10 | -63.32 | 37 |

Run A 2h는 iteration 414, elapsed 120.1분에서 종료됐다. 최종 checkpoint는 `checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/latest.pt`다.

최종 500-game bot eval suite:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli eval-suite \
  --checkpoint checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/latest.pt \
  --games 500 \
  --max-steps 1000 \
  --output docs/lost-cities-pure-rl/eval-a-2h.json
```

| Opponent | win_rate | avg_diff | timeouts | avg_game_length | policy_entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 0.658 | 15.328 | 0 | 395.224 | 1.794 |
| `passive_discard` | 0.000 | -3.974 | 0 | 152.376 | 1.708 |
| `safe_heuristic` | 0.116 | -76.424 | 321 | 728.536 | 1.567 |
| `safe_heuristic_loose` | 0.094 | -87.752 | 279 | 658.204 | 1.573 |
| `safe_heuristic_strict` | 0.122 | -67.700 | 374 | 822.592 | 1.558 |
| `noisy_safe` | 0.106 | -64.404 | 185 | 616.516 | 1.640 |

성공 기준 대비:

- `safe_heuristic` win rate는 0.116으로 0.50 기준에 크게 못 미친다.
- `safe_heuristic` mean score diff는 -76.424로 양수가 아니다.
- `random` win rate는 0.658로 0.95 기준에 못 미친다.
- `passive_discard`는 win rate 0.000이고 avg diff도 -3.974다.
- safe family timeout은 500게임 중 279-374회로, timeout 악화가 뚜렷하다.

따라서 Run A 2h는 pure self-play만으로 `safe_heuristic` 계열을 넘는다는 가설을 지지하지 못한다.

old/best checkpoint opponent 선정:

- 총 iteration `N=414` 기준 quartile checkpoint: `104`, `207`, `310`, `414`
- training eval 6개 bot 평균 win rate best checkpoint: `45`
- best 선정에 사용한 training eval은 100-game 중간 eval이고, 최종 판단은 아래 500-game eval 결과로 분리한다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli eval-suite \
  --checkpoint checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/latest.pt \
  --games 500 \
  --max-steps 1000 \
  --opponent-checkpoints \
    checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/iteration_00104.pt \
    checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/iteration_00207.pt \
    checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/iteration_00310.pt \
    checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/iteration_00414.pt \
    checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official/iteration_00045.pt \
  --output docs/lost-cities-pure-rl/eval-a-2h-checkpoints.json
```

| Opponent checkpoint | win_rate | avg_diff | timeouts | avg_game_length | policy_entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `iteration_00104.pt` | 0.302 | 5.020 | 3 | 188.524 | 1.730 |
| `iteration_00207.pt` | 0.344 | 10.958 | 186 | 484.632 | 1.629 |
| `iteration_00310.pt` | 0.096 | -1.160 | 422 | 874.592 | 1.478 |
| `iteration_00414.pt` | 0.114 | 0.074 | 497 | 995.176 | 1.418 |
| `iteration_00045.pt` | 0.258 | 9.536 | 197 | 512.180 | 1.428 |

latest-vs-latest self-play checkpoint 평가는 500게임 중 497회가 max-step timeout이다. latest policy가 자기 자신과도 거의 게임을 끝내지 못하므로, Run A 2h는 학습 안정성 기준에서도 실패 신호가 강하다.

### Run B 2h

Run B 2h를 시작했다. 이 run은 safe imitation checkpoint에서 network weights만 초기화하고, 이후 학습은 Run A와 같은 pure self-play 조건으로 진행한다. optimizer, iteration, replay memory, RNG, self-play league snapshot은 복원하지 않는다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_b.yaml \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_b_2h_official \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --max-hours 2
```

- PID: `314558`
- Console log: `checkpoints/lost_cities_deep_cfr_pure_self_play_b_2h_official/console.log`
- Training log: `checkpoints/lost_cities_deep_cfr_pure_self_play_b_2h_official/train.log`
- Metrics: `checkpoints/lost_cities_deep_cfr_pure_self_play_b_2h_official/metrics.jsonl`

시작 확인:

- `initialize_from_checkpoint` 로그에서 network만 초기화하고 optimizer/memory/RNG/league snapshots를 복원하지 않았음을 확인했다.
- `cutoff_rollouts=0` 유지.
- iteration 29 기준 `self_play_league_snapshots=20` cap에 도달했다.

Iteration 25 eval:

| Opponent | win_rate | avg_diff | timeouts |
| --- | ---: | ---: | ---: |
| `random` | 0.62 | 11.58 | 0 |
| `passive_discard` | 0.02 | -21.44 | 0 |
| `safe_heuristic` | 0.02 | -73.93 | 13 |
| `safe_heuristic_loose` | 0.03 | -82.73 | 13 |
| `safe_heuristic_strict` | 0.03 | -70.82 | 17 |
| `noisy_safe` | 0.06 | -67.71 | 7 |

Run B 초반은 safe pretrain 초기화에도 safe family 승률이 낮다. 다만 timeout은 Run A 같은 시간대보다 낮아, game-ending loop는 아직 덜하지만 score diff는 여전히 크게 음수다. 이후 곡선에서 timeout이 다시 증가하는지와 `passive_discard` 실패가 완화되는지를 함께 본다.

### Claude 3차 자문

Run A 최종 결과와 Run B iteration 25 결과를 바탕으로 Claude Opus 4.7 xhigh에 다시 자문했다. 받은 해석은 다음과 같다.

- Run A 2h 실패는 단순 budget 부족보다 `on_max_steps: score_diff`와 timeout penalty 부재가 만드는 reward/game-ending pressure 결함 신호가 강하다.
- latest-vs-latest checkpoint 평가에서 500게임 중 497회 timeout이 난 것은, self-play가 0-0 또는 expedition 미개시 균형으로 수렴하는 강한 증거다.
- Run B는 iteration 100 전후에 safe family timeout, `avg_opened_colors`, `avg_play_actions`, `policy_entropy`, mirror match timeout을 같이 봐야 한다.
- Run B가 booster라면 safe family timeout이 Run A 같은 시점보다 낮게 유지되고, expedition opening과 entropy가 단조 붕괴하지 않으며, safe score diff가 유의미하게 개선되어야 한다.
- Run B가 bias/붕괴라면 `discard_action_rate` 증가, `avg_opened_colors` 감소, timeout 증가, `passive_discard` 0승 회귀가 함께 나타날 것이다.
- Run C 전에는 timeout penalty, cutoff/max-depth 설계, best-history league slot, exploration schedule을 다시 검토해야 한다. 다만 Run C는 A/B pure self-play 비교 후에만 검토한다.

채택한 결정:

- Run B는 계획대로 pure self-play 조건을 깨지 않고 끝까지 실행한다.
- Run B 중간 문서화에서는 win rate만 보지 않고 timeout, `avg_game_length`, policy entropy, opening/play/discard 계열 지표도 같이 비교한다.
- Run B 종료 후에는 Run A와 같은 bot suite, quartile/best checkpoint suite, 그리고 가능하면 Run A latest와 Run B latest cross match를 평가 전용으로 추가한다.

Iteration 55 eval:

| Opponent | win_rate | avg_diff | timeouts | avg_game_length | policy_entropy | avg_opened_colors | play_rate | discard_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random` | 0.89 | 44.45 | 0 | 254.76 | 1.315 | 0.16 | 0.004 | 0.996 |
| `passive_discard` | 0.00 | -0.11 | 0 | 91.92 | 1.248 | 0.01 | 0.000 | 1.000 |
| `safe_heuristic` | 0.18 | -43.94 | 95 | 969.94 | 0.787 | 0.89 | 0.006 | 0.994 |
| `safe_heuristic_loose` | 0.20 | -49.36 | 94 | 957.46 | 0.800 | 0.94 | 0.006 | 0.994 |
| `safe_heuristic_strict` | 0.16 | -39.67 | 90 | 931.64 | 0.766 | 0.86 | 0.006 | 0.994 |
| `noisy_safe` | 0.20 | -41.35 | 54 | 778.90 | 0.875 | 0.87 | 0.006 | 0.994 |

Run B iteration 55는 Run A 같은 시간대보다 `random`과 safe family score diff가 좋다. 그러나 safe family timeout이 90/100 전후로 이미 높고, `discard_rate`가 거의 99% 이상이라 Claude가 지적한 discard-loop collapse 신호가 강하게 나타난다. `passive_discard`는 score diff가 거의 0까지 왔지만 여전히 win rate 0이다.

Iteration 70 eval:

| Opponent | win_rate | avg_diff | timeouts | avg_game_length | policy_entropy | avg_opened_colors | play_rate | discard_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random` | 0.80 | 32.88 | 0 | 347.86 | 1.112 | 0.03 | 0.000 | 1.000 |
| `passive_discard` | 0.00 | 0.00 | 0 | 92.42 | 1.147 | 0.00 | 0.000 | 1.000 |
| `safe_heuristic` | 0.19 | -37.85 | 99 | 997.62 | 0.656 | 0.28 | 0.002 | 0.998 |
| `safe_heuristic_loose` | 0.19 | -46.88 | 96 | 975.54 | 0.660 | 0.31 | 0.002 | 0.998 |
| `safe_heuristic_strict` | 0.22 | -28.28 | 99 | 998.36 | 0.643 | 0.27 | 0.002 | 0.998 |
| `noisy_safe` | 0.26 | -27.44 | 72 | 901.48 | 0.689 | 0.27 | 0.001 | 0.999 |

Run B iteration 70은 safe family win rate와 score diff가 iteration 55보다 조금 개선됐지만, safe family timeout이 96-99/100까지 올라갔다. `policy_entropy`, `avg_opened_colors`, `play_rate`가 함께 낮아져서, safe pretrain prior가 유지되기보다 discard-loop 방향으로 빠르게 붕괴하는 신호가 강하다.
