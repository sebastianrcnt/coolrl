# Lost Cities Zero-Pit PoC

## 목적

이 실험은 Lost Cities pure self-play가 `discard` / `draw_pile` 쪽으로 너무 빨리 몰빵하면서 expedition을 거의 열지 않는 zero-pit에 빠지는지 확인한다.

핵심 질문:

```text
random init pure self-play에서 regret matching epsilon만 키우면 zero-pit을 피하거나 약화할 수 있는가?
```

## 실험 이름

```text
lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e3
```

Config:

```text
configs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/config.yaml
```

Run output:

```text
runs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3
```

## 기준 실험과 차이

기준은 기존 Run A다.

```text
configs/lost_cities_deep_cfr_pure_self_play_a.yaml
```

차이는 의도적으로 하나만 둔다.

```yaml
regret_matching_epsilon: 1.0e-8  # 기존 Run A
regret_matching_epsilon: 1.0e-3  # zero-pit PoC
```

seed, network, traversal budget, self-play league, evaluation opponent, memory, optimization은 Run A와 동일하게 둔다. 목적은 safe pretrain이나 resume 없이, random init 조건에서 조기 확률 붕괴만 줄였을 때 행동 분포가 살아나는지 보는 것이다.

## 왜 1.0e-3인가

`1.0e-8`은 사실상 수치 안정성용 0이다. legal action 수가 20개 안팎이면 전체 바닥 질량은 거의 없다.

`1.0e-3`은 특정 action family가 너무 빨리 확률 0 근처로 죽는 것을 막을 만큼은 크지만, `1.0e-2`처럼 정책 전체를 강하게 랜덤화하는 값은 아니다.

이 PoC는 최종 성능 튜닝이 아니라 다음 질문에 답하기 위한 진단 실험이다.

```text
discard-loop가 "경험 부족"보다 "조기 확률 몰빵" 문제에 가까운가?
```

## 실행 명령

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/config.yaml
```

상태 확인:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli status \
  --checkpoint-dir runs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3
```

## 판정 기준

win rate보다 먼저 행동 분포를 본다.

좋은 신호:

- `play_action_rate`가 기존 Run A보다 뚜렷하게 높다.
- `avg_opened_colors`가 0 근처에 고착되지 않는다.
- `draw_deck_rate`가 올라가고 `draw_pile_rate`가 과도하게 높지 않다.
- `safe_heuristic` 계열 timeout이 줄어든다.
- `passive_discard` 상대 `avg_diff`가 0 근처 또는 양수로 이동한다.
- `random` 성능이 완전히 무너지지 않는다.

나쁜 신호:

- `discard_action_rate`가 여전히 0.98 이상에 붙는다.
- `draw_pile_rate`가 높고 `draw_deck_rate`가 낮아 deck 진행을 계속 피한다.
- `play_action_rate`가 오르더라도 `avg_diff`가 더 나빠진다.
- safe 계열 timeout이 줄지 않는다.
- `random` win rate가 빠르게 붕괴한다.

## 해석 원칙

이 실험은 pure self-play 조건을 유지한다.

- external bot은 evaluation에만 사용한다.
- safe imitation checkpoint를 쓰지 않는다.
- 기존 checkpoint에서 resume하지 않는다.
- reward, game rule, timeout semantics를 바꾸지 않는다.

성공하면 `regret_matching_epsilon` schedule 또는 더 작은 값(`1.0e-4`, `3.0e-4`)을 후속 실험한다. 실패하면 zero-pit 원인은 단순한 조기 확률 몰빵보다 terminal visibility, league 구성, game-ending pressure 쪽에 더 가깝다고 본다.
