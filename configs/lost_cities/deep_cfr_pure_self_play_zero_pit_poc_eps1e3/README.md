# Lost Cities Deep CFR Pure Self-Play Zero-Pit PoC eps1e3

이 폴더가 `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e3` 실험의 canonical 위치다. 기존 루트 `configs/*.yaml` 경로는 이 실험에 대해 유지하지 않는다.

## 목적

random init pure self-play에서 `regret_matching_epsilon`을 `1.0e-3`으로 키우면 discard/draw-pile 쪽으로 조기 붕괴하는 zero-pit 경향이 줄어드는지 확인한다.

## 기준 실험과 차이

기준은 기존 Run A다.

```text
configs/lost_cities_deep_cfr_pure_self_play_a.yaml
```

의도한 차이는 `regret_matching_epsilon` 하나다.

```yaml
regret_matching_epsilon: 1.0e-8  # Run A
regret_matching_epsilon: 1.0e-3  # 이 PoC
```

seed, network, traversal budget, self-play league, evaluation opponent, memory, optimization은 Run A와 동일하게 둔다.

## 실행

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/config.yaml
```

상태 확인:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli status \
  --checkpoint-dir runs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3
```

plot 생성:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli plot \
  --checkpoint-dir runs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3
```

## 주요 지표

win rate보다 먼저 행동 분포를 본다.

- `play_action_rate`
- `discard_action_rate`
- `draw_deck_rate`
- `draw_pile_rate`
- `eval_*_avg_opened_colors`
- `eval_*_avg_expedition_cards`
- `eval_safe_heuristic_avg_diff`
- `eval_safe_heuristic_win_rate`
- `eval_*_max_step_timeouts`

## 결과 위치

실행 결과는 `runs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/` 아래에 저장한다.

## 관련 문서

해석 노트와 판정 기준은 `docs/lost-cities-zero-pit-poc/README.md`에 둔다.
