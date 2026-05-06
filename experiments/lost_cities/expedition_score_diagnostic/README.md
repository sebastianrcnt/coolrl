# Lost Cities Expedition Score Diagnostic

학습 프로세스를 건드리지 않고 StrategyNet checkpoint를 offline evaluation으로 돌려 opened expedition의 최종 점수 분포를 분석한다.

## 실행

```bash
uv run python experiments/lost_cities/expedition_score_diagnostic/diagnose.py \
  --checkpoint checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability/latest.pt \
  --output /tmp/slot_aware_expedition_scores.json \
  --jsonl-output /tmp/slot_aware_expedition_scores.jsonl
```

여러 checkpoint를 한 번에 넘길 수 있다.

```bash
uv run python experiments/lost_cities/expedition_score_diagnostic/diagnose.py \
  --checkpoint checkpoints/run_a/latest.pt \
  --checkpoint checkpoints/run_b/latest.pt \
  --output /tmp/expedition_scores.json
```

slot-aware 실험의 `analyze.py`에 diagnostic plot을 함께 붙일 수 있다.

```bash
uv run python experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability/analyze.py \
  --run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability \
  --baseline-run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability \
  --expedition-diagnostic-json /tmp/slot_aware_expedition_scores.json \
  --expedition-plot-output /tmp/slot_aware_expedition_scores.png
```

## 출력

`rows`는 checkpoint x opponent 단위이며, 각 row는 `checkpoint_iteration`, `checkpoint_path`, `opponent`, `games`, `seed`를 포함한다. 이 값들로 기존 `metrics.jsonl`의 eval row와 후처리 join할 수 있다.

핵심 metric 그룹:

- per-game count: `per_game_positive_expeditions`, `per_game_negative_expeditions`, `per_game_breakeven_expeditions`, `per_game_bonus_expeditions`, `per_game_below_minus_20_expeditions`
- rates: `positive_expedition_rate`, `negative_expedition_rate`, `bonus_expedition_rate`
- score distribution: `avg_final_score_per_opened_expedition`, `final_expedition_score_p25`, `final_expedition_score_median`, `final_expedition_score_p75`, `final_expedition_score_p90`, `positive_expedition_score_mean`, `negative_expedition_score_mean`
- calibration: `first_open_recoverable_score_mean_for_positive_final`, `first_open_recoverable_score_mean_for_negative_final`
