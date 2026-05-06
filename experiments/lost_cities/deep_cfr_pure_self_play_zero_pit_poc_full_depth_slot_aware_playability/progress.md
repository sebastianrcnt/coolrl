# 진행 기록

## 2026-05-06

- 구현 sanity check 통과.
  - `uv run pytest src/coolrl/lost_cities/tests/test_deep_cfr_encoding.py src/coolrl/lost_cities/tests/test_deep_cfr_config.py src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py -q`
  - 결과: 90 passed.
  - input dim: base 1500, derived 1598, slot-aware 1694.
- smoke run 완료.
  - checkpoint: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability_smoke5`
  - iteration: 5
  - traversal health: node cutoff 0%, terminal 100%, avg endpoint depth 317.2, max depth 506.
  - safe 계열 평균: avg_diff -80.05, opened colors 2.14, 5-color count 6/100, bad_open_rate 85.5%.
  - random avg_diff +6.66.
  - 해석: 새 input dim과 evaluation metric 경로는 정상 동작한다. iteration 5 값은 초기 정책 확인용이며 selectivity 판정에는 사용하지 않는다.
- 4시간 본 run을 시작할 예정이다.
- 본 run 시작.
  - 시작 시각: 2026-05-06 16:32 KST
  - 실행 세션: `tmux` session `slot_aware_playability`
  - checkpoint: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability`
  - tail: `tail -f checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability/train.log`
  - 시작 로그 확인: input_dim 1694, effective_workers 8, node cutoff 0%.
