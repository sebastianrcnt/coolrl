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
- 첫 본 run은 iteration 5 직후 `OSError: [Errno 24] Too many open files`로 중단됐다.
  - 원인 후보는 worker batch로 torch Tensor `state_dict`를 직접 전달하면서 PyTorch multiprocessing이 fd 기반 storage sharing을 사용한 것이다.
  - worker payload를 CPU numpy copy로 넘기고 worker 내부에서 tensor로 복원하도록 수정했다.
  - `slot_aware_playability_fd_smoke8`이 iteration 8까지 통과해 기존 실패 지점을 넘었다.
- 본 run 재시작.
  - 재시작 시각: 2026-05-06 16:44 KST
  - 실행 세션: `tmux` session `slot_aware_playability`
  - checkpoint는 초기화 후 다시 생성했다.

### Iteration 115 중간 점검

- health는 정상이다.
  - latest/eval iteration: 115
  - node cutoff 0%, terminal 100%
  - avg endpoint depth 259.4, max depth 370
- score는 `derived_playability`보다 명확히 좋다.
  - safe 계열 평균 avg_diff: -37.91
  - `full_depth_derived_playability` baseline 대비 +11.40
  - random avg_diff: +36.79
- selectivity는 아직 약하다.
  - safe 계열 평균 opened colors: 4.95
  - 5-color count: 97.3/100
  - bad_open_rate: 85.8%
- 해석:
  - slot-level feature는 recovery/general play quality를 강하게 개선한 것으로 보인다.
  - bad_open_rate는 초반 88%대에서 85%대로 내려와 방향은 맞지만, opened colors가 4.95에 머물러 아직 entry-threshold selectivity emergence로 보기는 어렵다.
  - `derived_playability`와 달리 score 우위가 뚜렷하므로 조기 종료하지 않고 iteration 200까지 진행해 재판정한다.
- 재판정 기준:
  - iteration 150: health와 trend 확인
  - iteration 200: bad_open_rate 0.80 이하 또는 opened colors 4.85 이하이면 강한 신호
  - bad_open_rate 0.80-0.85면 부분 성공 진행 중으로 보고 iteration 300까지 계속 관찰
  - bad_open_rate 0.85 초과와 opened colors 4.95+가 유지되면 selectivity는 실패 쪽이지만, score 우위가 유지되면 iteration 300까지는 확인할 가치가 있다.
