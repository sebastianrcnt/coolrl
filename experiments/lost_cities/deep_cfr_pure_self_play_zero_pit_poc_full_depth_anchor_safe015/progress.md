# 진행 기록

## 2026-05-06

- 실험 record를 생성했다.
- 기준 실험은 `deep_cfr_pure_self_play_zero_pit_poc_full_depth`다.
- 변경 변수는 self-play league에 `safe_heuristic` anchor weight 0.15를 추가하는 것이다.
- smoke sanity check를 먼저 수행한 뒤 4시간 run을 시작한다.

### Smoke

1-iteration smoke:

- `league_anchor_traversal_rate`: 0.25
- `league_current_traversals`: 105
- `league_anchor_traversals`: 35
- `league_recent_traversals`, `league_older_traversals`: 0
- `node_limit_cutoff_traversal_rate`: 0.0
- `terminal_traversal_rate`: 1.0

해석:

- 첫 iteration은 snapshot이 아직 없어 recent/older bucket이 비활성화된다. 그래서 anchor 비율이 목표 0.15보다 높게 튈 수 있다.
- cutoff 없이 모든 traversal이 terminal에 도달했다.

5-iteration smoke:

- iter 5 `league_anchor_traversal_rate`: 0.1571
- iter 5 `league_current_traversals`: 72
- iter 5 `league_recent_traversals`: 46
- iter 5 `league_anchor_traversals`: 22
- iter 5 `league_older_traversals`: 0
- iter 5 `node_limit_cutoff_traversal_rate`: 0.0
- iter 5 `terminal_traversal_rate`: 1.0
- iter 5 safe avg_diff: -74.63
- iter 5 random avg_diff: -20.06
- iter 5 safe opened colors mean/std/count_5: 2.35 / 1.47 / 4

해석:

- snapshot이 생긴 뒤 anchor sampling은 목표 0.15와 거의 일치한다.
- `opened_colors_std`와 `opened_colors_count_5`가 eval metric에 정상 기록된다.
- smoke의 점수와 timeout은 초기 정책 상태라 판정 대상이 아니다. 여기서는 anchor mechanism, metric schema, cutoff 여부만 확인한다.
- anchor 자체 행동의 selectivity 기준은 별도 `safe_heuristic` 진단 결과를 사용한다. safe heuristic은 같은 calibre 상대에게 평균 약 3.7색을 열었다.

### 코드 검증

- `uv run pytest src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py src/coolrl/lost_cities/tests/test_deep_cfr_encoding.py -q`
- 결과: 58 passed

### 본 run 종료

- 시작: 2026-05-06 05:50:44 KST
- 종료: 2026-05-06 09:51:00 KST
- train exit status: 0
- analyze exit status: 0
- 최신 row iteration: 1219
- 최신 eval iteration: 1215
- report/plot 생성 완료

Traversal health:

- `node_limit_cutoff_traversal_rate`: 0.0%
- `terminal_traversal_rate`: 100.0%
- 최신 `league_anchor_traversal_rate`: 16.4%
- 최근 50 iteration 평균 `league_anchor_traversal_rate`: 15.7%
- avg endpoint depth: 258.6
- max depth reached: 404

최신 eval 핵심 관측:

- safe avg_diff: -56.96
- safe opened colors mean/std/count_5: 4.81 / 0.46 / 84
- safe_loose avg_diff: -60.55
- safe_loose opened colors mean/std/count_5: 4.86 / 0.40 / 88
- safe_strict avg_diff: -53.83
- safe_strict opened colors mean/std/count_5: 4.83 / 0.45 / 86
- safe 계열 평균 avg_diff: -57.11
- safe 계열 평균 opened colors: 4.83
- random avg_diff: +48.31

판정:

- anchor sampling과 metric 기록은 정상 동작했다.
- `safe_heuristic` anchor weight 0.15는 opening selectivity를 유도하지 못했다.
- 정책은 여전히 safe 계열 상대에게 평균 4.8색 이상을 열고, 5색 opening도 높은 빈도로 유지했다.
- safe 계열 점수는 `full_depth` baseline보다 약 18점 나빠졌다.
- random avg_diff는 +48.31로 후퇴하지 않았다. 따라서 이번 실패는 safe heuristic 분포에 과적합되어 random이 무너진 현상이 아니다.
- 현재 결론은 anchor pressure 0.15가 over-opening self-play 평형을 깨기에는 부족했거나, 단순 deterministic safe anchor 주입만으로는 selectivity 신호가 충분히 강하지 않았다는 것이다.
