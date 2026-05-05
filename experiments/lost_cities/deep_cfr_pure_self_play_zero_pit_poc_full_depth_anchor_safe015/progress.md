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
