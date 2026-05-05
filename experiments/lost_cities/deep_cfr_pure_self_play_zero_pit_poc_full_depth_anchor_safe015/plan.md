# 실험 계획

## 가설

`full_depth` 실험은 truncation bias를 제거하면 recovery skill이 self-play로 emerge한다는 점을 보였다. 그러나 opening selectivity는 emerge하지 않았다. 정책은 safe 계열 상대에게 평균 4.8-4.9색을 열었고, 5색 opening 빈도도 90% 이상이었다.

이번 실험의 가설은 다음과 같다.

`safe_heuristic` anchor를 self-play league에 작은 비율로 섞으면 self-mirror 평형이 깨지고, 정책이 safe 계열 상대에게 punish 당하지 않기 위해 opening selectivity를 학습할 수 있다.

## 기준 실험과의 차이

기준은 `deep_cfr_pure_self_play_zero_pit_poc_full_depth`다. 변경 변수는 league 구조뿐이다.

```yaml
self_play_league:
  current_weight: 0.425
  recent_weight: 0.255
  older_weight: 0.170
  anchor_weight: 0.150
  anchor_policy: safe_heuristic
```

기존 `current:recent:older = 0.5:0.3:0.2` 비율은 보존하되 전체 self-play mass를 0.85로 줄이고, 나머지 0.15를 deterministic `safe_heuristic` anchor에 배정한다.

## 운영 계획

- `max_hours: 4`
- 1-iteration smoke 후 본 run 시작
- smoke sanity check:
  - `league_anchor_traversal_rate`가 대략 0.15 근처인지 확인한다.
  - `node_limit_cutoff_traversal_rate`가 0% 근처인지 확인한다.
  - eval metric에 `opened_colors_std`, `opened_colors_count_5`가 기록되는지 확인한다.
  - anchor 자체의 기준 행동은 `safe_heuristic_behavior_diagnostic` 결과를 사용한다. safe heuristic은 같은 calibre 상대에게 평균 약 3.7색을 연다.

## 판정 기준

좋은 신호:

- safe 계열 `avg_diff`가 `full_depth` reference인 약 -39보다 개선된다.
- safe 계열 `avg_opened_colors`가 4.8-4.9에서 3.5-4.2 쪽으로 내려간다.
- safe 계열 `opened_colors_count_5` 비율이 90%+에서 의미 있게 하락한다.
- random avg_diff가 `full_depth` reference인 +40 전후에서 크게 무너지지 않는다.

가장 결정적인 신호:

```text
safe avg_diff 유지/개선 + opened_colors 감소 + 5-color frequency 감소
```

나쁜 신호:

- safe 점수만 좋아지고 opened colors는 계속 4.8+에 머문다. anchor가 recovery만 강화하고 selectivity를 못 유도한 것이다.
- opened colors는 줄지만 safe/random 점수가 붕괴한다. selective pressure는 생겼지만 recovery skill을 잃은 것이다.
- random avg_diff가 +20 이하로 크게 후퇴한다. anchor opponent에 specialization된 overfit 가능성이 크다.
- anchor traversal rate가 config와 크게 다르다. anchor mechanism 자체가 의도대로 동작하지 않은 것이다.

## 후속 판단

selectivity가 emerge하면 league 구조가 주요 obstacle이었다는 증거다.

selectivity가 emerge하지 않으면 다음 obstacle 후보는 encoding 또는 capacity다. 이 경우 `deep_cfr_encoding_selectivity_review`의 feature/capacity ablation으로 넘어간다.
