# 실험 설계: slot_aware_playability

## 가설

`derived_playability` 실패의 핵심 원인은 flat MLP가 색별 summary와 hand slot one-hot/action 사이의 conditional lookup을 안정적으로 학습하지 못한 것이다.

색별 회수 가능성 정보는 있었지만, 정책이 그 정보를 "이 슬롯의 play action은 나쁜 opening이다"로 직접 연결하지 못했다. slot-level action-local features를 넣으면 selectivity의 sufficient statistic이 표현 공간에 직접 들어간다.

## 기준 실험과의 차이

기준은 `full_depth_derived_playability`다.

변경은 encoding뿐이다.

```yaml
encoding:
  derived_playability: true
  slot_aware_playability: true
```

network capacity, traversal, optimization, evaluation, self-play league는 exp3와 동일하다.

## 추가 feature

기존 exp3의 색별 derived block은 유지한다.

추가로 hand slot마다 12개 feature를 붙인다. tier3 hand size 8 기준 총 96 dim이며, input dim은 1598에서 1694로 증가한다.

slot feature 순서:

1. `slot_card_color_recoverable_score_visible`
2. `slot_card_color_break_even_margin_visible`
3. `slot_card_would_start_color_commitment`
4. `slot_card_is_numeric_open`
5. `slot_card_is_wager_first_open`
6. `slot_card_is_playable_to_existing`
7. `slot_card_is_dead_numeric`
8. `slot_card_is_wager_before_numeric`
9. `slot_card_color_has_bonus_path`
10. `slot_card_is_bad_open_candidate`
11. `slot_card_open_risk_score`
12. `slot_card_is_safe_continuation`

`would_start_color_commitment`는 완전 빈 색에 처음 내는 play뿐 아니라 wager만 깔린 색에 첫 numeric을 내는 경우도 포함한다. 이 실험에서 중요한 entry-threshold decision은 "색을 아예 건드리는가"와 "wager-only 색을 numeric으로 확정하는가"를 모두 포함하기 때문이다.

`recoverable_score_visible`은 current expedition + playable hand만 사용하는 conservative 기준이다. unknown remaining이나 discard pile의 미래 회수 기대는 넣지 않는다.

## 판정 기준

핵심 metric:

- `bad_open_rate`
- `avg_opened_colors`
- `opened_colors_count_5`
- safe 계열 `avg_diff`

수동 모니터링 기준:

- iter 100: bad_open_rate가 초기 대비 10%p 이상 감소하면 좋은 초기 신호
- iter 150: 15%p 이상 감소하지 않으면 실패 가능성이 높음
- iter 200: 25%p 이상 감소하거나 opened colors가 4.85 이하이면 iter 300까지 볼 가치가 큼

성공:

- safe 계열 opened colors가 4.6 이하로 하락
- 5-color frequency가 75% 이하로 하락
- bad_open_rate가 25% 이하로 하락
- safe avg_diff가 exp3 대비 후퇴하지 않음

부분 성공:

- opened colors 4.6-4.85
- 5-color frequency가 75-90%
- bad_open_rate가 명확한 하락 추세

실패:

- opened colors 4.9 이상 plateau
- 5-color frequency 90% 이상
- bad_open_rate 80% 이상 유지

## 알려진 리스크

- slot feature가 opening risk를 직접 주더라도 self-play incentive가 bad open을 충분히 벌하지 않으면 selectivity가 emerge하지 않을 수 있다.
- feature가 play action 쪽에만 붙고 discard action에는 직접 붙지 않으므로, discard 선택의 정밀한 selectivity는 아직 부족할 수 있다.
- exp3보다 input dim이 96 증가하지만 network는 256x3 그대로라 capacity margin이 작을 수 있다.
