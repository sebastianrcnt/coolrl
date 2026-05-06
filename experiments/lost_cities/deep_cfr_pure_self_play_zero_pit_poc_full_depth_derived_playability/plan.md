# 실험 계획

## 가설

opening selectivity가 emerge하지 않은 dominant obstacle은 representation bottleneck이다.

정책이 색별 손익분기, 회수 가능 점수, playable/dead hand decomposition을 직접 입력으로 받으면, 같은 256x3 network와 pure self-play league에서도 over-opening 평형에서 벗어날 수 있다.

## 기준 실험과의 차이

기준은 `deep_cfr_pure_self_play_zero_pit_poc_full_depth`다. 변경 변수는 encoding뿐이다.

```yaml
encoding:
  derived_playability: true
```

network, optimization, traversal, evaluation, self-play league는 기준 실험과 동일하게 둔다.

## Derived Features

기본 encoding 뒤에 98 dim을 추가한다.

- 색별 19 dim x 5색
- 공통 3 dim

색별 feature:

- unopened 여부
- wager-only opened 여부
- 현재 expedition numeric sum, wager count, length, last numeric rank
- 손패의 해당 색 count, wager count
- playable numeric sum/count
- dead numeric sum/count
- recoverable margin no bonus
- recoverable score no bonus
- break-even까지 필요한 추가 numeric value
- discard top playable flag/value
- unknown remaining count
- bonus까지 필요한 카드 수

제외한 항목:

- `playable_remaining_count`: opponent hand information leakage 위험이 있어 넣지 않는다.
- `has_bonus_path`: `cards_needed_for_bonus == 0`으로 복원 가능하므로 encoding에는 넣지 않는다. 단, `bad_open_rate` 판정용 derived predicate로는 사용한다.

따라서 색별 feature 수는 다음과 같다.

```text
열림 상태 6
손패 분해 6
회수성 3
외부 정보 3
보너스 경계 1
= 19
```

## 판정 기준

좋은 신호:

- safe 계열 `avg_opened_colors`가 4.8-4.9에서 4.2 이하로 하락
- safe 계열 `opened_colors_count_5`가 60-70% 이하로 하락
- `bad_open_rate`가 15% 이하
- safe avg_diff가 `full_depth` baseline 대비 후퇴하지 않음
- random avg_diff가 +40 전후를 유지

부분 성공:

- opened mean 4.3-4.6
- 5-color frequency 70-85%
- bad_open_rate 15-30%
- safe avg_diff 후퇴가 작음

실패:

- opened mean 4.75 이상
- 5-color frequency 85% 이상
- bad_open_rate 개선 없음
- safe avg_diff가 개선되지 않거나 크게 후퇴

## 조기 점검

iter 300:

- cut이 아니라 확인 지점이다.
- opened mean, 5-color frequency, bad_open_rate 중 2개 이상이 움직이면 계속 본다.

iter 600:

- opened mean <= 4.65 또는 5-color frequency <= 80% 또는 bad_open_rate 감소 추세가 있으면 계속 볼 가치가 있다.
- safe avg_diff가 baseline 대비 20점 이상 후퇴하고 그 상태가 2회 연속 eval에서 유지되면 중단 후보다.

iter 900-1200:

- selectivity 신호가 없으면 representation summary만으로는 부족하다고 판정한다.

## 알려진 리스크

- 색별 summary는 action slot 수준의 play/discard 선택을 직접 표시하지 않는다. 부분 성공이면 slot-level derived features가 다음 후보가 된다.
- `has_bonus_path`는 encoding에서 제외했지만 `cards_needed_for_bonus == 0`으로 복원 가능하다.
- derived feature가 hand-conditioned selectivity를 돕더라도 pure self-play 평형이 계속 over-opening을 보상할 수 있다.
