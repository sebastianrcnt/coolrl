# Lost Cities Deep CFR Encoding Selectivity Review

## 질문

평가 기준은 `selectivity` 학습 관점에서 input이 충분히 표현력 있는가다.

1. hand가 어떻게 인코딩되는가?
2. 색별 expected expedition value, high-card count, wager card count 같은 selectivity 의사결정에 직접 도움 되는 feature가 있는가?
3. opponent expedition state, discard pile, deck remaining이 명시적으로 표현되는가?
4. input dimension은 얼마이고, 256x3 network가 hand-conditional selectivity를 표현하기에 충분해 보이는가?

## 요약 결론

현재 encoding은 raw public/private state를 꽤 충실하게 담지만, expedition opening selectivity를 쉽게 배우게 하는 색별 derived feature는 거의 없다.

정보가 완전히 없는 것은 아니다. hand, 양쪽 expedition, discard pile, public counts, deck remaining ratio가 모두 들어간다. 따라서 이론적으로는 256x3 MLP가 selectivity를 표현할 수 있다. 하지만 색별 value/risk 계산과 색 간 비교를 flat vector에서 직접 배워야 하므로 sample-inefficient하고, 현재 full-depth 실험의 over-opening pattern과 잘 맞는 약점이다.

## 현재 encoding 구조

대상 파일:

```text
src/coolrl/lost_cities/deep_cfr/encoding.py
```

tier3 기준 input dimension은 1500이다.

구성:

| block | dimension | 내용 |
| --- | ---: | --- |
| phase/current/player | 4 | phase one-hot, current-player match, player id |
| hand slots | 408 | 8 slots x 51. 카드 타입 50 + empty 1 |
| expeditions | 520 | 양쪽 player x 5 colors x 52. 색별 card counts, length, last rank |
| discard piles | 510 | 5 colors x 102. pile card counts, length, top card |
| public counts | 50 | 양쪽 expedition + discard에 공개된 카드 타입별 count |
| deck/turn | 2 | deck remaining ratio, turn count ratio |
| pending discard | 6 | pending discarded color or none |

## 1. hand 인코딩

hand는 slot별 one-hot이다.

```text
hand_size * (card_type_size + empty)
= 8 * (50 + 1)
= 408
```

각 slot은 특정 카드 타입 하나 또는 empty를 나타낸다. aggregated multi-hot은 아니며 slot order를 유지한다.

한계:

- 색별 hand count가 직접 없다.
- 색별 numeric sum이 직접 없다.
- 색별 high-card count가 직접 없다.
- 색별 wager/handshake count가 직접 없다.
- 현재 expedition의 last rank 이후에 낼 수 있는 playable hand count가 직접 없다.

즉 네트워크가 slot별 one-hot에서 색별 요약을 직접 합성해야 한다.

## 2. selectivity 의사결정 feature

다음 feature는 명시적으로 없다.

- 색별 expected expedition value
- 색별 hand numeric sum
- 색별 high-card count
- 색별 wager card count
- 색별 playable hand count
- 색별 playable hand sum
- 색별 current expedition score
- 색별 break-even margin
- 색별 committed penalty
- 색별 unseen useful cards estimate
- 색별 open desirability / risk proxy

selectivity에 필요한 계산은 대략 다음 형태다.

```text
open_value(color)
  ~= -20
     + 현재/손패/공개정보로 회수 가능한 value
     + future draw 기대값
     - 상대에게 주는 discard risk
```

현재 encoding은 이 값을 만들 raw 재료는 일부 제공하지만, 위 값을 직접 feature로 제공하지 않는다.

## 3. opponent / discard / deck 표현

명시적으로 표현되는 것:

- 내 expedition과 상대 expedition
  - 색별 card counts
  - expedition length
  - last numeric rank
- discard pile
  - 색별 card counts
  - pile length
  - top card one-hot
- public counts
  - 양쪽 expedition과 discard에 공개된 카드 타입별 count
- deck remaining ratio
- turn count ratio
- pending discarded color

한계:

- deck composition은 직접 표현되지 않는다.
- unseen card count by color는 public counts와 hand에서 간접 추론해야 한다.
- unseen useful cards above current rank도 직접 없다.
- 상대 hand belief는 없다. 이는 imperfect-information game이라 완전 정보는 불가능하지만, belief proxy feature는 추가 가능하다.

## 4. input dimension과 256x3 capacity 판단

tier3 input dimension은 1500이다.

현재 network는 `hidden_size=256`, `num_layers=3`이다. 이론적으로는 1500차원 raw vector에서 색별 selectivity 함수를 표현할 수 있다. 그러나 inductive bias가 약하다.

selectivity는 다음을 동시에 요구한다.

- 색별 hand potential 계산
- 현재 expedition 상태와 hand의 결합
- discard pile / deck remaining / turn count와의 결합
- 색 간 비교
- 몇 색을 열 것인지에 대한 count-level 조절

flat MLP는 이 구조를 명시적으로 공유하지 않는다. 색별로 같은 계산을 반복하고 비교하는 구조가 없기 때문에, 학습은 가능하더라도 데이터와 안정성이 많이 필요하다.

따라서 현재 결과는 network가 전혀 표현하지 못한다기보다는, representation과 architecture가 selectivity를 sample-efficient하게 만들지 못한다는 해석이 더 적절하다.

## 개선 후보

우선순위 높은 feature:

- `opened_colors_count`
- `unopened_colors_count`
- `total_committed_penalty`
- 색별 `hand_count`
- 색별 `hand_numeric_sum`
- 색별 `hand_high_card_count`
- 색별 `hand_wager_count`
- 색별 `playable_hand_count`
- 색별 `playable_hand_sum`
- 색별 `current_expedition_score`
- 색별 `break_even_margin`
- 색별 `last_rank_gap`
- 색별 `unseen_card_count`
- 색별 `unseen_useful_card_count`
- 색별 discard top availability / pile value proxy

architecture 후보:

- 색별 shared encoder 후 color pooling/comparison
- color-wise logits auxiliary head
- opened-colors count regularization 또는 auxiliary prediction

## 후속 실험 우선순위

바로 encoding을 바꾸기보다 anchor opponent 실험을 먼저 수행한다.

이유:

- full-depth 실험은 recovery skill이 self-play로 emerge할 수 있음을 보였다.
- safe heuristic diagnostic은 safe 자체가 평균 약 3.7색을 여는 selective policy임을 보였다.
- 따라서 먼저 league에 safe 계열 anchor를 주입해 selectivity가 유도 가능한지 직접 테스트하는 것이 결정적이다.

anchor 실험에서도 over-opening이 유지되면, 그 다음 원인 후보로 network capacity와 encoding feature engineering을 ablation한다.

권장 순서:

1. anchor opponent 주입: `safe_heuristic` weight 0.1-0.2
2. capacity ablation: 256x3 -> 512x5
3. encoding feature engineering: 색별 value/risk feature 추가

## 결론

현재 encoding은 충분한 raw information을 담고 있지만, selectivity에 필요한 색별 value/risk 계산을 직접 제공하지 않는다. 이론적 표현력은 있으나 학습 난이도가 높다.

따라서 `full_depth`의 over-opening은 우선 self-play league equilibrium 문제로 보고 anchor opponent 실험으로 검증하되, 그 다음 단계에서는 encoding과 architecture가 중요한 병목일 가능성이 높다.
