# Kuhn Poker로 배우는 CFR

이 문서는 게임 이론과 강화학습을 처음 접하는 사람을 위해 쓰여졌어요. Kuhn Poker 룰은 이미 안다고 가정합니다 (J/Q/K 세 장, 양쪽 1칩 ante, 체크/벳, 폴드/콜로 진행되는 미니 포커).

목표는 두 가지예요. 첫째, "AI가 포커 같은 불완전 정보 게임을 어떻게 풀 수 있는가"의 핵심 알고리즘인 CFR을 이해하기. 둘째, 그걸 직접 구현하고 검증할 수 있는 수준까지 가기.

수학 기호는 최소화하고, 직관과 비유를 우선합니다.

## 1. 왜 포커는 어려운가

오목이나 체스 같은 게임은 컴퓨터가 쉽게 풀어요. 모든 정보가 보드 위에 다 있으니까요. AlphaZero 같은 알고리즘이 잘 작동하는 이유예요.

포커는 다릅니다. 상대 카드가 안 보여요. "지금 최선의 행동이 뭐냐"가 상대 카드에 따라 달라지는데, 나는 그걸 모르죠.

그래서 포커의 좋은 전략은 **확률적**이어야 해요. K를 들어도 가끔 체크해야 하고, J를 들어도 가끔 벳해야 해요. 항상 K로 벳하면 상대가 "벳 = 강한 카드"로 학습해버리니까요. 강한 카드의 가치를 보호하려면 약한 카드로도 가끔 블러프해야 해요.

이런 확률적 최적 전략을 **mixed strategy nash equilibrium**이라고 불러요. CFR은 이 균형을 찾아주는 알고리즘이에요.

## 2. Regret이라는 개념

CFR을 이해하려면 먼저 "regret (후회)" 개념을 잡아야 해요.

가위바위보 100판을 친구랑 했다고 해봐요. 매번 바위만 냈고 다 졌어요. 친구가 다 보를 냈거든요. 복기하면서 이런 생각이 들어요.

- "보를 냈더라면 비겼겠지" → 결과 0
- "가위를 냈더라면 이겼겠지" → 결과 +1
- 실제로 한 행동 (바위) → 결과 -1

각 행동의 regret은 이렇게 정의해요.

```
행동 X의 regret = X를 했더라면의 결과 - 실제로 한 결과
```

위 예시에서:

- 보의 regret = 0 - (-1) = +1
- 가위의 regret = +1 - (-1) = +2
- 바위의 regret = -1 - (-1) = 0

**Regret이 크다 = 그 행동을 안 해서 손해를 봤다는 뜻**이에요. 가위를 안 내서 가장 손해 봤네요.

## 3. Regret Matching

이제 후회를 행동으로 옮길 차례예요. CFR의 첫 번째 핵심 규칙.

```
다음에 각 행동을 할 확률 = 그 행동의 regret / 모든 양수 regret의 합
```

100판 후 누적 regret이 가위 +200, 보 +100, 바위 0이라면:

- 다음 가위 낼 확률 = 200 / 300 ≈ 67%
- 다음 보 낼 확률 = 100 / 300 ≈ 33%
- 다음 바위 낼 확률 = 0 / 300 = 0%

후회가 큰 행동의 확률을 키우고, 후회 없는 행동은 안 함. 모든 regret이 0이거나 음수면 그냥 균등 확률 (가위바위보 1/3씩).

이걸 양쪽이 다 하면서 무한히 반복하면, 양쪽 모두 가위바위보를 1/3씩 내는 균형에 도달해요. 이게 nash equilibrium이고, **CFR은 이 균형에 수렴한다는 게 수학적으로 증명**되어 있어요.

## 4. Information Set

가위바위보는 단순해요. 한 번에 끝나니까. 그런데 Kuhn Poker는 여러 단계가 있어요.

- 카드를 받음
- 액션 선택 (체크/벳)
- 상대 액션 봄
- 또 액션 선택할 수도 있음
- 쇼다운

매 단계마다 "내가 보는 정보"가 달라요. K 받은 직후 vs J 받고 상대가 벳한 후 vs Q 받고 상대가 체크한 후. 다 다른 의사결정 상황이에요.

이렇게 **"내가 지금 보고 있는 정보의 단위"** 를 information set이라고 해요. Kuhn Poker는 information set이 정확히 12개예요.

### Player 1 (선플레이어)의 6개 infoset

처음 액션할 때 (history가 빈 상태):
- `J:` - J 받고 첫 액션
- `Q:` - Q 받고 첫 액션
- `K:` - K 받고 첫 액션

체크했는데 상대가 벳해서 다시 결정해야 할 때 (history `pb`):
- `J:pb`, `Q:pb`, `K:pb`

### Player 2 (후플레이어)의 6개 infoset

상대가 체크한 후 (history `p`):
- `J:p`, `Q:p`, `K:p`

상대가 벳한 후 (history `b`):
- `J:b`, `Q:b`, `K:b`

각 infoset에서 가능한 액션은 항상 두 개. `p` (체크 또는 폴드) 와 `b` (벳 또는 콜).

**중요한 점**: CFR은 각 information set마다 따로따로 regret을 누적해요. "K 받고 첫 액션" 상황의 regret 따로, "J 받고 상대 벳한 후" 상황의 regret 따로. 각자 학습.

## 5. Counterfactual의 의미

CFR의 첫 글자 C가 등장. **Counterfactual = 반사실적**.

가위바위보에서 "가위 냈더라면"은 단순했어요. 한 번에 끝나니까. 그런데 카드 게임은 내가 어떤 행동을 했냐에 따라 게임의 미래가 통째로 바뀌어요.

"K 받고 체크했더라면" vs "K 받고 벳했더라면"은 그 이후 상대 반응, 다음 내 반응 등 다른 미래를 만들어요.

CFR은 게임 트리를 따라 모든 가능한 미래를 계산해서, "이 information set에서 이 행동을 했더라면 평균적으로 결과가 어땠을까"를 정확히 구해요. 이게 **counterfactual value**예요.

여기서 미묘한 부분이 있어요. Regret을 update할 때 가중치가 들어가요.

```
infoset I에서 액션 a의 regret 증가량 = (상대가 자기 strategy로 I에 도달할 확률) × (a의 가치 - 노드 평균 가치)
```

왜 "상대 reach"를 곱하냐면, **내 행동의 영향을 빼고 본 reach 확률**이 counterfactual의 정확한 의미이기 때문이에요.

내가 K 받았을 때 의사결정은, 다른 카드 받았더라도 K를 받는다면 똑같이 했을 거예요 (information set이 같으니). 그래서 "내가 이 노드까지 도달한 확률"은 빼고, "상대가 자기 strategy로 여기 도달한 확률"만 가중치로 씀.

처음에는 직관에 안 맞아요. "왜 상대 확률을 곱하지?" 싶죠. 차근차근 곱씹으면 이해돼요. 핵심은 **"이 infoset에 도달한 모든 가능한 게임 상황 중에서, 상대 행동의 영향만큼 가중평균"** 을 낸다는 점이에요.

## 6. Tabular CFR 알고리즘

이제 알고리즘 전체. Kuhn Poker는 작아서 모든 information set을 딕셔너리에 저장할 수 있어요. 이걸 **tabular CFR**이라고 해요.

### 데이터 구조

각 infoset마다 두 개의 누적 테이블.

```python
regret_sum = {}      # {infoset: [pass_regret, bet_regret]}
strategy_sum = {}    # {infoset: [pass_strategy_누적, bet_strategy_누적]}
```

`regret_sum`은 다음 strategy 계산용. `strategy_sum`은 평균 strategy 계산용. **둘 다 필요해요.** CFR의 균형 strategy는 마지막 iteration의 strategy가 아니라 **모든 iteration의 평균**이거든요.

### 핵심 함수 세 개

#### `get_strategy(infoset)`

현재 누적 regret으로부터 현재 iteration의 strategy 계산. Regret matching.

```python
def get_strategy(infoset):
    regrets = regret_sum[infoset]
    positive = [max(r, 0) for r in regrets]
    s = sum(positive)
    if s > 0:
        return [r/s for r in positive]
    return [0.5, 0.5]  # 모두 0 이하면 균등
```

#### `cfr(history, p1_card, p2_card, reach_p1, reach_p2)`

게임 트리를 재귀로 traverse하는 핵심 함수. 각 노드에서:

1. Terminal이면 보상 반환
2. Player 노드면:
   - 현재 strategy 계산
   - 각 액션의 가치 = 그 액션 따라간 후 재귀호출 결과
   - 노드 가치 = strategy 가중평균
   - **Regret update**: 각 액션의 regret += 상대 reach × (액션 가치 - 노드 가치)
   - **Strategy update**: strategy_sum += 자기 reach × strategy
   - 노드 가치 반환

`reach_p1`, `reach_p2`는 "각 플레이어가 자기 strategy로 이 노드까지 도달할 확률"을 추적. CFR의 핵심 트릭이에요.

#### Main loop

```python
for iteration in range(100_000):
    for p1_card in [J, Q, K]:
        for p2_card in [J, Q, K]:
            if p1_card != p2_card:
                cfr("", p1_card, p2_card, 1.0, 1.0)

# 학습 끝나면 평균 strategy 추출
average_strategy = {
    infoset: normalize(strategy_sum[infoset])
    for infoset in strategy_sum
}
```

100,000번 iteration 돌리면 신기한 일이 일어나요. 각 infoset의 평균 strategy가 nash equilibrium에 수렴해요.

## 7. Kuhn의 알려진 균형

Kuhn Poker는 작아서 균형이 수학적으로 풀려 있어요. 검증할 때 비교 기준이 됩니다.

### Player 1 균형

| Infoset | Pass 확률 | Bet 확률 |
|---|---|---|
| `J:` (J 받고 첫 액션) | 1 - α | α |
| `Q:` (Q 받고 첫 액션) | 1 | 0 |
| `K:` (K 받고 첫 액션) | 1 - 3α | 3α |

여기서 α는 0과 1/3 사이 자유로운 값. 균형이 여러 개 있어요.

폴드/콜 결정 (체크 후 상대가 벳한 상황):

| Infoset | Fold 확률 | Call 확률 |
|---|---|---|
| `J:pb` | 1 | 0 |
| `Q:pb` | 2/3 | 1/3 |
| `K:pb` | 0 | 1 |

### Player 2 균형

상대가 체크한 후:

| Infoset | Check 확률 | Bet 확률 |
|---|---|---|
| `J:p` | 2/3 | 1/3 |
| `Q:p` | 1 | 0 |
| `K:p` | 0 | 1 |

상대가 벳한 후 (폴드/콜):

| Infoset | Fold 확률 | Call 확률 |
|---|---|---|
| `J:b` | 1 | 0 |
| `Q:b` | 2/3 | 1/3 |
| `K:b` | 0 | 1 |

### 균형의 EV

양쪽 다 균형 strategy를 둘 때, P1이 라운드당 평균 **-1/18 ≈ -0.0556** 잃어요. P2가 살짝 유리. 후포지션의 정보 우위 때문이에요.

이 -0.0556이 첫 번째 검증 기준이에요. 학습 끝난 후 평균 strategy의 EV가 이 값에 가까우면 균형 도달.

## 8. Exploitability: 진짜 검증 기준

EV만으로는 부족해요. 우연히 EV가 맞을 수도 있고, 더 큰 게임에서는 진짜 균형 EV를 모르기도 하니까요.

**Exploitability**가 더 본질적인 검증 메트릭이에요.

### 정의

> 내 strategy가 고정되어 있을 때, 상대가 best response를 두면 얼마나 나를 이길 수 있나?

이게 양쪽 모두 0에 가까우면 진짜 균형이에요. Nash equilibrium의 정의 자체가 "어느 쪽도 일방적으로 strategy 바꿔서 이득 못 봄"이거든요.

```
exploitability(strategy) = (BR_value(strategy, P1) + BR_value(strategy, P2)) / 2
```

여기서 `BR_value(strategy, X)`는 "X가 best response 두고 상대는 strategy 그대로 둘 때 X가 얻는 EV". 균형 strategy면 두 BR 값이 합쳐서 0이 나와야 해요 (zero-sum 게임이니까).

### Best Response 계산의 함정

Best response를 짤 때 **가장 흔한 버그**가 하나 있어요. 짚고 넘어갈 가치가 큰 디테일이에요.

순진하게 짜면 게임 트리를 재귀로 내려가면서 매 노드에서 max를 취해요. 그런데 이렇게 하면 **같은 information set에 속한 다른 노드들에서 다른 행동을 선택**하게 돼요.

예를 들어 K 받은 P2가 history `b` 시점이라고 해봐요. P1 카드는 J일 수도 Q일 수도 있어요 (P2는 모름). 순진한 BR은:

- P1이 J인 경우의 노드에서: "J 상대로는 b가 좋다"
- P1이 Q인 경우의 노드에서: "Q 상대로는 p가 좋다"

이렇게 별개로 max 취하면, P2가 "P1 카드 봐서 그에 맞춰 다르게 행동"하는 cheating BR이 돼요. 실제로는 P2는 P1 카드 못 보고 같은 information set에서 같은 결정을 해야 해요.

### 올바른 BR 계산법

Information set 단위로 단 하나의 결정론적 액션을 선택해야 해요. Kuhn에서는 가장 단순한 방법이 **모든 가능한 deterministic policy 열거**예요.

BR player가 6개 infoset, 각 2개 액션이라면 2^6 = 64가지 가능한 policy. 각각의 EV를 계산해서 max.

```python
def best_response_value(strategy, br_player):
    best_ev = -infinity
    for policy in all_2^6_pure_policies:
        ev = compute_ev_with_policy(policy, strategy)
        best_ev = max(best_ev, ev)
    return best_ev
```

이 방법은 작은 게임에서 정확. Leduc Poker 이상으로 가면 information set이 너무 많아서 (~1000개) 2^1000개 policy 열거 불가능. 그때는 dynamic programming 기반의 다른 방법이 필요해요.

### 검증 기대치

Tabular CFR 100k iteration이면 exploitability < 0.005가 정상. 보통 0.001 이하 나와요.

Sanity check 두 가지:
- BR 값이 균형값보다 **크거나 같음** (BR은 최소한 같거나 더 좋아야 함)
- BR_p1 + BR_p2가 양수, 그리고 그 합이 정확히 2 × exploitability

## 9. 실제 구현 결과

100,000 iteration 돌린 후 우리 구현 결과:

```
Expected value of average strategy profile for P1: -0.055553
Best response value for P1: -0.054704
Best response value for P2:  0.055956
Exploitability of average strategy: 0.000626
```

분석:

- EV = -0.055553 vs 이론값 -0.055556. 오차 ~3e-6. 거의 완벽한 일치.
- BR for P1 = -0.054704: P1이 best response 둬도 EV가 -0.055556에서 -0.054704로 약 +0.000852만큼만 개선.
- BR for P2 = 0.055956: P2가 best response 둬도 EV가 +0.055556에서 +0.055956으로 약 +0.000400만큼만 개선.
- Exploitability = 0.000626. 양쪽 모두 거의 균형.

CFR 수렴 속도가 O(1/√T)라서 100k iteration에서 ~0.001 수준은 이론적으로 기대되는 값이에요.

## 10. 다음 단계: Deep CFR

Tabular CFR은 information set을 딕셔너리에 저장해요. Kuhn (12개), Leduc (1,000개)까지는 가능. 하지만 본격적인 게임은 information set이 너무 많아요.

- Heads-up Limit Hold'em: 약 10^14
- Lost Cities: 약 10^15

이 정도면 메모리에 못 들어가요. 그리고 큰 게임은 information set 간에 유사성이 있어요. "비슷한 핸드 + 비슷한 베팅 패턴"은 비슷한 의사결정이 필요해요. Tabular는 이 둘을 완전히 별개로 학습.

여기서 **신경망**이 들어와요. Tabular CFR의 딕셔너리를 신경망으로 대체.

```python
# Tabular
regret = regret_sum["K:"]  # 딕셔너리 lookup

# Deep
encoded = encode_infoset("K", "")  # 벡터로 변환
regret = regret_network(encoded)   # 신경망 예측
```

신경망의 두 가지 강점:

1. **메모리 효율**: 신경망은 가중치 몇 KB. 10^15 information set도 같은 신경망이 다룸.
2. **일반화**: 비슷한 입력에 비슷한 출력. 한 infoset에서 배운 게 비슷한 infoset으로 transfer.

핵심 알고리즘 (regret matching, counterfactual value, 누적 update)는 완전히 같아요. **저장 방식만 딕셔너리에서 신경망으로 바뀐 거예요.**

이게 Deep CFR이고, 본격적인 imperfect info 게임 풀이의 표준 도구예요. 다음 문서에서 다룹니다.

## 정리

지금까지 다룬 내용:

1. 포커는 mixed strategy 균형이 필요한 imperfect info 게임
2. Regret = 안 한 행동의 후회. Regret matching으로 다음 행동 확률 결정
3. Information set = 의사결정자가 보는 정보 단위. Kuhn은 12개
4. Counterfactual value = 게임 트리 traverse로 정확히 계산
5. Tabular CFR = information set을 딕셔너리에 저장하는 구현
6. Exploitability = 진짜 검증 기준. Best response가 얼마나 이득 보는지
7. Best response 짤 때 information set을 존중해야 함 (가장 흔한 버그)

이 흐름이 손에 잡히면 Deep CFR도 자연스럽게 이해돼요. 다음 문서에서 Deep CFR로 넘어갑니다.
