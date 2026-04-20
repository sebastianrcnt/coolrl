# Step 1: 게임 로직 구현

## 목표

Config parametrized Lost Cities 규칙 엔진과 플레이 가능한 Textual 기반 TUI를 구현한다. Tier 1 (3색, rank 2-6, handshake 1, 손패 5장) 기준으로 먼저 완성하되, 코드는 파라미터로 tier 0/2/3 어디든 확장 가능하게 짠다.

학습 코드는 이 단계 밖이다. 여기서는 **게임이 정확히 돌아가는지**와 **사람이 직접 플레이해볼 수 있는지**만 확인한다.

## 비목표

- DMC 학습 루프 (step 2)
- 네트워크 모델, replay buffer (step 2)
- Tier 2/3 학습 및 튜닝 (step 3 이후)
- 웹/네이티브 GUI (TUI로 충분)

## 전체 맥락

이 프로젝트의 최종 목표는 Lost Cities를 plays 가능한 AI 에이전트를 훈련하는 것이다. 전체 학습 전략은 다음과 같다.

- **알고리즘**: DMC (Deep Monte Carlo) self-play, DouZero 방식 기반
- **행동 분해**: 한 턴을 card-step (카드 선택 + play/discard)과 draw-step (draw source 선택) 두 단계로 분리
- **학습 타겟**: 에피소드 종료 시 정규화된 점수차를 각 step의 Q(s, a) 회귀 타겟으로 사용
- **스케일 전략**: tier 0에서 완전탐색 해와 비교하여 알고리즘 검증, tier 1에서 random/휴리스틱 추월 확인, 필요시 tier 3으로 확장

Step 1은 이 모든 학습 작업의 **기반 인프라**를 만드는 단계다. 환경에 버그가 있으면 위에 아무리 정교한 알고리즘을 얹어도 의미 없기 때문에 규칙 정확성과 테스트 커버리지를 최우선으로 둔다.

## 프로젝트 배치

레포의 기존 패턴 (`src/coolrl/{project}/`) 을 따른다.

```
src/coolrl/lost_cities/
├── README.md              # 프로젝트 개요
├── docs/
│   └── step1.md           # 이 문서
├── game.py                # 규칙 엔진 + config
├── env.py                 # 학습용 래퍼 (step 2에서 채움, 스켈레톤만)
├── bots.py                # random, safe heuristic
├── tui.py                 # Textual 기반 플레이 UI
├── __init__.py
└── tests/
    ├── test_rules.py
    ├── test_scoring.py
    ├── test_masks.py
    └── test_config.py

configs/
└── lost_cities_tier1.yaml
```

## 핵심 설계

### Config

`game.py` 안에 dataclass로 둔다.

```python
@dataclass
class LostCitiesConfig:
    n_colors: int = 3
    n_ranks: int = 5              # 숫자 카드 개수 (예: 2~6이면 5개)
    min_rank: int = 2             # 숫자 카드 최소값
    n_handshakes: int = 1         # 색당 handshake 카드 수
    hand_size: int = 5
    expedition_penalty: int = -20
    bonus_threshold: int = 8      # 원정 카드 수 이 이상이면 보너스
    bonus_amount: int = 20
    seed: int | None = None
```

Tier 프리셋:

- `tier0`: `(2, 3, 2, 0, 3)` - 6장 덱, sanity check용
- `tier1`: `(3, 5, 2, 1, 5)` - 18장 덱, 이번 단계 기준
- `tier2`: `(4, 7, 2, 2, 6)` - 36장 덱
- `tier3`: `(5, 9, 2, 3, 8)` - 60장 덱, 오리지널

Tier별 YAML은 `configs/` 아래에 둔다.

### Card 표현

```python
@dataclass(frozen=True)
class Card:
    color: int       # 0 ~ n_colors-1
    rank: int        # 0 = handshake, 1 ~ n_ranks = 숫자 카드 인덱스
                     # 실제 숫자값은 min_rank + rank - 1
```

핸드셰이크끼리는 서로 구분하지 않는다 (같은 색 handshake 3장은 모두 동일 취급). 덱 생성 시 색당 `n_handshakes + n_ranks` 장.

### Game state

```python
class GameState:
    config: LostCitiesConfig
    deck: list[Card]                      # 셔플된 덱, pop으로 draw
    hands: list[list[Card]]               # hands[0], hands[1]
    expeditions: list[list[list[Card]]]   # expeditions[player][color]
    discards: list[list[Card]]            # discards[color], 색별 pile
    current_player: int                   # 0 or 1
    phase: Literal["card", "draw"]
    pending_discarded_color: int | None   # 이번 턴에 방금 버린 색 (draw 제약용)
    turn_count: int
```

### 규칙 요약 (정확히 구현)

**Phase 1 card-step**: 손패에서 카드 한 장 선택 후 둘 중 하나

- **Play**: 해당 색 원정에 추가. 오름차순 제약 (원정에 마지막으로 놓인 숫자카드보다 큰 rank여야). Handshake는 원정에 숫자카드가 아직 없을 때만 가능.
- **Discard**: 해당 색 discard pile에 추가. 카드가 손패에 있으면 항상 가능.

**Phase 2 draw-step**: 손패를 채울 카드 한 장 draw

- **Deck draw**: 덱에서 top 한 장. 덱이 비어있으면 불가.
- **Discard draw**: 어느 색 discard pile top 한 장. 단 **이번 턴에 방금 내가 버린 색은 불가** (pending_discarded_color 체크). Pile이 비어있으면 불가.

**종료 조건**: 덱이 비는 순간 현재 턴을 완료하고 종료. 정확히는 deck draw가 가능하지 않게 되면 그 시점에 게임 종료 (상대 턴까지 마저 돌지 않음 - 오리지널 규칙 확인 필요, 일반적으로는 "덱에서 마지막 카드가 draw된 후 그 턴까지만 완료하고 종료").

**점수 계산 (원정당)**:

- 원정이 비어있으면 0점
- 아니면 `(숫자카드 rank 합 - penalty) × (handshake 수 + 1)` + (카드 수 ≥ bonus_threshold면 bonus_amount)
- Handshake만 있고 숫자카드 없는 원정: 숫자합이 0이라 `(0 - penalty) × multiplier` 가 되어 음수가 심화됨. 이 엣지 케이스 주의.

**승패**: 두 플레이어 총점의 차.

### Action space

두 phase 각각 독립 action space.

**Card-step**: 크기 `2 × hand_size`

- Slot `i`의 play: action id `2i`
- Slot `i`의 discard: action id `2i + 1`
- Hand slot 정렬: 색 우선, rank 오름차순, 고정
- 빈 slot은 legal mask에서 false

**Draw-step**: 크기 `1 + n_colors`

- Deck draw: action id `0`
- Color `c`의 discard draw: action id `1 + c`

### Legal mask 계산

**Card-step mask (길이 `2 × hand_size`)**:

- Slot `i`에 카드 없으면 둘 다 false
- Play legal iff: (해당 색 원정이 비어있음) OR (카드가 handshake이고 원정에 숫자카드 없음) OR (카드가 숫자이고 카드 rank > 원정 마지막 숫자카드 rank)
- Discard legal iff: 카드 있음

**Draw-step mask (길이 `1 + n_colors`)**:

- Deck draw: `len(deck) > 0`
- Color `c` discard draw: `len(discards[c]) > 0` AND `c != pending_discarded_color`

**Invariant**: 각 phase에서 legal mask에 최소 1개는 true. Card phase는 손패가 있으면 discard는 항상 가능하므로 자연 성립. Draw phase는 덱 또는 아무 discard pile이 남아있으면 성립. 덱 비고 모든 pile 비는 상황은 게임이 이미 끝난 상태여야 함.

## Env 인터페이스 (step 2 대비 스켈레톤)

`env.py`에 RLCard 스타일 래퍼 껍데기만 만든다. 실제 feature tensor 구성은 step 2에서 채우고, 여기서는 인터페이스만 확정한다.

```python
class LostCitiesEnv:
    def __init__(self, config: LostCitiesConfig): ...
    def reset(self) -> dict: ...
    def step(self, action_id: int) -> tuple[dict, float, bool, dict]: ...
    def legal_actions(self) -> np.ndarray: ...  # bool 1D array
    @property
    def current_player(self) -> int: ...
    @property
    def phase(self) -> str: ...  # "card" or "draw"
```

`reset()`과 `step()`이 반환하는 obs dict 스키마:

```python
{
    "spatial": np.ndarray,      # shape는 step 2에서 확정, 일단 zeros placeholder
    "scalar": np.ndarray,       # 동일
    "legal_mask": np.ndarray,   # 현재 phase의 legal mask (bool)
    "phase": int,               # 0 = card, 1 = draw
    "player": int,              # 시점 플레이어
}
```

Step 1에서는 `spatial`, `scalar`는 placeholder zeros로 두고 `legal_mask`, `phase`, `player`만 제대로 채운다. Reward는 terminal에서만 non-zero, step 1에선 raw 점수차만 반환 (정규화는 step 2).

## TUI 설계 (Textual)

### 목표

- 키보드로 조작 가능
- 1P/2P 둘 다 로컬 사람이 플레이 (hot-seat)
- Random bot, safe heuristic bot 중 상대 선택 가능
- Bot끼리 자동 대전 모드도 지원 (100판 돌려서 평균 점수 확인용)

### 화면 레이아웃 예시 (tier 1)

```
┌─────────────────────────────────────────────┐
│  Opponent expeditions                       │
│  [Color 0] H1 | 3 4 6                       │
│  [Color 1] (empty)                          │
│  [Color 2] 5                                │
│                                             │
│  Discard piles (top / size)                 │
│  [0] 2 (3)  [1] 4 (1)  [2] -- (0)           │
│                                             │
│  Deck: 7 cards remaining                    │
│                                             │
│  My expeditions                             │
│  [Color 0] 2 5                              │
│  [Color 1] H1 | 2 3                         │
│  [Color 2] (empty)                          │
│                                             │
│  My hand (press 1-5 to select)              │
│  1:[0]H1  2:[0]4  3:[1]6  4:[2]3  5:[2]5    │
│                                             │
│  Selected: slot 3 = [color=1, rank=6]       │
│  Action: [P]lay  [D]iscard                  │
│                                             │
│  Status: Your turn, Phase 1 (card)          │
└─────────────────────────────────────────────┘
```

Phase 2 (draw)가 되면 하단이 draw source 선택으로 전환.

```
  Draw from: [0]Deck  [1]Discard-c0  [2]Discard-c1  [3]Discard-c2
```

이번 턴에 내가 방금 버린 색은 회색으로 비활성 표시.

### 조작

- **카드 선택**: 숫자 키 1-N (N = hand_size)
- **Play**: P
- **Discard**: D
- **Draw source**: 숫자 키 0-N (0 = 덱, 1~N = 색별 discard)
- **Quit**: Q

### 플레이어 모드 선택

TUI 시작 시 메뉴:

```
  Lost Cities (tier1)

  [1] Human vs Human (hot-seat)
  [2] Human vs Random bot
  [3] Human vs Safe heuristic bot
  [4] Random vs Random auto (100 games)
  [5] Random vs Safe heuristic auto (100 games)
```

Hot-seat 모드에선 플레이어 교대 시 "Player 2 turn - press any key when ready" 프롬프트로 화면을 가리는 옵션을 둔다. 기본은 바로 보여주고, `--hide-on-swap` 같은 옵션으로 활성화.

### Tier parameterization

TUI는 config를 받아서 동적으로 렌더링. Tier 1 기준으로 레이아웃을 짜되, tier 바꿔도 손패 슬롯 수, 색 수, rank 범위가 자동 조정되어야 한다. Tier 3 (손패 8장, 5색) 까지 화면에 들어가도록 여유 확보.

## Bots

`bots.py`에 둘 구현.

### Random bot

```python
class RandomBot:
    def act(self, obs: dict) -> int:
        legal = obs["legal_mask"]
        legal_indices = np.nonzero(legal)[0]
        return int(np.random.choice(legal_indices))
```

### Safe heuristic bot

Lost Cities 전통적 보수 전략. 수십 줄 수준.

**Card-step 로직**:

1. 손패 각 카드를 평가.
2. **Handshake play**: 해당 색에 숫자카드 아직 없고, 같은 색 다른 카드 (handshake 또는 충분히 큰 숫자) 2장 이상 있으면 play.
3. **숫자 play**: 해당 색 원정 이미 시작됐으면 오름차순 조건 만족하는 한 play. 시작 안 됐으면 같은 색 숫자카드 3장 이상 있고 그 중 2장이 rank >= 중간값일 때만 play (낮은 숫자로 원정 시작 회피).
4. 위 조건 모두 해당 없으면 **discard**. Discard 우선순위: 상대가 해당 색 원정 시작 안 했으면 덜 위험 → 낮은 rank 먼저.

**Draw-step 로직**:

1. 각 색 discard top을 확인. 그것이 내가 이번에 바로 쓸 수 있는 카드 (내 원정 오름차순 조건 만족 또는 handshake인데 해당 색 원정 비어있음) 면 거기서 draw.
2. 아니면 덱에서 draw.

완벽하지 않다. 강하지도 않다. 하지만 random보다 확실히 낫고, 학습 에이전트의 첫 baseline으로 충분하다.

## Tests (pytest)

### test_rules.py

- 덱 생성: `n_colors × (n_ranks + n_handshakes)` 장 정확히
- 초기 손패: 덱에서 `hand_size × 2` 장 빠짐
- Play 오름차순 강제: 원정에 5 있는데 3 play 시도하면 legal mask false
- Handshake after 숫자 금지: 원정에 숫자카드 있으면 handshake play 불가
- 방금 버린 색 draw 금지: color 2에 discard하고 같은 턴에 color 2 discard draw 불가
- 다음 턴에는 가능: 다음 턴에 color 2 discard draw legal
- 덱 소진 종료 처리

### test_scoring.py

- 빈 원정: 0점
- Handshake만 있는 원정: `(0 - penalty) × multiplier`
- 숫자만 있는 원정: `(sum - penalty) × 1`
- Handshake 2 + 숫자 3장: `(sum - penalty) × 3`
- 보너스 threshold 초과: + bonus_amount
- 수작업 계산 예제 3-5개 정확히 맞춤

### test_masks.py

- 모든 phase에서 legal mask에 최소 1개 true (invariant)
- 빈 손패 slot은 play/discard 둘 다 false
- 빈 discard pile은 draw 불가
- Random vs random 1000 에피소드 돌리는 fuzz test에서 invariant 위반 0회

### test_config.py

- Tier 0/1/2/3 config 모두 valid한 game 생성
- Random vs random 100 에피소드 평균 점수차가 0에 충분히 가까움 (tier별 tolerance 조정)

## Step 1 완료 기준

다음이 모두 통과하면 step 1 완료.

1. pytest 전체 통과
2. TUI에서 본인 vs 본인 한 판 완주 (hot-seat, tier 1)
3. TUI에서 본인 vs random bot 한 판 완주
4. TUI에서 random vs random 100판 자동 대전이 에러 없이 끝남
5. Safe heuristic이 random 상대로 100판 중 55% 이상 승률
6. Config를 tier 3으로 바꿔도 TUI가 화면 깨짐 없이 렌더링되고 random vs random 돌아감

## 다음 단계 예고

Step 2는 DMC 학습 루프를 올린다.

- `models.py`: DMCNet (conv + MLP + 두 개 head)
- `train.py`: actor/learner 구조, short-life FIFO replay, self-play
- Env의 spatial/scalar feature tensor 실제 구현
- Tier 0에서 완전탐색 해와 비교, tier 1에서 safe heuristic 추월 확인

Step 1이 탄탄하면 step 2는 훨씬 빨리 간다. 반대로 step 1이 허술하면 step 2에서 학습이 안 되는 이유를 찾다가 결국 step 1로 돌아오게 된다. 그래서 이 단계에서 pytest와 TUI 플레이 검증에 시간을 충분히 쓴다.