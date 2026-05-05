# 실험 계획

## 가설

Lost Cities Deep CFR self-play에서 safe 계열 휴리스틱 상대 성능이 회복되지 않는 주요 원인은 traversal 깊이 truncation으로 인한 학습 신호의 systematic bias다.

수학적 골격:

```text
Q_true(open) = -초기손해 + 나중에 붙일 카드들의 미래이득
Q_train(open) ≈ -초기손해   (depth 16에서 잘려서 회수 구간 미관측)
```

이 bias가 모든 opening의 advantage를 음수로 끌어당기기 때문에, `regret_matching_epsilon`으로 강제 탐색을 해도 좋은 opening과 나쁜 opening을 구분 학습할 수 없다. random 상대로는 살아나지만 safe 계열 상대로는 -60~-90 수준에 갇히는 현재 패턴이 이 가설과 일관된다.

## 기준 실험과의 차이

기준은 `eps1e4`다.

```text
experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/config.yaml
```

의도한 차이는 두 가지다.

### 1. max_depth 제거

```yaml
max_depth: 16    # eps1e4
max_depth: null  # full_depth
```

depth 16에서 끊어 `score_diff`로 평가하던 방식을 폐기하고, 게임이 자연 종료할 때까지 traversal을 진행한다.

### 2. node budget 확장

```yaml
max_nodes_per_traversal: 10000  # eps1e4
max_nodes_per_traversal: 300    # full_depth
```

tier3 평균 게임 길이는 60-70 player turns, 약 120-140 actions이고 보수적 플레이의 경우 200까지 갈 수 있다. 300은 거의 모든 게임이 자연 종료할 수 있는 안전 여유 폭이다. 이 값을 초과하는 비정상적으로 긴 게임만 cutoff로 처리된다.

### 3. traversals_per_player 재조정

```yaml
traversals_per_player: 500  # eps1e4
traversals_per_player: 70   # full_depth
```

traversal당 노드 수가 약 7-8배 늘어나므로, iteration당 wall-clock을 비슷하게 유지하기 위해 traversal 횟수를 줄인다.

```text
500 x 16   = 8000 nodes/iteration  (eps1e4)
70  x ~120 = 8400 nodes/iteration  (full_depth)
```

이 변경의 trade-off는 의도적이다. 폭, 즉 다양한 게임 시나리오를 줄이고 깊이, 즉 각 게임의 정직한 끝까지 신호에 투자한다. truncation bias 가설이 맞다면 이 trade-off는 학습 품질 향상으로 보상된다.

### 4. 그 외는 동일

`regret_matching_epsilon`, `outcome_sampling_epsilon`, network 구조, optimization, evaluation, self-play league 설정은 모두 `eps1e4`와 같다. 변경 변수를 분리하기 위해 다른 것은 건드리지 않는다.

## 운영 계획

`max_hours: 2`를 유지한다. `eps1e4`와 동일한 wall-clock 예산이다.

이 실험은 shallow clone과 Cython 적용이 확인된 뒤에만 실행한다. 현재 record 준비 단계에서는 실행하지 않는다.

상태 확인:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli status \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth
```

분석 리포트와 plot 생성:

```bash
uv run python experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth/analyze.py \
  --run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth \
  --baseline-run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4
```

## 판정 기준

판정의 핵심은 safe 계열 점수차의 중심값이 의미 있게 위로 이동했는가다. 진폭이 작아지는지는 부차적이다. variance 원인은 별개의 메커니즘일 가능성이 있기 때문이다.

좋은 신호:

- `safe_heuristic`, `safe_heuristic_loose`, `safe_heuristic_strict`의 평균 점수차 중심값이 `eps1e4` 대비 30점 이상 회복된다.
- random 상대 성능이 크게 무너지지 않는다.
- timeout이 낮은 수준을 유지한다.
- `play_action_rate`가 epsilon으로 강제된 수준 이상에서 자연스럽게 유지된다.

모호한 신호:

- safe 점수차가 진동하지만 중심값이 10-20점 정도 개선된다. iteration 누적 신호로 추가 판단이 필요하다.

나쁜 신호:

- safe 점수차가 `eps1e4`와 같은 -60~-90 영역에 머문다.
- random 성능이 붕괴한다.
- league non-stationarity로 인한 진폭이 폭증한다.

## 알려진 리스크

1. traversal 다양성 감소: 70 x 2 = 140 traversal/iteration이다. league snapshot이 다양해지는 속도가 느려져 self-play league non-stationarity 진동이 더 커질 수 있다.
2. variance 증가: 분산이 약 sqrt(500/70), 즉 약 2.7배 커진다. iteration별 metric이 더 흔들릴 수 있으므로 trend는 누적적으로 봐야 한다.
3. iteration 시간 변동: traversal이 게임 길이에 따라 달라지므로 iteration 시간이 `eps1e4`보다 더 들쭉날쭉할 수 있다.

리스크 1, 2는 후속 ablation에서 다룰 문제이고, 이 실험에서는 의도적으로 받아들인다.

## 사전 조건

이 실험은 옵션 1, shallow clone copy, 그리고 옵션 6, Cython, 이 적용된 상태를 전제한다.

실행 전 확인:

```bash
ls src/coolrl/lost_cities/*.so 2>/dev/null || echo "Cython 빌드물 없음"
grep -A2 "def clone" src/coolrl/lost_cities/game.py
```

clone에 여전히 `from_snapshot(self.to_snapshot())` 패턴이 보이거나 `.so` 파일이 없으면 사전 조건 미충족이다.
