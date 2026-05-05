# 진행 기록

## 2026-05-06

- 실험은 아직 실행 전이다.
- checkpoint 디렉터리는 아직 생성되지 않았다.
- `metrics.jsonl`은 아직 생성되지 않았다.
- 1차 확인은 실행 후 30-60분 지점에서 진행한다.
- 현재 record만 준비했고, 학습은 시작하지 않았다.
- 실행 전 shallow clone과 Cython 적용 여부를 확인해야 한다.
- smoke에서 `max_nodes_per_traversal=300`은 140 traversal 중 약 89.3%가 node cap에 걸렸다.
- smoke에서 `max_nodes_per_traversal=1000`은 node limit cutoff가 0%였고, endpoint depth는 평균 약 427.7, 최대 816이었다.
- 본 실험 config의 `max_nodes_per_traversal`을 1000으로 올렸다.

### 중간 관측: iter 45 행동 패턴 비교

상태: provisional. 2시간 run 종료 전 중간 해석이다.

`eps1e4`의 iter 40, 45, 50과 `full_depth`의 iter 45를 비교했다. 같은 초반 구간을 맞춰 보면, 점수보다 먼저 봐야 할 신호는 행동 패턴이다.

| run | iter | safe play rate 평균 | safe opened colors 평균 | safe timeout 평균 | safe avg_diff | random avg_diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `eps1e4` | 45 | 0.0068 | 1.08 | 85.3 | -56.1 | -7.1 |
| `full_depth` | 45 | 0.1758 | 4.33 | 9.3 | -67.9 | 11.5 |

해석:

- `max_depth` truncation 제거는 discard-only pit 탈출에는 강하게 작동한 것으로 보인다. safe 계열 상대 기준 play rate는 약 26배 커졌고, 열린 색 수는 약 4배 늘었으며, max step timeout은 크게 줄었다.
- 그러나 점수는 아직 좋아지지 않았다. `full_depth` iter 45의 safe 평균 점수차는 `eps1e4` iter 45보다 약 11.8점 나쁘다.
- 쉬운 말로 표현하면, 기존 `eps1e4`는 카드를 거의 안 내는 쪽으로 갇혔고, `full_depth`는 카드를 너무 많이 내는 쪽으로 움직였다. 전자는 under-opening, 후자는 over-opening에 가깝다.
- `full_depth`는 random 상대에게는 좋아졌다. `random_avg_diff`가 `eps1e4` iter 45의 -7.1에서 `full_depth` iter 45의 +11.5로 올라갔다. 반대로 safe 계열 상대에게는 아직 약하다. 이는 위험을 더 감수하는 정책이 약한 상대에게는 통하지만, selective하게 punish하는 상대에게는 손해를 보는 패턴과 일치한다.
- 따라서 현재까지의 더 정확한 가설은 다음과 같다. truncation bias는 정책을 under-opening 평형으로 끌어당기는 요인이 맞다. 하지만 truncation을 제거해도 정책이 자동으로 selective opening으로 가지는 않는다. self-play league 안에서 좋은 opening과 나쁜 opening을 구분하는 selectivity가 안정적으로 생기지 않으면 over-opening 평형으로도 빠질 수 있다.

남은 불확실성:

- iter 45는 아직 초반이다. 이후 iter 100-150에서 selectivity가 천천히 생길 가능성은 남아 있다.
- 다만 `current_weight=0.5` 구조에서는 현재 정책을 많이 상대하므로, 한 번 생긴 행동 패턴이 self-play 안에서 강화될 위험도 있다.

### 중간 관측: iter 110 회복 신호

상태: provisional. iter 45의 over-opening 해석을 업데이트한다.

iter 110까지 보면, `full_depth`는 점수도 의미 있게 회복하기 시작했다.

| iter | safe avg_diff | random avg_diff | safe play rate 평균 | safe opened colors 평균 | safe timeout 평균 | node cutoff rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | -67.9 | 11.5 | 0.1758 | 4.33 | 9.3 | 0.0% |
| 60 | -54.2 | 19.4 | 0.2421 | 4.68 | 5.7 | 0.0% |
| 75 | -42.4 | 29.7 | 0.2761 | 4.72 | 5.3 | 0.0% |
| 90 | -46.3 | 34.2 | 0.3502 | 4.91 | 2.7 | 0.0% |
| 105 | -44.1 | 33.0 | 0.3394 | 4.82 | 3.3 | 0.0% |
| 110 | -39.9 | 46.5 | 0.4275 | 4.96 | 0.7 | 0.0% |

업데이트된 해석:

- truncation 문제는 현재 실험에서 상당히 풀리고 있는 것으로 보인다. `node_limit_cutoff_traversal_rate`가 계속 0%이고, `terminal_traversal_rate`는 100%로 유지된다. 즉 학습 신호가 depth나 node cap에서 다시 잘리는 증거는 없다.
- 정책도 discard-only pit에서는 확실히 벗어났다. safe 계열 play rate가 계속 올라가고 timeout이 거의 사라졌다.
- 점수도 회복 중이다. safe 평균 점수차는 iter 45의 -67.9에서 iter 110의 -39.9까지 올라왔다. 이는 `eps1e4` baseline 대비 약 +24.8점 좋은 수준이다.
- 다만 이 회복을 selective opening의 성공으로 보기는 어렵다. safe 계열 opened colors 평균은 iter 110에서 4.96으로 거의 전색을 연다. 즉 좋은 색만 골라 여는 정책이라기보다, 거의 다 열고 그중 일부를 회수하는 쪽에 가깝다.
- 따라서 현재까지는 두 문제가 분리되어 보인다. truncation 문제는 full-depth traversal로 상당히 완화되고 있다. 반면 selectivity 문제는 아직 풀렸다는 증거가 약하다.

현재 가설:

- full-depth는 under-opening pit을 제거하는 데 필요하다.
- 하지만 full-depth만으로 selective opening이 자동으로 emerge한다고 보기는 어렵다.
- 이후 남은 핵심 질문은 over-opening 상태가 더 학습되며 자연스럽게 1-3색 selective opening으로 줄어드는지, 아니면 거의 전색을 여는 local equilibrium에 머무는지다.

### 중간 결론: iter 180

상태: provisional이지만, 두 학습 현상이 분리되어 있다는 증거가 강해졌다.

iter 180 기준:

| iter | safe avg_diff | random avg_diff | safe play rate 평균 | safe opened colors 평균 | safe timeout 평균 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 110 | -39.9 | 46.5 | 0.4275 | 4.96 | 0.7 |
| 150 | -39.5 | 45.0 | 0.3342 | 4.71 | 3.7 |
| 170 | -35.1 | 41.6 | 0.4082 | 4.90 | 1.3 |
| 180 | -43.6 | 38.4 | 0.3266 | 4.86 | 1.7 |

별도 checkpoint 진단에서 safe 계열 상대 100게임씩 다시 평가한 opened colors 분포:

| checkpoint | opened colors mean | opened colors std | 5-color 빈도 |
| --- | ---: | ---: | ---: |
| iter 110 | 4.81-4.82 | 0.65-0.66 | 91-92% |
| iter 150 | 4.84 | 0.61 | 93% |
| iter 180 latest | 4.84-4.89 | 0.47-0.64 | 93-94% |

해석:

- truncation bias 가설은 현재 실험에서 강하게 지지된다. full-depth traversal은 discard-only pit을 탈출시켰고, `node_limit_cutoff_traversal_rate`는 계속 0%, `terminal_traversal_rate`는 100%로 유지된다.
- safe 계열 점수도 `eps1e4` 대비 약 +20점 이상 위로 이동했다. 즉 truncation을 제거하면 단순히 카드를 더 내는 것뿐 아니라, self-play 안에서 열린 expedition을 어느 정도 회수하는 skill도 emerge한다.
- 그러나 selectivity emergence 가설은 현재 league 설정에서는 기각 쪽으로 기운다. `safe_opened_colors` 평균은 4.96에서 4.87 근처에 머물고, 별도 분산 진단에서도 5색을 여는 게임 비율이 iter 110의 91-92%에서 iter 180의 93-94%로 줄지 않고 오히려 늘었다.
- 따라서 두 현상은 직교적으로 분리된다. 하나는 “끝까지 보는 신호가 있어야 discard-only pit을 탈출하고 회수 skill을 배운다”는 truncation 문제다. 다른 하나는 “어떤 expedition을 열지 고르는 능력이 league 안에서 생기느냐”는 selectivity 문제다. 이번 설정에서는 첫 번째는 풀리고 있지만 두 번째는 풀리지 않는다.

부가 신호:

- `random_avg_diff`는 iter 110의 +46.5에서 iter 180의 +38.4로 후퇴했다. safe 점수도 -35~-44 사이를 오간다. 이는 self-play 분포에 대한 overfitting 또는 league snapshot rotation의 영향일 수 있다.
- score oscillation 진폭은 대략 10점 내외로, `eps1e4`에서도 보였던 진동과 비슷하다. 다만 baseline이 위로 이동했다. 따라서 oscillation 자체는 truncation의 직접 결과라기보다 league snapshot rotation/non-stationarity의 함수로 보는 것이 더 그럴듯하다.

다음 실험 후보:

1. league에 anchor opponents를 주입한다. 예: safe 계열 또는 고정 policy를 0.1-0.2 weight로 섞는다. 이는 selectivity가 유도 가능한지 직접 테스트하므로 가장 결정적이다.
2. league weight를 비대칭화한다. 예: `older_weight > current_weight`로 두어 현재 정책 self-play의 자기강화를 약화한다.
3. exploration을 강화한다. 다만 이 선택지는 selectivity inducibility 자체를 직접 검증하기보다는 탐색 부족 가능성을 완화하는 쪽이다.
