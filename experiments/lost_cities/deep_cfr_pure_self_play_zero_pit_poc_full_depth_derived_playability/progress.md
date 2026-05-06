# 진행 기록

## 2026-05-06

- 실험 record를 생성했다.
- 실험은 아직 실행 전이다.
- 변경 변수는 `encoding.derived_playability: true`뿐이다.
- checkpoint 디렉터리는 아직 생성하지 않았다.
- `metrics.jsonl`은 아직 생성되지 않았다.
- 실행 전 단위 테스트와 smoke를 수행해야 한다.

### 구현 검증

- `uv run pytest src/coolrl/lost_cities/tests/test_deep_cfr_encoding.py src/coolrl/lost_cities/tests/test_deep_cfr_config.py src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py -q`
- 결과: 87 passed
- `git diff --check`: 통과
- config load 확인:
  - `experiment_name`: `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability`
  - `encoding.derived_playability`: true
  - `input_dim`: 1598
  - `self_play_league.anchor_weight`: 0.0

### 주의

- 아직 본 실험은 실행하지 않았다.
- smoke run은 별도 checkpoint에서 실행했다.

### Smoke

명령:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability/config.yaml \
  --max-iterations 5 \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability_smoke5
```

로그:

```bash
tail -f checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability_smoke5/train.log
```

결과:

- 시작 로그에서 `input_dim=1598` 확인
- iter 5까지 완료
- `node_limit_cutoff_traversal_rate`: 0.0
- `terminal_traversal_rate`: 1.0
- iter 5 league rates: current 0.6143, recent 0.3857, older 0.0, anchor 0.0
- eval key count: 283
- safe avg_diff: -62.01
- safe opened colors mean/count_5: 1.84 / 0
- safe bad/good open rate: 0.8967 / 0.1033
- safe opening play actions: 184
- safe opening recoverable score p25: -22.0
- random avg_diff: -5.7

해석:

- smoke는 통과다. derived encoding, pure self-play league, terminal traversal, 새 opening quality metric 기록이 모두 정상 동작한다.
- smoke의 점수와 selectivity 값은 초기 정책 상태라 본 판정에 사용하지 않는다.

### 0.5차 점검: 개념 정리와 초기 관측

점검 시점:

- 최신 row iteration: 약 220
- 본 실험은 계속 실행 중이다.

이번 실험 해석을 위해 문제를 두 층으로 나눈다.

`Readability Problem`:

- 모델이 입력을 보고 이 색이 회수 가능한지, 나쁜 open인지 읽을 수 있느냐의 문제다.

`Incentive Problem`:

- 모델이 나쁜 open을 피했을 때 실제 self-play 학습에서 보상을 받느냐의 문제다.
- self-play league 안의 상대도 대부분 5색 over-opening을 하면, bad open이 충분히 punish되지 않을 수 있다.

`Readability Problem`도 다시 둘로 나뉜다.

`Recovery Readability`:

- 이미 열었거나 열어버린 색에서 어떤 카드를 붙여 점수를 회수할지 읽는 능력이다.

`Opening Selectivity Readability`:

- 애초에 어떤 색을 열지 말지 판단하는 능력이다.

현재 관측:

- derived features는 `Recovery Readability`를 개선한 듯한 신호가 있다.
- random 상대 같은 iteration 구간에서 `full_depth`보다 final score가 덜 나쁘고, avg_diff가 더 빠르게 좋아지는 구간이 있다.
- play rate가 단순히 더 높은 것은 아닌데 expedition card 수와 final score 효율이 좋아진다. 이는 무작정 더 많이 내는 것이 아니라, 열린 expedition에서 회수하는 능력이 빨라졌을 가능성과 일관된다.

하지만 `Opening Selectivity Readability` 개선 신호는 아직 없다.

- safe 계열 상대에서 opened colors는 여전히 약 4.95다.
- 5-color count는 약 98/100 수준이다.
- bad_open_rate는 약 90%다.
- opening recoverable score p25는 약 -24 근처다.
- 같은 iteration 구간에서 `full_depth`보다 오히려 더 open-heavy인 구간도 있다.

현재 해석:

- derived features가 완전히 무효라는 증거는 아니다.
- 현재 효과가 있다면 "색을 덜 여는 능력"이 아니라 "열고 난 뒤 회수하는 능력" 쪽으로 먼저 나타나고 있다.
- 핵심 목표였던 opening selectivity는 아직 emerge하지 않았다.
- 따라서 `Incentive Problem`이 계속 강하게 의심된다. 즉 모델이 bad open을 읽을 수 있더라도, pure self-play objective가 그 행동을 충분히 벌하지 않을 수 있다.

계속 볼 metric:

- `bad_open_rate`가 90%대에서 내려가는가
- `good_open_rate`가 올라가는가
- `opening_recoverable_score_p25`가 -24 근처에서 올라가는가
- `opened_colors_count_5`가 98/100에서 내려가는가
- safe avg_diff가 `full_depth` baseline보다 유지 또는 개선되는가

### 본 run 종료

- 시작: 2026-05-06 15:00:46 KST
- 사용자 지시로 조기 종료: 2026-05-06 16:09:06 KST
- train exit status: 143
- post-run analyze exit status: 0
- 최신 row iteration: 320
- 최신 eval iteration: 320
- report/plot 생성 완료

Traversal health:

- `node_limit_cutoff_traversal_rate`: 0.0%
- `terminal_traversal_rate`: 100.0%
- avg endpoint depth: 260.2
- max depth reached: 348

최신 eval 핵심 관측:

- safe avg_diff: -49.38
- safe opened colors mean/count_5: 4.96 / 98
- safe bad/good open rate: 0.886 / 0.114
- safe opening recoverable score p25: -28.0
- safe_loose avg_diff: -54.49
- safe_strict avg_diff: -44.06
- safe 계열 평균 avg_diff: -49.31
- safe 계열 평균 opened colors: 4.96
- safe 계열 평균 5-color count: 97.67/100
- safe 계열 평균 bad/good open rate: 0.888 / 0.112
- safe 계열 평균 opening recoverable score p25: -25.83
- random avg_diff: +42.88

판정:

- `derived_playability`는 opening selectivity를 유도하지 못했다.
- safe 계열 상대에서 opened colors와 5-color frequency는 plateau로 남았다.
- bad open rate도 약 89%로 유지되어, opening quality 자체도 개선되지 않았다.
- score는 `full_depth` baseline보다 safe 계열에서 약 10점 후퇴했다.
- random avg_diff는 +42.88로 유지되어, weak opponent 상대로 recovery/playability 효율은 살아 있다.

실험 해석:

- 색별 derived features는 완전한 negative result라기보다, 효과가 잘못된 행동 축으로 나타난 결과다.
- random 상대 early/mid 구간에서 final score와 avg_diff가 빨리 좋아졌으므로, `Recovery Readability`에는 도움을 준 것으로 보인다.
- 그러나 이 도움은 "나쁜 색을 피한다"가 아니라 "열고 나서 더 잘 수습한다"로 쓰였다.
- 즉 entry threshold를 낮추는 방향으로 작동했고, `Opening Selectivity Readability`로 전이되지 않았다.
- 현재 결과는 색별 summary feature가 cross-slot context는 제공하지만, hand slot action과 직접 alignment되지 않는다는 가설을 강화한다.
- 다음 feature 실험은 slot-level derived features가 더 적합하다. 예: slot color recoverable score, slot is bad open candidate, slot is playable to existing expedition.
