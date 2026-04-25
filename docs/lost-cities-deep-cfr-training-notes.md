# Lost Cities Deep CFR 학습 노트

## Overnight Run 요약

첫 번째 overnight Deep CFR run은 플레이 가능한 checkpoint를 만들었지만, 학습된 행동은 수동적인 discard-only policy로 붕괴한 것으로 보입니다.

Run 설정과 규모:

- Config: `configs/lost_cities_deep_cfr_overnight.yaml`
- Checkpoint directory: `checkpoints/lost_cities_deep_cfr_overnight`
- Runtime target: CPU 기준 약 8 wall-clock hours
- Completed iterations: `746`
- Metrics에 기록된 total traversal nodes: 약 `298.4M`
- 마지막으로 기록된 eval row: iteration `745`

Iteration 745에서 관찰된 eval:

- Versus `random`: `win_rate=0.94`, `avg_diff=44.9`, `max_step_timeouts=0`
- Versus `safe_heuristic`: `win_rate=0.08`, `avg_diff=-37.0`, `max_step_timeouts=1`

Human GUI trace에서 확인한 내용:

- Deep CFR checkpoint는 expedition을 하나도 열지 않았습니다.
- 최종 점수는 `0`이었습니다.
- 인간 플레이어가 `220-0`으로 이겼습니다.

## 해석

Random opponent 결과는 오해를 부를 수 있었습니다. Lost Cities에서 random player는 약한 expedition을 열고 -20 expedition penalty를 받으면서 스스로 점수를 깎는 경우가 많습니다. 따라서 대부분 discard만 하면서 0점에 머무는 policy도 random을 상대로는 유용한 expedition play를 배우지 않았는데도 강해 보일 수 있습니다.

`safe_heuristic` 결과와 human GUI trace는 같은 우려를 가리킵니다. 이 checkpoint는 expedition을 언제 열고 어떻게 확장할지 배운 것이 아니라, 손실을 피하는 법만 학습했을 가능성이 있습니다.

Lost Cities Deep CFR 학습을 random win rate만으로 판단하지 마세요.

## 추가된 Diagnostics

Evaluation은 이제 StrategyNetBot에 대해 passive-policy diagnostics를 기록합니다.

- `avg_final_score`
- `avg_opponent_score`
- `avg_opened_colors`
- `avg_opponent_opened_colors`
- `avg_expedition_cards`
- `avg_play_actions`
- `avg_discard_actions`
- `play_action_rate`
- `discard_action_rate`
- `max_step_timeouts`

Opened color는 final state에서 길이가 0보다 큰 expedition을 뜻합니다.

Passive baseline eval opponent도 추가되었습니다.

- Deep CFR eval opponent name: `passive_discard`
- GUI / bot registry name: `passive-discard`

Passive baseline은 discard가 legal이면 expedition에 play하지 않고 항상 discard하며, 가능하면 deck에서 draw합니다. 유용하게 학습된 모델이라면 random을 이겨야 할 뿐 아니라, behavior diagnostics에서도 `passive_discard`와 구분되어야 합니다.

## Post-merge checkpoint 재평가

Passive diagnostics가 추가된 뒤 기존 overnight checkpoint를 다시 평가했습니다. 결과는 passive no-expedition collapse를 강하게 확인합니다.

```text
Evaluation vs random: games=500 win_rate=0.988 avg_diff=52.94 avg_final_score=-1.62 avg_opponent_score=-54.56 avg_opened_colors=0.18 play_action_rate=0.006 discard_action_rate=0.994 wins=494 losses=4 draws=2 max_step_timeouts=0
Evaluation vs safe_heuristic: games=500 win_rate=0.068 avg_diff=-38.26 avg_final_score=-1.49 avg_opponent_score=36.77 avg_opened_colors=0.16 play_action_rate=0.001 discard_action_rate=0.999 wins=34 losses=465 draws=1 max_step_timeouts=14
Evaluation vs passive_discard: games=500 win_rate=0.002 avg_diff=-1.58 avg_final_score=-1.58 avg_opponent_score=0.00 avg_opened_colors=0.16 play_action_rate=0.008 discard_action_rate=0.992 wins=1 losses=74 draws=425 max_step_timeouts=0
```

Interpretation:

- `random` 상대 `win_rate=0.988`은 성공 지표가 아닙니다.
- 모델의 own `avg_final_score`는 `-1.62`로 음수입니다.
- `avg_opened_colors=0.18`, `play_action_rate=0.006`이므로 expedition을 거의 열지 않습니다.
- `safe_heuristic` 상대로는 `win_rate=0.068`로 크게 집니다.
- `passive_discard` 상대로는 대부분 비기고, 가끔 expedition을 열어 손해를 보며 집니다.

따라서 이 checkpoint는 game value를 만드는 policy가 아니라, 상대가 random하게 자해하기를 기다리는 near-passive policy로 간주합니다. 이 checkpoint를 기준으로 training을 더 오래 돌리는 것은 권장하지 않습니다.

## 현재 실패 가설

가장 유력한 가설은 cutoff value bias입니다.

Lost Cities는 expedition을 여는 순간 -20 penalty를 먼저 받습니다. 현재 traversal cutoff fallback이 현재 `score_diff`를 반환하면, shallow/depth-limited 또는 node-limited traversal에서는 expedition opening의 즉시 손실은 보이지만 이후의 upside는 잘 보이지 않을 수 있습니다.

첫 overnight run의 마지막 로그는 대략 다음과 같았습니다.

```text
nodes=400000
cutoffs=186959
node_limit_cutoffs=1135
cutoff_rate=0.4674
node_limit_cutoff_rate=0.0028
```

즉 많은 trajectory가 terminal utility가 아니라 현재 score-diff bootstrap에 의해 평가됩니다. 이 조건에서는 “안 여는 것”이 안정적으로 좋아 보이는 attractor가 될 수 있습니다.

다만 아직 cutoff value만이 원인이라고 단정하지 않습니다. 함께 확인할 후보는 다음과 같습니다.

- traversal depth / node budget 부족
- current-score cutoff의 opening penalty bias
- utility scale 문제
- strategy averaging 또는 target quality 문제
- memory/update 부족
- information-state encoding 또는 action conversion 문제
- deterministic argmax evaluation이 low-probability expedition action을 숨기는 문제

## 다음 의사결정 실험 계획

다음 실험의 목적은 “깊이와 노드 예산만 늘리면 delayed reward를 발견하는가, 아니면 cutoff value semantics를 바꿔야 하는가”를 분리하는 것입니다.

### 1. 보수적 진단 run

먼저 training semantics를 바꾸지 않고 depth/node budget만 늘린 diagnostic run을 짧게 실행합니다.

권장 변경:

```yaml
traversal:
  max_depth: 16
  max_nodes_per_traversal: 20000
```

운영 방식:

- `eval_every: 10`
- opponents: `random`, `safe_heuristic`, `passive_discard`
- 30~50 iterations 정도만 먼저 확인
- deterministic eval과 sampled GUI를 둘 다 확인

초기 중단 기준:

- `avg_opened_colors < 0.5`
- `play_action_rate < 0.02`
- `avg_final_score`가 0 근처 또는 음수
- `passive_discard`와 behavior가 거의 구분되지 않음

이 조건이 유지되면 단순히 더 오래 학습하지 말고 cutoff/value intervention으로 넘어갑니다.

### 1 결과: depth/node budget 단독 실험

`configs/lost_cities_deep_cfr_diagnostic_depth16_nodes20k.yaml`로 depth/node budget만 늘린 diagnostic run을 실행했습니다.

설정:

- `max_depth: 16`
- `max_nodes_per_traversal: 20000`
- `traversals_per_player: 20`
- opponents: `random`, `safe_heuristic`, `passive_discard`

결과:

- iteration 30 eval에서 `avg_opened_colors=0`, `avg_expedition_cards=0`, `play_action_rate=0`
- latest checkpoint 500-game eval도 `random`, `safe_heuristic`, `passive_discard` 모두에서 expedition play가 0이었습니다.
- `random` 상대 `win_rate=0.998`였지만 own `avg_final_score=0.00`이고 opponent가 `-55.27`을 기록했기 때문입니다.
- `safe_heuristic` 상대로는 `win_rate=0.030`, `avg_diff=-35.83`
- `passive_discard` 상대로는 500 games 모두 draw

판정:

- depth 16과 20k node budget만으로는 passive no-expedition collapse를 해결하지 못했습니다.
- 따라서 단순히 더 깊게/더 오래 학습하는 방향보다 cutoff/value semantics를 바꾸는 실험으로 넘어가는 것이 맞습니다.

### 2. Cutoff value intervention 후보

Depth/node budget만으로 passive collapse가 해소되지 않으면 현재 cutoff fallback인 current `score_diff`를 바꾸는 실험을 설계합니다.

후보:

- cutoff 지점에서 1~4회 terminal rollout 후 평균 `score_diff` 사용
- 별도 value network bootstrap
- utility normalization/scaling
- terminal-only deeper samples를 더 많이 확보
- curriculum으로 낮은 tier 또는 완화된 penalty/bonus에서 시작

가장 먼저 고려할 intervention은 short rollout-to-terminal cutoff입니다. 이는 handcrafted Lost Cities heuristic을 직접 넣는 대신 terminal utility를 더 자주 보게 하는 방법이므로, heuristic contamination이 상대적으로 작습니다.

### 2 진행 상황: random rollout cutoff

`random_rollout` cutoff value mode를 추가했습니다.

설정:

```yaml
traversal:
  cutoff_value_mode: random_rollout
  cutoff_rollouts: 1
  cutoff_rollout_policy: random
  cutoff_rollout_max_steps: 10000
```

의도:

- 기존 기본값은 계속 `score_diff`입니다.
- `random_rollout`은 depth cutoff와 node-budget fallback에서 현재 `score_diff` 대신 cutoff state를 복사한 뒤 random legal action으로 terminal 또는 max steps까지 진행하고, traverser 기준 final `score_diff`를 반환합니다.
- handwritten Lost Cities heuristic이나 reward shaping은 넣지 않았습니다.

초기 관찰:

- rollout stats는 정상 기록됩니다: `cutoff_rollouts`, `cutoff_rollout_steps`, `cutoff_rollout_max_step_timeouts`, `avg_cutoff_rollout_steps`
- terminal random rollout은 매우 비쌉니다.
- Conservative rollout config 기준 iteration당 약 `4.7분`
- iteration당 약 `158k~164k` cutoff rollouts
- iteration당 약 `42M~45M` rollout action steps
- rollout max-step timeout은 현재 관찰된 early iterations에서 `0`

Benchmark 관련:

- `origin/main`에는 `benchmark-traversal --mode {compare,single,mp}`가 추가되었습니다.
- rollout config에서는 `--mode compare`가 single-process benchmark까지 먼저 실행하므로 매우 느립니다.
- 빠른 multiprocessing benchmark만 보려면 다음 형태를 사용합니다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli benchmark-traversal \
  --config configs/lost_cities_deep_cfr_cutoff_random_rollout.yaml \
  --mp-workers 4 \
  --iteration 1 \
  --mode mp
```

주의:

- `--mode mp`는 benchmark 시간을 줄일 뿐 training 자체를 빠르게 만들지는 않습니다.
- terminal random rollout을 그대로 30 iterations까지 미는 것은 가능하지만 비용이 큽니다.

다음 실용적 스텝:

- latest early checkpoint라도 500-game eval을 실행해 nonzero expedition play가 이미 나타나는지 확인합니다.
- 효과 신호가 없거나 비용이 너무 크면 capped rollout config를 별도 실험으로 만듭니다. 예: `cutoff_rollout_max_steps: 100` 또는 `300`
- capped rollout도 passive collapse를 깨지 못하면 value network bootstrap을 다음 후보로 검토합니다.
- `cutoff_rollouts`를 2 또는 4로 늘리는 것은 terminal rollout 비용이 너무 커서, 먼저 capped rollout 또는 더 싼 value estimate가 필요합니다.

### 3. 성공 판정

새 run이 유망하려면 random win rate보다 아래 지표가 먼저 좋아져야 합니다.

- `avg_final_score`가 양수 방향으로 올라감
- `avg_opened_colors`가 0에서 명확히 벗어남
- `play_action_rate`가 passive baseline과 구분됨
- `passive_discard` 상대로 우위가 생김
- `safe_heuristic` 상대로 `avg_diff` 또는 `win_rate`가 개선됨
- GUI trace에서 expedition을 열고 확장하는 행동이 보임

## Traversal Worker 튜닝 노트

Deep CFR traversal multiprocessing에서 실제 executor worker 수는 요청한 worker 수와 다를 수 있습니다.

- `requested_workers`: 설정/CLI에서 요청한 worker 수
- `effective_workers`: 실제 사용된 worker 수
- `effective_workers = min(requested_workers, num_batches)`

`num_batches`는 아래 값으로 결정됩니다.

```text
num_batches = 2 players * ceil(traversals_per_player / traversal_worker_chunk_size)
```

예시:

```yaml
traversals_per_player: 20
traversal_worker_chunk_size: 4
num_workers: 12
```

위 설정의 배치 수는 `2 * ceil(20 / 4) = 10`이므로 실제 worker는 최대 10입니다.

worker를 더 사용하고 싶다면:

- `traversal_worker_chunk_size`를 줄여 배치 수를 늘리거나
- `traversals_per_player`를 늘려 배치 수를 늘립니다.

## Commands

Status:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli status \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_overnight
```

Plot:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli plot \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_overnight
```

Random 상대로 evaluation:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
  --checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt \
  --games 500 \
  --opponent random
```

Safe heuristic 상대로 evaluation:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
  --checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt \
  --games 500 \
  --opponent safe_heuristic
```

Passive baseline 상대로 evaluation:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
  --checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt \
  --games 500 \
  --opponent passive_discard
```

GUI에서 checkpoint 플레이:

```bash
uv run lost-cities-gui \
  --mode pvc \
  --tier tier3 \
  --backend python \
  --deep-cfr-checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt
```

Sampled Deep CFR actions로 플레이:

```bash
uv run lost-cities-gui \
  --mode pvc \
  --tier tier3 \
  --backend python \
  --deep-cfr-checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt \
  --deep-cfr-sample
```

## 다음 실험

추천 실험:

- 새로운 diagnostics를 iteration 1부터 켠 상태로 training을 다시 실행하여 passive collapse가 `metrics.jsonl`, status, plot에서 보이도록 합니다.
- `random`, `safe_heuristic`, `passive_discard`를 함께 추적하고, `random` 하나에만 의존하지 않습니다.
- `latest.pt`만 보지 말고 여러 iteration의 checkpoints를 확인하여 expedition play가 나타났다가 다시 붕괴하는지 살펴봅니다.
- 유망해 보이는 eval row 이후에는 GUI spot check를 실행하고, terminal games의 trace를 export합니다.
- Deterministic eval과 `--deep-cfr-sample` GUI play를 비교합니다. Deterministic argmax는 유용하지만 낮은 확률로만 남아 있는 expedition actions를 숨길 수 있습니다.
- Diagnostics가 계속 passive하게 남는다면 training semantics를 별도로 조사합니다: traversal depth/node limits, strategy target quality, chance/opponent sampling, utility scaling.

유용한 checkpoint의 성공 기준:

- `avg_opened_colors`가 random, safe_heuristic, passive_discard를 상대로 모두 명확히 0보다 높아야 합니다.
- `play_action_rate`가 명확히 0보다 높고, `passive_discard`와 구분되지 않는 수준이면 안 됩니다.
- `avg_final_score`가 충분히 자주 양수가 되어야 합니다. 즉, 모델이 단순히 penalty를 피하는 것이 아니라 실제 value를 만들고 있어야 합니다.
- 단순히 0점에 머무르는 방식이 아니라 expedition play를 하면서 `random`을 이겨야 합니다.
- `safe_heuristic`을 상대로도 실질적인 개선이 있어야 합니다.
- GUI trace에서 모델이 그럴듯한 expedition을 열고 확장하는 모습이 보여야 합니다.
