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
