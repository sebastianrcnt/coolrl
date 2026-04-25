# Lost Cities Deep CFR Training Notes

## Overnight Run Summary

The first overnight Deep CFR run produced a playable checkpoint, but the learned behavior appears collapsed to a passive discard-only policy.

Run setup and scale:

- Config: `configs/lost_cities_deep_cfr_overnight.yaml`
- Checkpoint directory: `checkpoints/lost_cities_deep_cfr_overnight`
- Runtime target: about 8 wall-clock hours on CPU
- Completed iterations: `746`
- Total traversal nodes recorded across metrics: about `298.4M`
- Latest recorded eval row: iteration `745`

Observed eval at iteration 745:

- Versus `random`: `win_rate=0.94`, `avg_diff=44.9`, `max_step_timeouts=0`
- Versus `safe_heuristic`: `win_rate=0.08`, `avg_diff=-37.0`, `max_step_timeouts=1`

Human GUI trace finding:

- The Deep CFR checkpoint opened no expeditions.
- It finished at score `0`.
- The human player won `220-0`.

## Interpretation

The random-opponent result was misleading. A random Lost Cities player often self-damages by opening weak expeditions and taking the -20 expedition penalty. A policy that mostly discards and stays at 0 can therefore look strong against random without learning useful expedition play.

The safe_heuristic result and the human GUI trace point to the same concern: the checkpoint may have learned to avoid downside rather than learning when to open and extend expeditions.

Do not judge Lost Cities Deep CFR training only by random win rate.

## Added Diagnostics

Evaluation now tracks passive-policy diagnostics for the StrategyNetBot:

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

Opened color means an expedition has length greater than 0 in the final state.

There is also a passive baseline eval opponent:

- Deep CFR eval opponent name: `passive_discard`
- GUI / bot registry name: `passive-discard`

The passive baseline never plays to expedition if discard is legal and draws from deck when legal. A useful trained model should beat random and should not match `passive_discard` on behavior diagnostics.

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

Evaluate against random:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
  --checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt \
  --games 500 \
  --opponent random
```

Evaluate against safe heuristic:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
  --checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt \
  --games 500 \
  --opponent safe_heuristic
```

Evaluate against the passive baseline:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
  --checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt \
  --games 500 \
  --opponent passive_discard
```

Play the checkpoint in GUI:

```bash
uv run lost-cities-gui \
  --mode pvc \
  --tier tier3 \
  --backend python \
  --deep-cfr-checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt
```

Play with sampled Deep CFR actions:

```bash
uv run lost-cities-gui \
  --mode pvc \
  --tier tier3 \
  --backend python \
  --deep-cfr-checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt \
  --deep-cfr-sample
```

## Next Experiments

Recommended experiments:

- Re-run training with the new diagnostics enabled from iteration 1 so passive collapse is visible in `metrics.jsonl`, status, and plots.
- Track `random`, `safe_heuristic`, and `passive_discard` together; do not rely on `random` alone.
- Inspect checkpoints at multiple iterations, not only `latest.pt`, to see whether expedition play appears and then collapses.
- Run GUI spot checks after promising eval rows and export traces for terminal games.
- Compare deterministic eval and `--deep-cfr-sample` GUI play; deterministic argmax may hide useful but underweighted expedition actions.
- If diagnostics remain passive, investigate training semantics separately: traversal depth/node limits, strategy target quality, chance/opponent sampling, and utility scaling.

Success criteria for a useful checkpoint:

- `avg_opened_colors` is clearly above 0 against random, safe_heuristic, and passive_discard.
- `play_action_rate` is clearly above 0 and not indistinguishable from `passive_discard`.
- `avg_final_score` is positive often enough to show the model is creating value, not just avoiding penalties.
- It beats `random` without merely staying at 0.
- It improves materially against `safe_heuristic`.
- GUI traces show the model opening and extending plausible expeditions.
