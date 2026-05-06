# Lost Cities Deep CFR Rollout Cutoff Experiment

> Historical note: this experiment config was removed after the result was documented. Use [`src/coolrl/lost_cities/README.md`](../../../src/coolrl/lost_cities/README.md) for current runnable training commands.

## Goal

This experiment checks whether rollout-based cutoff values can move Lost Cities Deep CFR away from the previous no-expedition behavior.

The key question is whether a model starts opening expeditions and playing cards when cutoff states are evaluated with terminal rollout returns instead of the current score difference.

## Motivation

The earlier checkpoint looked good against a random opponent, but that result was misleading. The policy mostly avoided opening expeditions, while random opponents often lost points by opening bad expeditions.

The suspected failure mode is cutoff bias. In Lost Cities, opening an expedition has an immediate penalty. If a cutoff returns the current score difference, the traversal may see the immediate cost but not the later upside.

A separate depth and node-budget diagnostic run did not fix this behavior, so this experiment changes the cutoff value estimate instead.

## Experiment setup

Historical config, removed after the experiment was documented:

```text
configs/lost_cities_deep_cfr_cutoff_random_rollout.yaml
```

Important traversal settings:

```yaml
cutoff_value_mode: random_rollout
cutoff_rollouts: 1
cutoff_rollout_policy: random
cutoff_rollout_max_steps: 10000
num_workers: auto
traversal_worker_chunk_size: 1
```

This does not add game heuristics, reward shaping, or rule changes. It only changes how cutoff states are valued.

## Primary success signal

The first question is not whether the policy is strong. The first question is whether it stops behaving like a no-expedition baseline.

Good signs:

- `avg_opened_colors` becomes greater than zero
- `avg_expedition_cards` becomes greater than zero
- `play_action_rate` becomes greater than zero
- the policy no longer mostly draws against `passive_discard`

Bad signs:

- `avg_opened_colors` remains zero
- `avg_expedition_cards` remains zero
- `play_action_rate` remains zero
- behavior remains indistinguishable from `passive_discard`

## Secondary signal

Escaping the no-expedition behavior is not enough. The policy may over-open expeditions and score poorly.

After the primary signal appears, track:

- `avg_final_score`
- avg diff against `passive_discard`
- avg diff and win rate against `safe_heuristic`
- whether play rate becomes more selective over time

## Run summary

This run was executed with the active `random_rollout` cutoff config through iteration 30.

The run answered the primary question clearly: `random_rollout` cutoff values did break the no-expedition collapse. The policy no longer stayed passive. It opened nearly all colors and played many cards.

However, the resulting behavior was not yet strong. The model moved from under-opening to over-opening: it opened expeditions and played cards, but still scored poorly, including against the passive discard baseline.

## Eval snapshots

### Iteration 10

| opponent | win_rate | avg_diff | avg_final_score | opened | exp_cards | play_rate | discard_rate | timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.70 | 28.24 | -14.06 | 4.98 | 12.98 | 0.159 | 0.841 | 0 |
| safe_heuristic | 0.06 | -48.30 | -36.52 | 4.94 | 11.20 | 0.021 | 0.979 | 10 |
| passive_discard | 0.22 | -22.68 | -22.68 | 4.86 | 12.52 | 0.381 | 0.619 | 0 |

### Iteration 20

| opponent | win_rate | avg_diff | avg_final_score | opened | exp_cards | play_rate | discard_rate | timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.94 | 46.18 | -11.62 | 5.00 | 13.50 | 0.321 | 0.679 | 0 |
| safe_heuristic | 0.04 | -41.06 | -28.54 | 5.00 | 12.44 | 0.430 | 0.570 | 0 |
| passive_discard | 0.12 | -27.24 | -27.24 | 4.96 | 12.26 | 0.468 | 0.532 | 0 |

### Iteration 30

| opponent | win_rate | avg_diff | avg_final_score | opened | exp_cards | play_rate | discard_rate | timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.84 | 39.20 | -11.70 | 5.00 | 13.60 | 0.330 | 0.670 | 0 |
| safe_heuristic | 0.02 | -46.96 | -34.82 | 4.94 | 11.86 | 0.433 | 0.567 | 0 |
| passive_discard | 0.16 | -24.52 | -24.52 | 4.96 | 12.48 | 0.468 | 0.532 | 0 |

## Interpretation

Primary result:

```text
random_rollout cutoff breaks passive no-expedition collapse.
```

Evidence:

- `avg_opened_colors` moved from near zero in the old checkpoint to about five.
- `avg_expedition_cards` moved well above zero.
- `play_action_rate` became clearly nonzero.
- the `passive_discard` matchup no longer consisted of mostly draws.

Secondary result:

```text
random_rollout cutoff currently over-corrects into over-opening.
```

Evidence:

- `avg_final_score` remains negative across opponents.
- `passive_discard` still wins on average because it scores 0 while the learned policy opens expeditions and loses points.
- `safe_heuristic` remains much stronger.
- play rates stay high rather than becoming selective.

This means the intervention did inject useful terminal information, but the terminal random rollout value is probably too noisy or too naive to produce good expedition selection by itself.

## Decision

Stop this run at iteration 30. The core question has been answered.

Do not simply train this exact setup longer unless there is a specific reason to test long-run convergence. The more useful next step is a new value/cutoff experiment that preserves the ability to open expeditions but reduces over-opening.

## Next experiment candidates

### Candidate A: Capped rollout cutoff

Reduce rollout horizon to lower variance and cost.

Possible config direction:

```yaml
traversal:
  cutoff_value_mode: random_rollout
  cutoff_rollouts: 1
  cutoff_rollout_max_steps: 300
```

Concern: if the rollout does not reach terminal, using partial `score_diff` may reintroduce some opening-penalty bias. This experiment is still useful because it tests whether cheaper, lower-variance rollout estimates are enough to avoid both passive collapse and over-opening.

### Candidate B: Value normalization or clipping

The advantage losses became large after terminal rollout values were introduced. The target scale may be too wide or too noisy.

Possible directions:

- normalize rollout returns before creating regret targets
- clip cutoff values to a reasonable score range
- track value distribution statistics for cutoff returns

Concern: normalization changes learning dynamics and must be documented carefully.

### Candidate C: Learned value bootstrap

Train a value network for cutoff states and use it instead of terminal random rollouts.

Possible advantages:

- cheaper than terminal rollouts after initial training
- lower variance if the value model becomes stable
- can be trained from terminal outcomes or rollout targets

Concern: more implementation complexity and another model to validate.

### Candidate D: Better rollout policy without hand-coded Lost Cities heuristics

The historical terminal rollout used uniformly random legal actions. That may produce extremely poor terminal estimates.

Possible directions:

- rollout with the current average strategy network
- rollout with a mixed policy: random plus current strategy
- rollout using sampled rather than argmax policy

Concern: rollout policy coupling can introduce feedback loops and must be tested carefully.

## Historical recommended next step

This was the recommended next step when the rollout cutoff experiment was active. The config is not part of the current runnable presets:

```text
capped random rollout cutoff
```

Suggested first run:

```yaml
experiment_name: lost_cities_deep_cfr_cutoff_random_rollout_capped300
traversal:
  cutoff_value_mode: random_rollout
  cutoff_rollouts: 1
  cutoff_rollout_max_steps: 300
```

Keep the same diagnostics:

- `random`
- `safe_heuristic`
- `passive_discard`
- `avg_opened_colors`
- `avg_expedition_cards`
- `play_action_rate`
- `avg_final_score`

Decision criteria for the next run:

- If it returns to `opened=0`, capped rollout reintroduced passive bias.
- If it keeps `opened≈5` and `avg_final_score<0`, capped rollout did not solve over-opening.
- If it keeps nonzero opening while improving `avg_final_score` and `passive_discard` avg diff, continue or scale up.

## Follow-up: safe heuristic rollout cutoff

The next runnable intervention replaced uniformly random cutoff rollouts with the existing `safe_heuristic` bot:

```yaml
experiment_name: lost_cities_deep_cfr_safe_rollout300
traversal:
  cutoff_value_mode: random_rollout
  cutoff_rollouts: 1
  cutoff_rollout_policy: safe_heuristic
  cutoff_rollout_max_steps: 300
```

Iteration 50 completed in about 357 seconds. Compared with capped random rollout, cutoff rollout length fell from about 257 to about 152 average steps. The final 50-game training eval moved `random` to `win_rate=0.54`, `avg_diff=+2.72`, `avg_final_score=-39.06`, `avg_opened_colors=4.16`.

A separate 500-game eval of `checkpoints/lost_cities_deep_cfr_safe_rollout300/latest.pt` confirmed the random-bot improvement:

```text
Evaluation vs random: games=500 win_rate=0.548 avg_diff=6.41 avg_final_score=-34.22 avg_opponent_score=-40.63 avg_opened_colors=4.22 play_action_rate=0.132 discard_action_rate=0.868 wins=274 losses=220 draws=6 max_step_timeouts=0
```

This is enough to clear the first practical bar: stable improvement over `random`. It does not solve the stronger bot problem; the same checkpoint remained far behind `safe_heuristic` in the 50-game training eval.

## Status

This experiment is complete enough to guide the next intervention. Final long-run strength is not established, but the cutoff-value failure mode is now better localized:

```text
score_diff cutoff -> passive no-expedition collapse
terminal random rollout cutoff -> expedition play appears, but over-opening remains
safe_heuristic rollout cutoff -> clears random bot, still loses badly to safe_heuristic
```
