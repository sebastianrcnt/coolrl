# Lost Cities Deep CFR Rollout Cutoff Experiment

## Goal

This experiment checks whether rollout-based cutoff values can move Lost Cities Deep CFR away from the previous no-expedition behavior.

The key question is whether a model starts opening expeditions and playing cards when cutoff states are evaluated with terminal rollout returns instead of the current score difference.

## Motivation

The earlier checkpoint looked good against a random opponent, but that result was misleading. The policy mostly avoided opening expeditions, while random opponents often lost points by opening bad expeditions.

The suspected failure mode is cutoff bias. In Lost Cities, opening an expedition has an immediate penalty. If a cutoff returns the current score difference, the traversal may see the immediate cost but not the later upside.

A separate depth and node-budget diagnostic run did not fix this behavior, so this experiment changes the cutoff value estimate instead.

## Experiment setup

Active config:

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

## Next decision

If the primary signal never appears, another value intervention is needed.

If the primary signal appears but scores remain poor, investigate rollout variance, value scaling, capped rollouts, or value-network bootstrap.

## Status

This document records the experiment design. Final results are not recorded yet.
