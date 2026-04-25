# Lost Cities Deep CFR Training Notes

## Current Caveat

The first overnight Deep CFR checkpoint was playable, but a human-vs-checkpoint trace showed passive no-expedition behavior: the checkpoint opened no expeditions, finished at score 0, and lost 220-0.

Random-opponent win rate was misleading for this run. A random Lost Cities player often self-damages by opening weak expeditions and taking the -20 expedition penalty, so a policy that mostly discards and stays at 0 can look strong against random without learning useful expedition play.

Do not judge Lost Cities Deep CFR training only by random win rate. Track behavior diagnostics alongside win rate:

- `avg_final_score`
- `avg_opened_colors`
- `play_action_rate`

Also compare against `passive_discard`. A useful trained model should beat random while avoiding collapse to the same no-expedition behavior as the passive baseline.
