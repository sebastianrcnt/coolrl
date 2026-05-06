# Safe Heuristic Behavior Diagnostic

- games per matchup: `100`
- seed: `61`
- max steps: `1000`

| opponent | role | avg score | avg diff | win rate | opened mean | opened std | opened hist | play rate | discard rate | timeout rate | avg length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| safe_heuristic | anchor_safe | 8.85 | 0.28 | 0.53 | 3.67 | 0.722 | 1:3, 3:30, 4:61, 5:6 | 0.13 | 0.87 | 0.29 | 391.64 |
| safe_heuristic | opponent_policy | 8.57 | -0.28 | 0.45 | 3.65 | 0.779 | 1:2, 2:4, 3:30, 4:55, 5:9 | 0.129 | 0.871 | 0.29 | 391.64 |
| safe_heuristic_loose | anchor_safe | 8.65 | 0.47 | 0.48 | 3.75 | 0.712 | 2:5, 3:26, 4:58, 5:11 | 0.2 | 0.8 | 0.15 | 263.08 |
| safe_heuristic_loose | opponent_policy | 8.18 | -0.47 | 0.52 | 3.91 | 0.68 | 3:28, 4:53, 5:19 | 0.206 | 0.794 | 0.15 | 263.08 |
| safe_heuristic_strict | anchor_safe | 8.58 | 5.12 | 0.52 | 3.71 | 0.725 | 1:1, 2:3, 3:30, 4:56, 5:10 | 0.145 | 0.855 | 0.25 | 358.06 |
| safe_heuristic_strict | opponent_policy | 3.46 | -5.12 | 0.44 | 3.62 | 0.562 | 2:1, 3:39, 4:57, 5:3 | 0.13 | 0.87 | 0.25 | 358.06 |
| random | anchor_safe | 35.8 | 104.03 | 1 | 4.33 | 0.53 | 3:3, 4:61, 5:36 | 0.423 | 0.577 | 0 | 164.18 |
| random | opponent_policy | -68.23 | -104.03 | 0 | 4.96 | 0.196 | 4:4, 5:96 | 0.23 | 0.77 | 0 | 164.18 |
| passive_discard | anchor_safe | 41.97 | 41.97 | 0.96 | 3.97 | 0.359 | 3:8, 4:87, 5:5 | 0.451 | 0.549 | 0 | 148.48 |
| passive_discard | opponent_policy | 0 | -41.97 | 0.04 | 0 | 0 | 0:100 | 0 | 1 | 0 | 148.48 |
| noisy_safe | anchor_safe | 13.37 | 24.22 | 0.78 | 3.79 | 0.637 | 2:1, 3:30, 4:58, 5:11 | 0.364 | 0.636 | 0 | 148.7 |
| noisy_safe | opponent_policy | -10.85 | -24.22 | 0.22 | 4.59 | 0.531 | 3:2, 4:37, 5:61 | 0.366 | 0.634 | 0 | 148.7 |
