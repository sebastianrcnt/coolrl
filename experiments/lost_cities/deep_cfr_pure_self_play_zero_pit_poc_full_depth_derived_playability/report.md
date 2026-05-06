# Lost Cities zero-pit 분석 리포트

- run: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability`
- 최신 row iteration: `320`
- 최신 eval iteration: `320`
- baseline run: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth`
- baseline 최신 eval iteration: `320`

## traversal health

- node limit cutoff traversal rate: `0.0%`
- terminal traversal rate: `100.0%`
- avg endpoint depth: `260.2`
- max depth reached: `348`
- avg nodes per traversal: `261.2`
- top endpoint buckets: `200-299=116 (82.9%), 300-399=19 (13.6%), 100-199=5 (3.6%)`

| opponent | 승률 | 평균 점수차 | play 비율 | discard 비율 | 덱 드로우 비율 | 파일 드로우 비율 | 평균 개방 색 | 개방 색 표준편차 | 5색 개방 수 | opening play 수 | bad open 수 | weak open 수 | good open 수 | bad open 비율 | weak open 비율 | good open 비율 | opening 회수 점수 평균 | opening 회수 점수 p25 | opening margin 평균 | 개방 색당 평균 점수 | 평균 원정 카드 | max step timeout | policy entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_safe | 0.24 | -31.12 | 0.4098 | 0.5902 | 0.681 | 0.319 | 4.97 | 0.2985 | 99 | 497 | 444 | 0 | 53 | 0.8934 | 0 | 0.1066 | -15.5231 | -24 | -9.4728 | -6.358 | 13.28 | 0 | 1.5821 |
| noisy_safe_opponent | - | - | - | - | - | - | - | 0.629 | 46 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| passive_discard | 0.04 | -55.26 | 0.3233 | 0.6767 | 0.3827 | 0.6173 | 4.62 | 0.7181 | 72 | 462 | 426 | 0 | 36 | 0.9221 | 0 | 0.0779 | -17.6234 | -26 | -10.3139 | -12.087 | 10.28 | 0 | 1.3674 |
| passive_discard_opponent | - | - | - | - | - | - | - | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| random | 0.95 | 42.88 | 0.3128 | 0.6872 | 0.495 | 0.505 | 5 | 0 | 100 | 500 | 446 | 0 | 54 | 0.892 | 0 | 0.108 | -16.216 | -26 | -9.62 | -2.398 | 14.76 | 0 | 1.6216 |
| random_opponent | - | - | - | - | - | - | - | 0.1706 | 97 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| safe_heuristic | 0.06 | -49.38 | 0.3419 | 0.6581 | 0.5537 | 0.4463 | 4.96 | 0.3137 | 98 | 496 | 440 | 0 | 56 | 0.8871 | 0 | 0.1129 | -15.5464 | -26 | -9.3508 | -7.4255 | 12.96 | 2 | 1.5953 |
| safe_heuristic_loose | 0.06 | -54.49 | 0.4137 | 0.5863 | 0.6651 | 0.3349 | 4.98 | 0.14 | 98 | 498 | 442 | 0 | 56 | 0.8876 | 0 | 0.1124 | -15.4378 | -25.5 | -9.3032 | -7.026 | 13.07 | 1 | 1.5579 |
| safe_heuristic_loose_opponent | - | - | - | - | - | - | - | 0.6108 | 13 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| safe_heuristic_opponent | - | - | - | - | - | - | - | 0.593 | 9 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| safe_heuristic_strict | 0.08 | -44.06 | 0.325 | 0.675 | 0.5331 | 0.4669 | 4.93 | 0.4302 | 97 | 493 | 439 | 0 | 54 | 0.8905 | 0 | 0.1095 | -15.6085 | -26 | -9.4199 | -7.4415 | 12.87 | 3 | 1.6013 |
| safe_heuristic_strict_opponent | - | - | - | - | - | - | - | 0.6177 | 9 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## baseline 대비 delta

| opponent | 승률 delta | 평균 점수차 delta | play 비율 delta | discard 비율 delta | 덱 드로우 비율 delta | 파일 드로우 비율 delta | 평균 개방 색 delta | 개방 색 표준편차 delta | 5색 개방 수 delta | opening play 수 delta | bad open 수 delta | weak open 수 delta | good open 수 delta | bad open 비율 delta | weak open 비율 delta | good open 비율 delta | opening 회수 점수 평균 delta | opening 회수 점수 p25 delta | opening margin 평균 delta | 개방 색당 평균 점수 delta | 평균 원정 카드 delta | max step timeout delta | policy entropy delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_safe | -0.04 | -3.5 | +0.013 | -0.013 | -0.0556 | +0.0556 | -0.01 | - | - | - | - | - | - | - | - | - | - | - | - | - | +0.54 | -1 | +0.0058 |
| passive_discard | +0.04 | -2.26 | -0.0459 | +0.0459 | -0.2944 | +0.2944 | -0.04 | - | - | - | - | - | - | - | - | - | - | - | - | - | +0.62 | 0 | -0.0484 |
| random | +0.05 | +1.99 | +0.0251 | -0.0251 | -0.0126 | +0.0126 | +0.02 | - | - | - | - | - | - | - | - | - | - | - | - | - | +0.59 | 0 | -0.0232 |
| safe_heuristic | -0.1 | -11.79 | -0.0167 | +0.0167 | -0.1146 | +0.1146 | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | +0.39 | 0 | +0.0034 |
| safe_heuristic_loose | -0.07 | -10.59 | +0.0721 | -0.0721 | +0.0277 | -0.0277 | +0.02 | - | - | - | - | - | - | - | - | - | - | - | - | - | +0.55 | -2 | +0.069 |
| safe_heuristic_strict | -0.05 | -7.79 | -0.0165 | +0.0165 | -0.1021 | +0.1021 | -0.01 | - | - | - | - | - | - | - | - | - | - | - | - | - | +0.42 | +1 | +0.0448 |
