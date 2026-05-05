# Lost Cities zero-pit 분석 요약

- run: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e3`
- 최신 row iteration: `211`
- 최신 eval iteration: `210`
- baseline run: `checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official`
- baseline 최신 eval iteration: `410`

| opponent | 승률 | 평균 점수차 | play 비율 | discard 비율 | 덱 드로우 비율 | 파일 드로우 비율 | 평균 개방 색 | 평균 원정 카드 | max step timeout | policy entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_safe | 0.08 | -65.44 | 0.0955 | 0.9045 | 0.5242 | 0.4758 | 2.72 | 4.4 | 0 | 1.5624 |
| passive_discard | 0.02 | -13.44 | 0.0776 | 0.9224 | 0.4898 | 0.5102 | 1.15 | 2.29 | 0 | 1.773 |
| random | 0.73 | 22.69 | 0.0476 | 0.9524 | 0.4594 | 0.5406 | 1.65 | 2.68 | 0 | 1.7879 |
| safe_heuristic | 0 | -79.72 | 0.0669 | 0.9331 | 0.3477 | 0.6523 | 2.87 | 4.62 | 12 | 1.5914 |
| safe_heuristic_loose | 0 | -87.04 | 0.0913 | 0.9087 | 0.3867 | 0.6133 | 3.21 | 5.52 | 7 | 1.5267 |
| safe_heuristic_strict | 0.01 | -72.62 | 0.0556 | 0.9444 | 0.3222 | 0.6778 | 2.72 | 4.21 | 13 | 1.6321 |

## baseline 대비 delta

| opponent | 승률 delta | 평균 점수차 delta | play 비율 delta | discard 비율 delta | 덱 드로우 비율 delta | 파일 드로우 비율 delta | 평균 개방 색 delta | 평균 원정 카드 delta | max step timeout delta | policy entropy delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_safe | -0.02 | -2.12 | +0.0785 | -0.0785 | - | - | +0.98 | +1.56 | -37 | -0.1003 |
| passive_discard | +0.02 | -7.8 | +0.063 | -0.063 | - | - | +0.76 | +1.73 | 0 | +0.0463 |
| random | +0.03 | +0.76 | +0.022 | -0.022 | - | - | +0.34 | +0.41 | 0 | -0.0334 |
| safe_heuristic | -0.13 | -2.26 | +0.0485 | -0.0485 | - | - | +0.67 | +1.14 | -57 | +0.014 |
| safe_heuristic_loose | -0.14 | +1.92 | +0.067 | -0.067 | - | - | +0.7 | +1.36 | -53 | -0.0498 |
| safe_heuristic_strict | -0.1 | -1.27 | +0.041 | -0.041 | - | - | +0.83 | +1.29 | -60 | +0.0459 |
