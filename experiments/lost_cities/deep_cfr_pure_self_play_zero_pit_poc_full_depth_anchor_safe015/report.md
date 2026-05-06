# Lost Cities zero-pit 분석 리포트

- run: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_anchor_safe015`
- 최신 row iteration: `1219`
- 최신 eval iteration: `1215`
- baseline run: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth`
- baseline 최신 eval iteration: `320`

## traversal health

- node limit cutoff traversal rate: `0.0%`
- terminal traversal rate: `100.0%`
- avg endpoint depth: `258.6`
- max depth reached: `404`
- avg nodes per traversal: `259.6`
- top endpoint buckets: `200-299=82 (58.6%), 300-399=36 (25.7%), 100-199=21 (15.0%)`

| opponent | 승률 | 평균 점수차 | play 비율 | discard 비율 | 덱 드로우 비율 | 파일 드로우 비율 | 평균 개방 색 | 개방 색 표준편차 | 5색 개방 수 | 평균 원정 카드 | max step timeout | policy entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_safe | 0.22 | -44.4 | 0.3786 | 0.6214 | 0.8669 | 0.1331 | 4.86 | 0.4247 | 89 | 11.04 | 0 | 1.6462 |
| noisy_safe_opponent | - | - | - | - | - | - | - | 0.5881 | 36 | - | - | - |
| passive_discard | 0.13 | -35.68 | 0.3785 | 0.6215 | 0.408 | 0.592 | 4.91 | 0.3491 | 93 | 11.77 | 0 | 1.6972 |
| passive_discard_opponent | - | - | - | - | - | - | - | 0 | 0 | - | - | - |
| random | 0.92 | 48.31 | 0.3697 | 0.6303 | 0.7843 | 0.2157 | 5 | 0 | 100 | 13.42 | 0 | 1.6488 |
| random_opponent | - | - | - | - | - | - | - | 0.2713 | 92 | - | - | - |
| safe_heuristic | 0.08 | -56.96 | 0.3689 | 0.6311 | 0.8901 | 0.1099 | 4.81 | 0.4625 | 84 | 10.44 | 0 | 1.6434 |
| safe_heuristic_loose | 0.08 | -60.55 | 0.3798 | 0.6202 | 0.8856 | 0.1144 | 4.86 | 0.4005 | 88 | 10.66 | 0 | 1.63 |
| safe_heuristic_loose_opponent | - | - | - | - | - | - | - | 0.6147 | 13 | - | - | - |
| safe_heuristic_opponent | - | - | - | - | - | - | - | 0.5896 | 9 | - | - | - |
| safe_heuristic_strict | 0.1 | -53.83 | 0.3632 | 0.6368 | 0.8919 | 0.1081 | 4.83 | 0.4484 | 86 | 10.45 | 0 | 1.6616 |
| safe_heuristic_strict_opponent | - | - | - | - | - | - | - | 0.5709 | 5 | - | - | - |

## baseline 대비 delta

| opponent | 승률 delta | 평균 점수차 delta | play 비율 delta | discard 비율 delta | 덱 드로우 비율 delta | 파일 드로우 비율 delta | 평균 개방 색 delta | 개방 색 표준편차 delta | 5색 개방 수 delta | 평균 원정 카드 delta | max step timeout delta | policy entropy delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_safe | -0.06 | -16.78 | -0.0182 | +0.0182 | +0.1304 | -0.1304 | -0.12 | - | - | -1.7 | -1 | +0.0699 |
| passive_discard | +0.13 | +17.32 | +0.0093 | -0.0093 | -0.2691 | +0.2691 | +0.25 | - | - | +2.11 | 0 | +0.2815 |
| random | +0.02 | +7.42 | +0.082 | -0.082 | +0.2767 | -0.2767 | +0.02 | - | - | -0.75 | 0 | +0.0039 |
| safe_heuristic | -0.08 | -19.37 | +0.0104 | -0.0104 | +0.2218 | -0.2218 | -0.15 | - | - | -2.13 | -2 | +0.0514 |
| safe_heuristic_loose | -0.05 | -16.65 | +0.0382 | -0.0382 | +0.2483 | -0.2483 | -0.1 | - | - | -1.86 | -3 | +0.1412 |
| safe_heuristic_strict | -0.03 | -17.56 | +0.0218 | -0.0218 | +0.2567 | -0.2567 | -0.11 | - | - | -2 | -2 | +0.1052 |
