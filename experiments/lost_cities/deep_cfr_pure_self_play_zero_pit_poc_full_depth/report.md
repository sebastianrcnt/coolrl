# Lost Cities zero-pit 분석 리포트

- run: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth`
- 최신 row iteration: `322`
- 최신 eval iteration: `320`
- baseline run: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4`
- baseline 최신 eval iteration: `215`

## 결론

이 실험은 종료했다. `max_depth=null`, `max_nodes_per_traversal=1000`으로 full-depth traversal을 수행한 결과, truncation bias 가설은 강하게 지지된다. traversal은 node cap에 걸리지 않았고, safe 계열 상대 점수는 `eps1e4` 대비 약 +20~30점 위로 이동했다.

동시에 selectivity emergence는 현재 pure self-play league 설정에서는 부정된다. 최종 eval에서도 safe 계열 상대의 평균 개방 색은 4.94-4.96으로 거의 전색 opening에 머문다. 별도 checkpoint 진단에서도 `full_depth` 정책은 safe 계열 상대에게 5색 opening 빈도 90%+를 보였고, safe heuristic 자체는 safe 계열 상대에게 평균 약 3.7색을 열었다.

따라서 두 학습 현상은 분리된다.

- recovery skill: self-play로 emerge한다. 거의 모든 expedition을 열더라도 열린 expedition에서 점수를 회수하는 능력은 학습된다.
- selectivity: 이 league 설정에서는 emerge하지 않는다. 어떤 expedition을 열지 고르는 능력은 anchor 없는 self-play 평형 안에서 안정적으로 생기지 않았다.

다음 실험은 league에 `safe_heuristic` 계열 anchor opponent를 0.1-0.2 weight로 주입해 selectivity inducibility를 직접 테스트하는 것이 가장 결정적이다.

## traversal health

- node limit cutoff traversal rate: `0.0%`
- terminal traversal rate: `100.0%`
- avg endpoint depth: `276.2`
- max depth reached: `392`
- avg nodes per traversal: `277.2`
- top endpoint buckets: `200-299=95 (67.9%), 300-399=42 (30.0%), 100-199=3 (2.1%)`

| opponent | 승률 | 평균 점수차 | play 비율 | discard 비율 | 덱 드로우 비율 | 파일 드로우 비율 | 평균 개방 색 | 평균 원정 카드 | max step timeout | policy entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_safe | 0.28 | -27.62 | 0.3968 | 0.6032 | 0.7365 | 0.2635 | 4.98 | 12.74 | 1 | 1.5764 |
| passive_discard | 0 | -53 | 0.3691 | 0.6309 | 0.6771 | 0.3229 | 4.66 | 9.66 | 0 | 1.4157 |
| random | 0.9 | 40.89 | 0.2877 | 0.7123 | 0.5076 | 0.4924 | 4.98 | 14.17 | 0 | 1.6449 |
| safe_heuristic | 0.16 | -37.59 | 0.3585 | 0.6415 | 0.6683 | 0.3317 | 4.96 | 12.57 | 2 | 1.5919 |
| safe_heuristic_loose | 0.13 | -43.9 | 0.3416 | 0.6584 | 0.6374 | 0.3626 | 4.96 | 12.52 | 3 | 1.4889 |
| safe_heuristic_strict | 0.13 | -36.27 | 0.3415 | 0.6585 | 0.6352 | 0.3648 | 4.94 | 12.45 | 2 | 1.5565 |

## baseline 대비 delta

| opponent | 승률 delta | 평균 점수차 delta | play 비율 delta | discard 비율 delta | 덱 드로우 비율 delta | 파일 드로우 비율 delta | 평균 개방 색 delta | 평균 원정 카드 delta | max step timeout delta | policy entropy delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_safe | +0.18 | +18.87 | +0.2765 | -0.2765 | +0.0704 | -0.0704 | +2.22 | +7.59 | 0 | -0.1112 |
| passive_discard | 0 | -49.11 | +0.3618 | -0.3618 | -0.0281 | +0.0281 | +4.47 | +9.47 | 0 | -0.2502 |
| random | +0.04 | +8.1 | +0.2406 | -0.2406 | -0.0475 | +0.0475 | +3.64 | +11.78 | 0 | -0.2081 |
| safe_heuristic | +0.11 | +25.39 | +0.281 | -0.281 | +0.124 | -0.124 | +2.59 | +8.4 | -3 | -0.0995 |
| safe_heuristic_loose | +0.1 | +30.42 | +0.2299 | -0.2299 | +0.0277 | -0.0277 | +2.11 | +7.38 | +2 | -0.1681 |
| safe_heuristic_strict | +0.08 | +20.65 | +0.2755 | -0.2755 | +0.0303 | -0.0303 | +2.92 | +9.13 | -1 | -0.16 |
