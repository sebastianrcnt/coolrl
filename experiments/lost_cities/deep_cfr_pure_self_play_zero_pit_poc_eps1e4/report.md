# Lost Cities zero-pit 분석 리포트

- run: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4`
- 최신 row iteration: `219`
- 최신 eval iteration: `215`

| opponent | 승률 | 평균 점수차 | play 비율 | discard 비율 | 덱 드로우 비율 | 파일 드로우 비율 | 평균 개방 색 | 평균 원정 카드 | max step timeout | policy entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_safe | 0.1 | -46.49 | 0.1202 | 0.8798 | 0.6661 | 0.3339 | 2.76 | 5.15 | 1 | 1.6875 |
| passive_discard | 0 | -3.89 | 0.0074 | 0.9926 | 0.7052 | 0.2948 | 0.19 | 0.19 | 0 | 1.6659 |
| random | 0.86 | 32.79 | 0.0471 | 0.9529 | 0.5551 | 0.4449 | 1.34 | 2.39 | 0 | 1.853 |
| safe_heuristic | 0.05 | -62.98 | 0.0775 | 0.9225 | 0.5442 | 0.4558 | 2.37 | 4.17 | 5 | 1.6914 |
| safe_heuristic_loose | 0.03 | -74.32 | 0.1117 | 0.8883 | 0.6097 | 0.3903 | 2.85 | 5.14 | 1 | 1.657 |
| safe_heuristic_strict | 0.05 | -56.92 | 0.066 | 0.934 | 0.6049 | 0.3951 | 2.02 | 3.32 | 3 | 1.7165 |

## 관측

- 실행은 수동 중단 전 `metrics.jsonl` 기준 iteration 219까지 기록됐다.
- 마지막 완성 eval은 iteration 215다.
- 초반 safe 계열 eval에서는 `play_action_rate`가 0.01 이하로 내려가고 timeout이 80회 이상까지 증가했다.
- 후반에는 safe 계열 timeout이 1~5회로 감소했고, `play_action_rate`는 0.0660~0.1117까지 증가했다.
- `random` 상대는 마지막 eval에서 승률 0.86, 평균 점수차 32.79를 기록했다.
- safe 계열 상대는 마지막 eval에서 승률 0.03~0.05, 평균 점수차 -74.32~-56.92를 기록했다.
- `passive_discard` 상대는 마지막 eval에서 평균 점수차 -3.89였지만 승률은 0이었다.

## 결론

- `regret_matching_epsilon=1.0e-4` pure self-play 설정은 zero-pit timeout을 후반에 완화했다.
- timeout 완화는 safe 계열 승률 개선으로 이어지지 않았다.
- 마지막 eval 기준 정책은 random 상대에는 강해졌지만 safe 계열 상대에는 큰 음수 점수차를 유지했다.
- 이 run은 zero-pit 탈출 가능성과 safe 계열 robustness가 별개 현상임을 보였다.
