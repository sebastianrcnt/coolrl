# 진행 기록

## 종료

- iteration 211에서 중지했다.
- 총 학습 시간은 약 60.6분이다.
- 실행 결과는 `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e3/` 아래에 저장했다.

## 산출물

- [training_metrics.png](training_metrics.png)
- [report.md](report.md)
- [report.json](report.json)

## 최종 관찰

- `regret_matching_epsilon=1.0e-3`은 zero-pit 완화에는 효과가 있었다.
- safe 계열 timeout이 크게 줄었고, baseline 대비 play 비율과 opened colors가 증가했다.
- safe 계열 상대 성능은 개선되지 않았다.
- 후속 실험은 `regret_matching_epsilon=1.0e-4` 또는 `3.0e-4` static run, 혹은 초반 `1.0e-3` 후 낮추는 schedule을 우선 검토한다.
