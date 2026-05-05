# Lost Cities Zero-Pit PoC

이 문서는 Lost Cities zero-pit 진단 실험군의 도메인 수준 인덱스다. 개별 실험의 설계, 진행 기록, 최종 결과는 `experiments/lost_cities/<experiment_slug>/` 아래에 둔다.

## 핵심 질문

```text
random init pure self-play에서 regret matching epsilon만 키우면 zero-pit을 피하거나 약화할 수 있는가?
```

이 질문은 pure self-play가 `discard` / `draw_pile` 쪽으로 너무 빨리 몰리면서 expedition을 거의 열지 않는 실패 모드가 단순 조기 확률 붕괴인지 확인하기 위한 것이다.

## 실험 기록

| experiment | 상태 | 핵심 변경 | 기록 |
| --- | --- | --- | --- |
| `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e3` | 종료 | `regret_matching_epsilon=1.0e-3` | [README](../../experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/README.md), [plan](../../experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/plan.md), [progress](../../experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/progress.md), [report](../../experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/report.md) |
| `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4` | 실행 전 | `regret_matching_epsilon=1.0e-4` | [README](../../experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/README.md), [plan](../../experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/plan.md), [progress](../../experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/progress.md) |

## 공통 해석 원칙

- external bot은 evaluation에만 사용한다.
- safe imitation checkpoint를 쓰지 않는다.
- 기존 checkpoint에서 resume하지 않는다.
- reward, game rule, timeout semantics를 바꾸지 않는다.
- win rate보다 행동 분포와 timeout을 먼저 본다.

## 현재 판단

`eps1e3`은 zero-pit 완화에는 효과가 있었지만 safe 계열 상대 성능은 개선하지 못했다. 다음 실행 대상은 `eps1e4`이며, 30~60분 지점에서 행동 분포와 safe 계열 지표를 먼저 확인한다.
