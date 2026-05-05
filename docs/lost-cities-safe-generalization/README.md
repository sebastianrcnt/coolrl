# Lost Cities safe-generalization 실험 로그

목표는 `safe_heuristic`을 이기면서도 `random`, `passive_discard`, variant-safe 상대 성능을 유지하는 Lost Cities 정책을 만드는 것이다.

현재 성공 기준은 500-game eval 기준으로 다음처럼 둔다.

- `safe_heuristic`: `win_rate >= 0.50`, `avg_diff > 0`
- `random`: `win_rate >= 0.90`
- `passive_discard`: `win_rate >= 0.45`
- variant-safe: `safe_heuristic_loose`, `safe_heuristic_strict`, `noisy_safe`에서 큰 붕괴 없음

## 현재 결론

2026-05-05 기준으로 목표는 아직 미달이다. 가장 좋은 실전용 기준 checkpoint는 여전히 `checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt`다.

이 checkpoint는 `random`과 `passive_discard`에는 강하지만 `safe_heuristic`에는 아직 진다. 이후 DAGGER-style imitation과 successful-policy replay를 시도했지만, safe 상대 win rate를 0.5 이상으로 올리지 못했다.

## 핵심 관찰

- Pure/self-play Deep CFR 계열은 cutoff bias와 discard spiral 때문에 현재 예산에서 `safe_heuristic`을 안정적으로 이기는 신호를 만들지 못했다.
- Safe heuristic imitation은 기본 플레이 능력을 빠르게 만든다. `random`과 `passive_discard` 상대 성능은 크게 올라간다.
- 단순 fixed-safe best-response CFR은 pretrain 정책을 빠르게 망가뜨렸다. 원인은 reservoir cold start, 큰 outcome-sampled regret target, pretrain anchor 부재, BR 목적과 strategy averaging의 부정합으로 본다.
- DAGGER-style safe-label imitation은 `passive_discard` 대응을 개선하지만, `safe_heuristic` exploitation을 만들지는 못했다.
- Successful-policy replay는 base policy가 실제로 safe를 이긴 판의 행동만 강화했지만, safe win rate는 오르지 않았다. 즉 “이긴 판 행동 모방”만으로는 timeout/루프와 평균 점수 열세를 해결하지 못했다.

## 구현된 실험 도구

- `pretrain-heuristic --dataset-mode aggregated`
  - safe-vs-safe, policy-vs-safe, policy-vs-policy 상태분포를 섞어 수집한다.
  - target action은 항상 해당 상태에서 `SafeHeuristicBot`이 고르는 행동이다.

- `pretrain-heuristic --dataset-mode successful_policy_vs_safe`
  - `--base-checkpoint` 정책이 `safe_heuristic`을 상대로 이긴 게임에서, 정책이 실제로 둔 행동만 target으로 저장한다.

- `pretrain-heuristic --init-checkpoint`
  - 기존 checkpoint에서 네트워크를 초기화한 뒤 imitation update를 수행한다.
  - 새 네트워크 초기화가 기존 정책을 날려버리는 문제를 피하기 위한 옵션이다.

- `fine-tune-policy`
  - `strategy_net`만 episodic policy-gradient 방식으로 fine-tune한다.
  - anchor checkpoint와의 KL penalty, entropy bonus, reward clipping을 지원한다.
  - Deep CFR advantage/reservoir 경로와 독립적으로, safe 상대 outcome 신호를 직접 넣기 위한 다음 실험용이다.

- Eval opponent 확장
  - `safe_heuristic_loose`
  - `safe_heuristic_strict`
  - `noisy_safe`

## 실험 결과 요약

| Checkpoint | random | passive_discard | safe_heuristic | 비고 |
| --- | ---: | ---: | ---: | --- |
| `safe_adv_imitation/latest.pt` | 0.986 | 0.536 | 0.312 | 현재 기준 best. safe `avg_diff=-19.1` 수준 |
| `aggregated_from_safe_adv.pt` | 0.974 | 0.350 | 0.154 | 512x4 새 초기화라 기존 정책 보존 실패 |
| `aggregated_init_from_safe_adv.pt` | 0.980 | 0.658 | 0.284 | passive는 개선, safe는 미개선 |
| `aggregated_init_from_safe_adv_e1.pt` | 0.984 | 0.698 | 0.246 | update를 줄여도 safe 미개선 |
| `successful_policy_vs_safe_e2.pt` | 0.984 | 0.530 | 0.276 | 이긴 판 replay도 safe 미개선 |

`safe_adv_imitation/latest.pt`의 variant-safe 500-game eval:

| Opponent | win_rate | avg_diff | max_step_timeouts |
| --- | ---: | ---: | ---: |
| `safe_heuristic_loose` | 0.254 | -26.03 | 127 |
| `safe_heuristic_strict` | 0.326 | -16.51 | 139 |
| `noisy_safe` | 0.426 | -7.19 | 17 |

## 재현 명령

현재 GUI에서 둬볼 checkpoint:

```bash
uv run lost-cities-gui \
  --mode pvc \
  --tier tier3 \
  --backend python \
  --deep-cfr-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --deep-cfr-device cpu
```

Aggregated imitation:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli pretrain-heuristic \
  --config configs/lost_cities_deep_cfr_safe_dagger_256.yaml \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_256/aggregated_init_from_safe_adv.pt \
  --dataset-mode aggregated \
  --base-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --games 1200 --epochs 4 --batch-size 2048 --max-steps 1000
```

Successful-policy replay:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli pretrain-heuristic \
  --config configs/lost_cities_deep_cfr_safe_dagger_256.yaml \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_256/successful_policy_vs_safe_e2.pt \
  --dataset-mode successful_policy_vs_safe \
  --base-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --games 2000 --epochs 2 --batch-size 2048 --max-steps 1000
```

다음 policy-gradient fine-tune 파일럿:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli fine-tune-policy \
  --config configs/lost_cities_deep_cfr_safe_dagger_256.yaml \
  --checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_256/policy_gradient_safe_e1000.pt \
  --games 1000 \
  --opponent safe_heuristic \
  --learning-rate 1.0e-6 \
  --kl-coef 0.05 \
  --entropy-coef 0.001 \
  --reward-scale 100 \
  --reward-clip 2 \
  --max-steps 1000
```

## 다음 우선순위

1. `fine-tune-policy`로 safe 상대 outcome 신호를 직접 넣는다.
2. 500-game eval에서 `random >= 0.90`, `passive >= 0.45`가 유지되는지 먼저 확인한다.
3. safe win rate가 0.35 이상으로 오르지 않으면, 단순 policy-gradient도 폐기하고 action-level safe-rollout improvement label 또는 MCTS-style one-step improvement로 넘어간다.
4. safe win rate가 0.40 이상으로 오르면, `safe_heuristic_loose`, `safe_heuristic_strict`, `noisy_safe`까지 500-game eval을 돌린다.

## 검증

다음 테스트가 통과했다.

```bash
uv run pytest src/coolrl/lost_cities/tests/test_deep_cfr_config.py src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py
```

마지막 확인 결과: `82 passed`.
