# Lost Cities safe-generalization 실험 로그

목표는 `safe_heuristic`을 이기면서도 `random`, `passive_discard`, variant-safe 상대 성능을 유지하는 Lost Cities 정책을 만드는 것이다.

현재 성공 기준은 500-game eval 기준으로 다음처럼 둔다.

- `safe_heuristic`: `win_rate >= 0.50`, `avg_diff > 0`
- `random`: `win_rate >= 0.90`
- `passive_discard`: `win_rate >= 0.45`
- variant-safe: `safe_heuristic_loose`, `safe_heuristic_strict`, `noisy_safe`에서 큰 붕괴 없음

## 현재 결론

2026-05-05 기준으로 목표는 아직 미달이다. 가장 좋은 실전용 기준 checkpoint는 여전히 `checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt`다.

이 checkpoint는 `random`과 `passive_discard`에는 강하지만 `safe_heuristic`에는 아직 진다. 이후 DAGGER-style imitation, successful-policy replay, KL-anchored episodic policy-gradient, action-level safe-rollout label을 시도했지만, safe 상대 win rate를 0.5 이상으로 올리지 못했다.

현재까지 가장 좋은 safe-generalization 후보는 `checkpoints/lost_cities_deep_cfr_safe_dagger_256/safe_action_rollout_g600_e1.pt`다. `safe_heuristic` win rate는 `0.328`로 목표에는 부족하지만, 기준 checkpoint의 `0.312`보다 조금 높고 `random=0.986`, `passive_discard=0.562`를 유지했다.

## 핵심 관찰

- Pure/self-play Deep CFR 계열은 cutoff bias와 discard spiral 때문에 현재 예산에서 `safe_heuristic`을 안정적으로 이기는 신호를 만들지 못했다.
- Safe heuristic imitation은 기본 플레이 능력을 빠르게 만든다. `random`과 `passive_discard` 상대 성능은 크게 올라간다.
- 단순 fixed-safe best-response CFR은 pretrain 정책을 빠르게 망가뜨렸다. 원인은 reservoir cold start, 큰 outcome-sampled regret target, pretrain anchor 부재, BR 목적과 strategy averaging의 부정합으로 본다.
- DAGGER-style safe-label imitation은 `passive_discard` 대응을 개선하지만, `safe_heuristic` exploitation을 만들지는 못했다.
- Successful-policy replay는 base policy가 실제로 safe를 이긴 판의 행동만 강화했지만, safe win rate는 오르지 않았다. 즉 “이긴 판 행동 모방”만으로는 timeout/루프와 평균 점수 열세를 해결하지 못했다.
- KL-anchored episodic policy-gradient도 1000게임 파일럿에서는 safe 상대 개선을 만들지 못했다. `random`과 `passive_discard` 유지 기준은 통과했지만, safe win rate는 `0.280`으로 기준 checkpoint보다 낮았다.
- Action-level safe-rollout label은 지금까지 유일하게 safe win rate와 avg_diff를 동시에 소폭 개선했다. 다만 개선폭은 작다: `safe_adv_imitation`의 safe `0.312 / avg_diff=-19.1`에서 `safe_action_rollout_g600_e1`의 `0.328 / avg_diff=-15.08`까지 올라간 정도다.
- `safe_action_rollout` label은 기존 policy/safe action과 불일치하는 비율이 높고, 학습 accuracy가 낮다. 즉 label이 기존 정책과 충돌하는 hard correction을 주고 있다. 이 방향은 신호가 있지만 label 품질/분산 관리가 필요하다.

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

- `pretrain-heuristic --dataset-mode safe_action_rollout`
  - base policy가 `safe_heuristic`을 상대로 지거나 timeout 나는 trajectory를 우선 수집한다.
  - 각 hard state에서 legal action을 하나 적용한 뒤 짧은 policy-vs-safe rollout으로 action value를 비교한다.
  - target은 `SafeHeuristicBot` 행동이 아니라 rollout score가 가장 높은 action이다.

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
| `policy_gradient_safe_e1000.pt` | 0.982 | 0.478 | 0.280 | KL-anchored PG 1000게임. 유지 기준은 간신히 통과, safe 미개선 |
| `safe_action_rollout_g50_e1.pt` | 0.976 | 0.496 | 0.284 | 120 label. 유지 기준 통과, safe 미개선 |
| `safe_action_rollout_g200_e2.pt` | 0.986 | 0.544 | 0.326 | 1000 label. 첫 유의미한 safe 개선 |
| `safe_action_rollout_g600_e1.pt` | 0.986 | 0.562 | 0.328 | 3000 label. 현재 best, safe `avg_diff=-15.08` |

`safe_action_rollout_g600_e1.pt` 500-game 상세:

| Opponent | win_rate | avg_diff | avg_final_score | max_step_timeouts |
| --- | ---: | ---: | ---: | ---: |
| `random` | 0.986 | 75.36 | 7.66 | 0 |
| `passive_discard` | 0.562 | 6.39 | 6.39 | 0 |
| `safe_heuristic` | 0.328 | -15.08 | -6.70 | 71 |

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

Policy-gradient fine-tune 파일럿:

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

이 파일럿의 500-game eval 결과:

```text
vs random: win_rate=0.982 avg_diff=73.75 avg_final_score=7.56 max_step_timeouts=0
vs passive_discard: win_rate=0.478 avg_diff=3.90 avg_final_score=3.90 max_step_timeouts=0
vs safe_heuristic: win_rate=0.280 avg_diff=-19.89 avg_final_score=-8.78 max_step_timeouts=123
```

시도했지만 checkpoint가 남지 않은 더 강한 PG 파일럿:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli fine-tune-policy \
  --config configs/lost_cities_deep_cfr_safe_dagger_256.yaml \
  --checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_256/policy_gradient_safe_adv_lr3e6_g4000.pt \
  --games 4000 \
  --opponent safe_heuristic \
  --max-steps 1000 \
  --learning-rate 3.0e-6 \
  --reward-scale 60 \
  --reward-clip 2 \
  --kl-coef 0.10 \
  --entropy-coef 0.001 \
  --grad-clip 0.5
```

이 run은 프로세스 종료 후 target checkpoint가 생성되지 않았다. stdout session을 잃은 상태라 실패 원인은 확인하지 못했고, 결과로 쓰지 않는다.

Action-level safe-rollout improvement label 파일럿:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli pretrain-heuristic \
  --config configs/lost_cities_deep_cfr_safe_dagger_256.yaml \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_256/safe_action_rollout.pt \
  --dataset-mode safe_action_rollout \
  --base-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --games 200 --epochs 2 --batch-size 1024 --max-steps 1000 \
  --improvement-rollouts 1 \
  --improvement-rollout-max-steps 300 \
  --improvement-max-examples 2000
```

이 mode는 전체 episode reward를 한 번에 밀어주는 PG보다 더 직접적인 action-level correction을 기대한다. 단, label 생성 비용이 크므로 처음에는 `--improvement-max-examples`를 작게 두고 500-game eval로 신호만 확인한다.

실제로 가장 좋은 결과를 낸 command:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli pretrain-heuristic \
  --config configs/lost_cities_deep_cfr_safe_dagger_256.yaml \
  --output checkpoints/lost_cities_deep_cfr_safe_dagger_256/safe_action_rollout_g600_e1.pt \
  --dataset-mode safe_action_rollout \
  --base-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --games 600 --epochs 1 --batch-size 512 --max-steps 1000 \
  --improvement-rollouts 1 \
  --improvement-rollout-max-steps 100 \
  --improvement-max-examples 3000
```

이 run은 600게임에서 hard trajectory 13개를 찾고 3000 label을 만들었다. 학습 accuracy는 `0.1203`으로 낮았지만 eval은 소폭 개선됐다.

중단한 run:

```text
safe_action_rollout_g200_s300_e1.pt
policy_gradient_batch16_g2000.pt
policy_gradient_batch4_g1000.pt
```

`safe_action_rollout_g200_s300_e1.pt`는 `--improvement-rollout-max-steps 300`으로 label horizon을 늘려 보던 중 사용자가 중단했다. checkpoint는 생성되지 않았고 결과로 쓰지 않는다. 중간 로그상 400 state 기준 `disagreement_rate=0.795`, `avg_candidates=8.6`이었다.

`policy_gradient_batch16_g2000.pt`는 batched/normalized policy-gradient 실험으로 잠깐 시작됐지만, “일단 그만” 지시에 따라 중단했다. checkpoint는 생성되지 않았고 결과로 쓰지 않는다.

`policy_gradient_batch4_g1000.pt`도 같은 이유로 중단했다. checkpoint는 생성되지 않았고 결과로 쓰지 않는다.

## 다음 우선순위

1. 단순 policy-gradient는 우선순위를 낮춘다. `policy_gradient_safe_e1000.pt`가 safe win rate `0.280`에 그쳐 기준 checkpoint보다 낫지 않았다.
2. `safe_action_rollout` 방향은 유지하되, label 품질을 개선한다. 현재는 label이 너무 noisy해서 accuracy가 낮고 개선폭이 작다.
3. 다음 세부 실험 후보는 `--improvement-top-k 4`와 낮은 `--learning-rate 1e-5`다. 전체 legal action을 모두 비교하는 대신 policy top-k + safe action만 비교해서 label 분산과 비용을 줄인다.
4. draw-discard loop를 끊는 draw phase label을 별도로 분석한다. safe 상대 timeout이 여전히 71건이라 draw policy 교정이 핵심일 가능성이 높다.
5. 새 checkpoint는 먼저 `random >= 0.90`, `passive >= 0.45`, `safe >= 0.35`를 500-game eval에서 확인한다. `safe >= 0.40`까지 오르면 variant-safe 3종도 평가한다.

## 검증

다음 테스트가 통과했다.

```bash
uv run pytest src/coolrl/lost_cities/tests/test_deep_cfr_config.py src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py
```

마지막 확인 결과: `83 passed`.

## 커밋 / 푸시

코드와 1차 문서화는 다음 커밋으로 `main`에 push했다.

```text
af0f675 Lost Cities safe 일반화 실험 도구 추가
```

이 커밋에는 imitation/PG 실험 도구, variant-safe eval opponent, safe-generalization config, GUI 실행 보조 수정, 테스트가 포함된다.

이후 action-level safe-rollout label 구현은 다음 커밋으로 `main`에 push했다.

```text
e053356 Lost Cities safe action rollout 라벨링 추가
```
