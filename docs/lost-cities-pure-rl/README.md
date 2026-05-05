# Lost Cities Pure RL Self-Play 실험

## 목적

`safe_heuristic`, `passive_discard`, imitation, rollout label 없이 pure self-play만으로 Lost Cities 정책이 강해지고 일반화되는지 확인한다. 핵심 질문은 다음 하나다.

```text
외부 휴리스틱 없이 자기 자신과만 훈련해도 safe_heuristic 계열을 넘을 수 있는가?
```

## 고정 원칙

훈련 상대는 current policy와 과거 self-play snapshot만 사용한다.

훈련에서 제외한다.

- `random`
- `passive_discard`
- `safe_heuristic`
- `safe_heuristic_loose`
- `safe_heuristic_strict`
- `noisy_safe`
- imitation data
- rollout label

외부 bot은 평가에만 사용한다.

## 구현 상태

이 브랜치에서 pure self-play league를 Deep CFR traversal에 추가했다.

- `traversal.opponent_policy: self_play_league`
- `traversal.self_play_league.current_weight: 0.5`
- `traversal.self_play_league.recent_weight: 0.3`
- `traversal.self_play_league.older_weight: 0.2`
- `traversal.self_play_league.recent_window: 5`
- `traversal.self_play_league.max_snapshots: 20`
- `traversal.self_play_league.snapshot_every: 1`

이 mode는 opponent node에서 외부 bot을 쓰지 않고, current advantage policy와 이전 iteration의 advantage snapshot을 비율대로 섞어 action을 샘플한다. `max_snapshots`를 넘으면 가장 오래된 snapshot부터 버린다.

평가 지표에는 기존 `win_rate`, `avg_diff`, timeout 관련 지표에 더해 다음을 추가했다.

- `eval_<opponent>_avg_game_length`
- `eval_<opponent>_policy_entropy`

## 실험군

### Run A: random init pure self-play

Config:

```text
configs/lost_cities_deep_cfr_pure_self_play_a.yaml
```

초기화는 random init이다. 훈련에는 self-play league만 쓴다.

### Run B: safe pretrain init pure self-play

Config:

```text
configs/lost_cities_deep_cfr_pure_self_play_b.yaml
```

초기화는 기존 safe imitation checkpoint를 사용한다. 이후 훈련에는 self-play league만 쓴다. config architecture는 `checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt`와 맞춘 `256x3`이다.

## 실행 명령

먼저 2시간 budget을 같은 조건으로 비교한다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_a.yaml \
  --max-hours 2
```

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_b.yaml \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --max-hours 2
```

6시간, 12시간, 24시간 budget은 같은 config에서 checkpoint directory만 분리해 실행한다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_a.yaml \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_a_6h \
  --max-hours 6
```

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_b.yaml \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_b_6h \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --max-hours 6
```

12시간과 24시간은 `_12h`, `_24h` suffix와 `--max-hours 12`, `--max-hours 24`를 사용한다.

## 평가 명령

학습 중 상태 확인:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli status \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_a
```

고정 평가 suite는 다음 opponent 전체를 사용한다.

```bash
for opponent in random passive_discard safe_heuristic safe_heuristic_loose safe_heuristic_strict noisy_safe; do
  uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
    --checkpoint checkpoints/lost_cities_deep_cfr_pure_self_play_a/latest.pt \
    --games 500 \
    --max-steps 1000 \
    --opponent "$opponent"
done
```

Run B는 checkpoint path만 바꿔 같은 명령으로 평가한다.

과거 self-play checkpoint나 best historical checkpoint 상대 평가는 `--opponent-checkpoint`를 사용한다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli eval \
  --checkpoint checkpoints/lost_cities_deep_cfr_pure_self_play_a/latest.pt \
  --games 500 \
  --max-steps 1000 \
  --opponent-checkpoint checkpoints/lost_cities_deep_cfr_pure_self_play_a/iteration_00005.pt
```

## 성공 기준

최소 기준:

- `safe_heuristic` win rate `>= 0.50`
- `safe_heuristic` avg diff `> 0`
- `random` win rate `>= 0.95`
- `passive_discard` 성능 유지 또는 상승
- timeout rate 악화 없음

더 좋은 기준:

- safe family 평균 승률 `>= 0.50`
- safe family 평균 score diff `> 0`
- self-play checkpoint league에서도 계속 상승

## 판정

- Run A 성공: pure self-play만으로 충분하다.
- Run B만 성공: safe pretraining은 유용한 warm start다.
- Run B가 초반만 좋고 정체: safe pretraining이 초기 바이어스일 수 있다.
- A/B 모두 실패: pure 검증을 닫고 외부 opponent pool 또는 rollout label을 검토한다.

Run C는 A/B가 실패한 뒤에만 검토한다. Run C는 pure self-play 실험이 아니므로 이 문서의 성공 판정에는 섞지 않는다.
