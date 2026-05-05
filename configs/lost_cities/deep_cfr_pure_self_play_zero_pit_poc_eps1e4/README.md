# Lost Cities Deep CFR Pure Self-Play Zero-Pit PoC eps1e4

이 폴더가 `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4` 실험의 canonical 위치다.

## 목적

`eps1e3` 실험에서 `regret_matching_epsilon=1.0e-3`은 zero-pit을 완화했지만 safe 계열 상대 성능은 개선하지 못했다. 이 실험은 epsilon을 `1.0e-4`로 낮춰 zero-pit 탈출을 유지하면서 과한 opening과 낮은 selectivity를 줄일 수 있는지 확인한다.

## 기준 실험과 차이

기준은 종료된 `eps1e3` PoC다.

```text
configs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/config.yaml
```

의도한 주요 차이는 `regret_matching_epsilon` 하나다.

```yaml
regret_matching_epsilon: 1.0e-3  # eps1e3
regret_matching_epsilon: 1.0e-4  # 이 PoC
```

checkpoint는 디스크 사용량을 줄이기 위해 10 iteration마다 보존한다.

```yaml
save_every_iteration: false
save_iteration_interval: 10
save_latest_only: false
```

checkpoint 디렉터리는 repo 아래 경로를 유지하되, 실제 저장 위치는 2TB HDD로 연결한다.

```text
checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4
-> /mnt/2tbhdd/coolrl-checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4
```

## 실행

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/config.yaml
```

상태 확인:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli status \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4
```

plot 생성:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli plot \
  --checkpoint-dir checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4
```

분석 요약:

```bash
uv run python configs/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/analyze.py \
  --run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4
```

## 판정 기준

좋은 신호:

- `eps1e3` 대비 safe 계열 `avg_diff`가 개선된다.
- `play_action_rate`가 완전히 0 근처로 돌아가지 않는다.
- `eval_*_max_step_timeouts`가 낮게 유지된다.
- `random` 상대 성능이 크게 무너지지 않는다.
- `passive_discard` 평균 점수차가 덜 음수로 이동한다.

나쁜 신호:

- `play_action_rate`가 다시 0.01~0.02 근처로 내려가고 timeout이 증가한다.
- safe 계열 `avg_diff`가 여전히 -70~-100에 머문다.
- `random` 성능이 붕괴한다.

## 결과

아직 실행 전이다.
