# Lost Cities Deep CFR Pure Self-Play Zero-Pit PoC Full Depth Derived Playability

이 폴더가 `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability` 실험의 canonical 위치다.

## 상태

준비 완료. 아직 실행하지 않았다.

## 목적

`full_depth`와 `anchor_safe015` 이후에도 safe 계열 상대에서 opening selectivity가 emerge하지 않았다.

이 실험은 network capacity와 league 구조를 그대로 둔 채, encoding에 색별 회수 가능성 derived features만 추가한다. 정책이 Lost Cities 룰상 손익분기와 회수 가능성을 읽기 쉬운 좌표계를 받으면 selectivity를 학습하는지 확인한다.

## 기준 실험

- record: `experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth`
- checkpoint: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth`
- reference: safe avg_diff 약 -39, random avg_diff 약 +40-45, safe opened colors 4.8-4.9

## 변경 변수

- `encoding.derived_playability: true`
- input dim: 1500 -> 1598
- derived block: 색별 19 dim x 5색 + 공통 3 dim
- network: 256x3 유지
- league: pure self-play 유지
- anchor: 없음
- traversal depth/node budget: `full_depth`와 동일

## 핵심 metric

- `avg_opened_colors`
- `opened_colors_count_5`
- `bad_open_rate`
- `weak_open_rate`
- `good_open_rate`
- `opening_recoverable_score_mean`
- `opening_recoverable_score_p25`
- `opening_margin_mean`
- `avg_score_per_opened_color`
- safe/random avg_diff

## 파일

- [config.yaml](config.yaml): 실행 config
- [plan.md](plan.md): 실험 설계와 판정 기준
- [progress.md](progress.md): 실행 진행 기록
- [analyze.py](analyze.py): metrics 분석 스크립트

## 실행

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability/config.yaml
```

## 분석

```bash
uv run python experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability/analyze.py \
  --run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability \
  --baseline-run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth
```
