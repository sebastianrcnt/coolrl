# Lost Cities Deep CFR Pure Self-Play Zero-Pit PoC Full Depth Derived Playability

이 폴더가 `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability` 실험의 canonical 위치다.

## 상태

완료. 색별 derived playability features는 opening selectivity를 유도하지 못한 것으로 판정한다.

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

## 결과 요약

- run: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability`
- 최신 iteration: 320
- 최신 eval iteration: 320
- node limit cutoff traversal rate: 0.0%
- terminal traversal rate: 100.0%
- safe 계열 평균 avg_diff: -49.31
- safe 계열 평균 opened colors: 4.96
- safe 계열 5색 opening 빈도: 약 97.7/100
- safe 계열 bad open rate: 88.8%
- random avg_diff: +42.88

결론:

- `derived_playability` 단독 개입은 opening selectivity를 만들지 못했다.
- iteration 320 기준 safe 계열 opened colors는 4.95-5.0에 포화되어 있었고, early training 이후 지속적인 하락 추세가 없었다.
- `full_depth` baseline의 iteration 320과 비교하면 safe 계열 avg_diff는 비슷하거나 다소 나빴고, opened colors와 5-color frequency는 완전 포화 쪽으로 후퇴했다.
- 이 개입은 약한 상대에 대한 일반적인 expedition management/recovery는 일부 유지하거나 개선했을 수 있지만, entry-threshold 문제는 풀지 못했다.
- 따라서 색별 sufficient statistics만으로는 부족하며, 남은 병목은 색별 가치 요약과 실제 hand slot의 play/discard decision 사이의 action/slot-level alignment라는 가설을 지지한다.

## 파일

- [config.yaml](config.yaml): 실행 config
- [plan.md](plan.md): 실험 설계와 판정 기준
- [progress.md](progress.md): 실행 진행 기록
- [analyze.py](analyze.py): metrics 분석 스크립트
- [report.md](report.md): 최종 분석 리포트
- [report.json](report.json): 최종 분석 JSON
- `analysis_metrics.png`: metric plot
- `analysis_latest_heatmap.png`: 최신 eval heatmap
- `analysis_delta_heatmap.png`: baseline delta heatmap
- `analysis_summary.png`: compact summary plot

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
