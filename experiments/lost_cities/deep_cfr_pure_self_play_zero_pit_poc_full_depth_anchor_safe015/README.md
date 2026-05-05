# Lost Cities Deep CFR Pure Self-Play Zero-Pit PoC Full Depth Anchor Safe015

이 폴더가 `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_anchor_safe015` 실험의 canonical 위치다.

## 상태

smoke 통과. 4시간 run을 시작한다.

## 목적

`full_depth` 실험은 truncation bias를 완화하고 recovery skill을 학습했지만, safe 계열 상대에서 4.8-4.9색을 여는 over-opening pattern을 유지했다.

이 실험은 self-play league에 `safe_heuristic` anchor를 0.15 weight로 주입해 self-mirror 평형을 깨고, opening selectivity가 유도 가능한지 확인한다.

중요한 부작용 위험은 opponent specialization이다. selectivity가 좋아지더라도 random avg_diff가 `full_depth`의 +40 전후에서 +20 이하로 크게 후퇴하면, robust한 selectivity가 아니라 safe heuristic 분포에 특화된 학습일 수 있다.

## 기준 실험

- record: `experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth`
- checkpoint: `checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth`
- reference: iter 320 기준 safe avg_diff 약 -39.3, random avg_diff 약 +40.9, safe opened colors 약 4.94-4.96

## 파일

- [config.yaml](config.yaml): 실행 config
- [plan.md](plan.md): 실험 설계와 판정 기준
- [progress.md](progress.md): 실행 진행 기록
- [analyze.py](analyze.py): metrics 분석 스크립트

실험 종료 후 추가될 파일:

- `report.md`, `report.json`
- `analysis_metrics.png`
- `analysis_latest_heatmap.png`
- `analysis_delta_heatmap.png`
- `analysis_summary.png`

## 실행

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_anchor_safe015/config.yaml
```

## 분석

```bash
uv run python experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_anchor_safe015/analyze.py \
  --run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_anchor_safe015 \
  --baseline-run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth
```
