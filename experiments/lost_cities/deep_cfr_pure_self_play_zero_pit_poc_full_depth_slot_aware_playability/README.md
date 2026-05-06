# Lost Cities Deep CFR Pure Self-Play Zero-Pit PoC Slot-Aware Playability

이 폴더가 `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability` 실험의 canonical 위치다.

## 상태

진행 예정.

## 목적

`derived_playability`가 색별 회수 가능성은 제공했지만 opening selectivity를 만들지 못한 원인이 색 summary와 hand slot action 사이의 alignment 부족인지 검증한다.

이번 실험은 exp3의 색별 derived features를 유지하고, 각 hand slot마다 그 카드 색의 회수 점수, opening risk, continuation 여부를 직접 추가한다. anchor opponent는 넣지 않고 pure self-play를 유지한다.

## 기준

비교 기준은 `full_depth_derived_playability`다.

- exp3 최종: safe 계열 opened colors 4.95-5.0 plateau, bad_open_rate 약 89%
- 이번 목표: slot-level action-local feature가 bad open을 직접 줄이는지 확인

## 파일

- [config.yaml](config.yaml): 실행 config
- [plan.md](plan.md): 실험 설계와 판정 기준
- [progress.md](progress.md): 실행 진행 기록
- [analyze.py](analyze.py): metrics 분석 스크립트

실험 종료 후 추가될 파일:

- `report.md`, `report.json`: 최종 분석 리포트
- `analysis_metrics.png`: 실험별 분석 plot
- `analysis_summary.png`: 핵심 요약 plot
- `analysis_latest_heatmap.png`: 최신 eval opponent x metric heatmap
- `analysis_delta_heatmap.png`: baseline 대비 delta heatmap

## 실행

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability/config.yaml
```

## 분석

```bash
uv run python experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability/analyze.py \
  --run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability \
  --baseline-run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability
```

`report.md`와 `report.json`을 갱신하려면 `--write-report`를 명시한다.
