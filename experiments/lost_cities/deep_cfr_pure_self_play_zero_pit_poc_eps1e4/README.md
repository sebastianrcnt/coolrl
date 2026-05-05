# Lost Cities Deep CFR Pure Self-Play Zero-Pit PoC eps1e4

이 폴더가 `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4` 실험의 canonical 위치다.

## 상태

아직 실행 전이다.

## 목적

`regret_matching_epsilon=1.0e-4`가 `eps1e3`의 zero-pit 완화 효과를 유지하면서 과한 opening과 낮은 selectivity를 줄일 수 있는지 확인한다.

## 파일

- [config.yaml](config.yaml): 실행 config
- [plan.md](plan.md): 실험 설계와 판정 기준
- [progress.md](progress.md): 실행 진행 기록
- [analyze.py](analyze.py): metrics 분석 스크립트
- `report.md`: 실행 후 생성할 최종 분석 리포트
- `report.json`: 실행 후 생성할 구조화된 분석 결과
- `analysis_metrics.png`: 실행 후 생성할 실험별 분석 plot

## 실행

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/config.yaml
```

## 분석

```bash
uv run python experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/analyze.py \
  --run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4 \
  --json-output experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/report.json \
  --markdown-output experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/report.md \
  --plot-output experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e4/analysis_metrics.png
```
