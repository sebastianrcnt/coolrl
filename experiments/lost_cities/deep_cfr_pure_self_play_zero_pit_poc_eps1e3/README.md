# Lost Cities Deep CFR Pure Self-Play Zero-Pit PoC eps1e3

이 폴더가 `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e3` 실험의 canonical 위치다.

## 상태

종료됨. iteration 211에서 중지했고 총 학습 시간은 약 60.6분이다.

## 목적

`regret_matching_epsilon=1.0e-3`이 random init pure self-play의 zero-pit 경향을 줄이는지 확인한다.

## 파일

- [config.yaml](config.yaml): 실행 config
- [plan.md](plan.md): 실험 설계와 판정 기준
- [progress.md](progress.md): 실행 진행 기록
- [report.md](report.md): 최종 분석 리포트
- [report.json](report.json): 구조화된 분석 결과
- [analyze.py](analyze.py): metrics 분석 스크립트
- [analysis_metrics.png](analysis_metrics.png): 실험별 분석 plot
- [analysis_latest_heatmap.png](analysis_latest_heatmap.png): 최신 eval opponent x metric heatmap. 색은 metric별로 정규화하고 숫자는 raw 값을 표시한다.
- `analysis_delta_heatmap.png`: baseline run을 지정하면 생성되는 baseline 대비 delta heatmap. 색은 metric별로 정규화하고 숫자는 delta raw 값을 표시한다.

## 실행

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/config.yaml
```

## 분석

```bash
uv run python experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_eps1e3/analyze.py \
  --run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e3
```

Run A checkpoint가 로컬에 있으면 `--baseline-run checkpoints/lost_cities_deep_cfr_pure_self_play_a_2h_official`을 추가해 baseline delta를 함께 생성한다.
기본 분석 실행은 `analysis_metrics.png`와 `analysis_latest_heatmap.png`를 생성한다.
`report.md`와 `report.json`을 갱신하려면 `--write-report`를 명시한다.

## 결론

`regret_matching_epsilon=1.0e-3`은 zero-pit 완화에는 효과가 있었지만 safe 계열 상대 성능은 개선하지 못했다. 후속 실험은 `eps1e4`를 우선 실행한다.
