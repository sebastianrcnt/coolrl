# Lost Cities Deep CFR Pure Self-Play Zero-Pit PoC Full Depth

이 폴더가 `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth` 실험의 canonical 위치다.

## 상태

종료. iter 322 저장 이후, iter 323 traversal 도중 중단했다.

## 최종 요약

- `max_depth=null` full-depth traversal은 node cap 없이 진행됐다. 최종 eval 기준 `node_limit_cutoff_traversal_rate=0.0%`, `terminal_traversal_rate=100.0%`.
- truncation bias 가설은 강하게 지지된다. safe 계열 평균 점수차는 `eps1e4` baseline 대비 약 +25점 회복했다.
- recovery skill은 pure self-play로 emerge했다.
- selectivity emergence는 현재 league 설정에서는 부정된다. 최종 eval에서도 safe 계열 상대 평균 개방 색은 4.94-4.96으로 거의 전색 opening이다.
- 별도 [safe heuristic behavior diagnostic](../safe_heuristic_behavior_diagnostic/)에서 safe heuristic 자체는 safe 계열 상대에게 평균 약 3.7색을 열고, safe vs safe 점수는 0 근처에 모였다. `full_depth` 정책에는 약 1색 이상의 over-opening gap이 있다.
- 다음 우선순위는 league에 safe 계열 anchor opponent를 0.1-0.2 weight로 주입하는 실험이다.

## 실행 전 조정

smoke에서 `max_nodes_per_traversal=300`은 너무 낮았다. 140 traversal 중 약 89.3%가 node cap에 걸렸고, `max_nodes_per_traversal=1000`에서는 같은 1 iteration smoke 기준 node limit cutoff가 0%로 내려갔다. endpoint depth 분포는 주로 300-500대에 있고 800대까지 긴 꼬리가 있어, full-depth 신호를 보존하기 위해 본 실험 config는 1000을 사용한다.

## 목적

`max_depth=null`로 traversal truncation을 제거했을 때 safe 계열 휴리스틱 상대 성능이 회복되는지 확인한다.
회복된다면 이전 실험들의 safe 격차의 주요 원인이 truncation bias라는 가설이 강하게 지지된다.

## 파일

- [config.yaml](config.yaml): 실행 config
- [plan.md](plan.md): 실험 설계와 판정 기준
- [progress.md](progress.md): 실행 진행 기록
- [analyze.py](analyze.py): metrics 분석 스크립트 (`eps1e4`에서 복사)

- `report.md`, `report.json`: 최종 분석 리포트
- `analysis_metrics.png`: 실험별 분석 plot. eval 지표와 traversal depth, endpoint cutoff rate, 최신 endpoint depth bucket 분포를 함께 그린다.
- `analysis_latest_heatmap.png`: 최신 eval opponent x metric heatmap
- `analysis_delta_heatmap.png`: `eps1e4` baseline 대비 delta heatmap
- `analysis_summary.png`: 설명 텍스트, traversal health, 최신 heatmap, baseline delta, endpoint bucket을 합친 summary plot

## 실행

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth/config.yaml
```

## 분석

```bash
uv run python experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth/analyze.py \
  --run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth \
  --baseline-run checkpoints/lost_cities_deep_cfr_pure_self_play_zero_pit_poc_eps1e4
```

기본 분석 실행은 `analysis_metrics.png`, `analysis_latest_heatmap.png`, `analysis_delta_heatmap.png`, `analysis_summary.png`를 생성한다.
`report.md`와 `report.json`을 갱신하려면 `--write-report`를 명시한다.

## 사전 조건

이 실험은 shallow clone과 Cython 적용 후 실행했다.
