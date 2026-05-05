# Safe Heuristic Behavior Diagnostic

이 폴더는 Lost Cities safe heuristic 계열 bot의 기준 행동 분포를 측정하는 진단 record다.

## 목적

`full_depth` self-play 정책의 over-opening 패턴을 해석하기 위해 safe heuristic 자체가 평균 몇 색을 여는지, safe 계열끼리의 점수 분포가 0 근처에 모이는지 확인한다.

## 실행

```bash
uv run python experiments/lost_cities/safe_heuristic_behavior_diagnostic/diagnose.py \
  --games 100 \
  --json-output experiments/lost_cities/safe_heuristic_behavior_diagnostic/report.json \
  --markdown-output experiments/lost_cities/safe_heuristic_behavior_diagnostic/report.md
```

## 기록 지표

- opened colors mean/std/histogram
- play/discard/draw rate
- average score, win rate, score diff
- game length
- terminal/timeout count and rate

## 결과 요약

- safe vs safe는 평균 점수차 +0.28로 거의 zero-sum 0 근처에 모였다.
- safe heuristic은 safe 계열 상대에게 평균 약 3.7색을 연다.
- `full_depth` 정책은 같은 safe 계열 상대에게 평균 4.8-4.9색을 열었으므로 약 1색 이상의 over-opening gap이 있다.
