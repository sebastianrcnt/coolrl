# 진행 기록

## 2026-05-06

- 실험은 아직 실행 전이다.
- checkpoint 디렉터리는 아직 생성되지 않았다.
- `metrics.jsonl`은 아직 생성되지 않았다.
- 1차 확인은 실행 후 30-60분 지점에서 진행한다.
- 현재 record만 준비했고, 학습은 시작하지 않았다.
- 실행 전 shallow clone과 Cython 적용 여부를 확인해야 한다.
- smoke에서 `max_nodes_per_traversal=300`은 140 traversal 중 약 89.3%가 node cap에 걸렸다.
- smoke에서 `max_nodes_per_traversal=1000`은 node limit cutoff가 0%였고, endpoint depth는 평균 약 427.7, 최대 816이었다.
- 본 실험 config의 `max_nodes_per_traversal`을 1000으로 올렸다.
