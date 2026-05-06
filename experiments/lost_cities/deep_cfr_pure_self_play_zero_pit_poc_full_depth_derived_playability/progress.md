# 진행 기록

## 2026-05-06

- 실험 record를 생성했다.
- 실험은 아직 실행 전이다.
- 변경 변수는 `encoding.derived_playability: true`뿐이다.
- checkpoint 디렉터리는 아직 생성하지 않았다.
- `metrics.jsonl`은 아직 생성되지 않았다.
- 실행 전 단위 테스트와 smoke를 수행해야 한다.

### 구현 검증

- `uv run pytest src/coolrl/lost_cities/tests/test_deep_cfr_encoding.py src/coolrl/lost_cities/tests/test_deep_cfr_config.py src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py -q`
- 결과: 87 passed
- `git diff --check`: 통과
- config load 확인:
  - `experiment_name`: `lost_cities_deep_cfr_pure_self_play_zero_pit_poc_full_depth_derived_playability`
  - `encoding.derived_playability`: true
  - `input_dim`: 1598
  - `self_play_league.anchor_weight`: 0.0

### 주의

- 아직 본 실험은 실행하지 않았다.
- smoke run도 아직 실행하지 않았다. 사용자가 실행을 지시하면 1-5 iteration smoke에서 `input_dim=1598`, eval의 bad/opening metric 기록 여부, cutoff 0%, terminal 100%를 먼저 확인한다.
