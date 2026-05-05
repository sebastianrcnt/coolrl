# 진행 로그

## 2026-05-05

### 준비

- Branch: `lost-cities-pure-rl-self-play`
- 진행 디렉터리: `docs/lost-cities-pure-rl`
- Run A config: `configs/lost_cities_deep_cfr_pure_self_play_a.yaml`
- Run B config: `configs/lost_cities_deep_cfr_pure_self_play_b.yaml`

### 코드 변경

- `traversal.opponent_policy: self_play_league`를 추가했다.
- self-play league는 current/recent/older advantage snapshot을 `50/30/20` 비율로 섞는다.
- 외부 bot은 training path에서 쓰지 않고, 기존 eval opponent로만 남긴다.
- evaluation 결과에 `avg_game_length`, `policy_entropy`를 추가했다.
- `train` command에 `--max-hours`, `--max-iterations`, `--checkpoint-dir` override를 추가했다.
- Run B가 pretrain optimizer/iteration을 이어받지 않도록 `train --init-checkpoint`를 추가했다.
- old/best checkpoint 상대 평가를 위해 `eval --opponent-checkpoint`를 추가했다.

### 아직 실행 전

장시간 학습 run은 아직 시작하지 않았다. 먼저 config/test 검증 후 2h Run A/B부터 실행한다.

### 검증

다음 테스트를 통과했다.

```bash
uv run pytest \
  src/coolrl/lost_cities/tests/test_deep_cfr_config.py \
  src/coolrl/lost_cities/tests/test_deep_cfr_smoke.py \
  src/coolrl/lost_cities/tests/test_deep_cfr_traversal.py
```

결과:

```text
78 passed in 12.18s
```

실제 train CLI config load도 확인했다.

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_a.yaml \
  --max-iterations 0 \
  --checkpoint-dir /tmp/coolrl_pure_rl_a_config_check
```

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli train \
  --config configs/lost_cities_deep_cfr_pure_self_play_b.yaml \
  --init-checkpoint checkpoints/lost_cities_deep_cfr_safe_adv_imitation/latest.pt \
  --max-iterations 0 \
  --checkpoint-dir /tmp/coolrl_pure_rl_b_config_check
```

두 명령 모두 `device=cuda`, `input_dim=1500`, `actions=22`로 시작 조건을 로드했고, Run B는 safe checkpoint를 optimizer/iteration 없이 network weights만 초기화했다.
