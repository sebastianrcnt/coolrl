RL 배워나가는 시원 여정 기록

- [Tabular CFR for Kuhn Poker](src/coolrl/kuhn_poker/tabular_cfr.md)
- 9x9 Omok self-play RL with tinygrad

## 9x9 Omok

Smoke run:

```bash
uv run python -m coolrl.omok.train --config configs/omok_smoke.yaml
```

Short local run, using tinygrad's default device. On a MacBook this is usually `METAL`:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml
```

Force METAL explicitly:

```bash
uv run python -m coolrl.omok.train --config configs/omok_quick.yaml --device METAL
```

Play against a saved checkpoint:

```bash
uv run python -m coolrl.omok.gui --config configs/omok_quick.yaml --checkpoint checkpoints/omok_quick
```

Full reference-sized profile:

```bash
uv run python -m coolrl.omok.train --config configs/omok_full.yaml --device METAL
```

GUI controls: left click to move, `R` reset, `S` swap side, `N`/`P` next or previous checkpoint, `L` reload checkpoint list, `M` force AI move, `Esc` quit.
