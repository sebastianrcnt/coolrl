# Lost Cities

Config-parametrized Lost Cities implementation for `coolrl`.

Step 1 contains:

- A pure Python rules engine in `game.py`
- A placeholder RL wrapper in `env.py`
- Random and safe heuristic bots in `bots.py`
- A Textual hot-seat / bot-play TUI in `tui.py`
- Pytest coverage under `tests/`

Run the TUI with:

```bash
lost-cities --tier tier1
```

If the package is not installed with the `lost-cities` extra, install `numpy`,
`pyyaml`, and `textual` before using the env or TUI.
