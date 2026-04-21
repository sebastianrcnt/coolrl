from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from coolrl.omok.tui import app as tui_app
from coolrl.omok.tui import onnx as tui_onnx
from coolrl.omok.tui.app import build_argparser, config_from_args


def _touch(path):
    path.write_bytes(b"fake")
    return path


def test_tui_single_model_is_used_for_both_sides(tmp_path) -> None:
    model = _touch(tmp_path / "self.onnx")
    args = build_argparser().parse_args(["--model", str(model)])

    config = config_from_args(args)

    assert config.model_paths.black == model
    assert config.model_paths.white == model
    assert config.model_paths.mode == "single"
    assert config.simulations == 256
    assert config.move_delay == 0.05
    assert config.debug_lines == 1000
    assert not config.infinite


def test_tui_infinite_ignores_cli_seed(tmp_path) -> None:
    model = _touch(tmp_path / "self.onnx")
    args = build_argparser().parse_args(["--model", str(model), "--seed", "123", "--infinite"])

    config = config_from_args(args)

    assert config.infinite
    assert config.seed is None


def test_tui_two_models_are_color_specific(tmp_path) -> None:
    black = _touch(tmp_path / "black.onnx")
    white = _touch(tmp_path / "white.onnx")
    args = build_argparser().parse_args(
        ["--black-model", str(black), "--white-model", str(white), "--board-size", "15"]
    )

    config = config_from_args(args)

    assert config.board_size == 15
    assert config.model_paths.black == black
    assert config.model_paths.white == white
    assert config.model_paths.mode == "versus"


def test_tui_single_color_model_falls_back_to_self_play(tmp_path) -> None:
    model = _touch(tmp_path / "black.onnx")
    args = build_argparser().parse_args(["--black-model", str(model)])

    config = config_from_args(args)

    assert config.model_paths.black == model
    assert config.model_paths.white == model
    assert config.model_paths.mode == "single"


def test_tui_rejects_mixed_single_and_color_models(tmp_path) -> None:
    model = _touch(tmp_path / "self.onnx")
    black = _touch(tmp_path / "black.onnx")
    args = build_argparser().parse_args(["--model", str(model), "--black-model", str(black)])

    with pytest.raises(ValueError, match="either --model"):
        config_from_args(args)


def test_tui_cuda_device_requires_cuda_provider() -> None:
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider was requested"):
        tui_onnx._select_providers("cuda", ["CPUExecutionProvider"])


def test_tui_auto_device_can_fallback_to_cpu_provider() -> None:
    assert tui_onnx._select_providers("auto", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_tui_tensorrt_device_requires_tensorrt_provider() -> None:
    with pytest.raises(RuntimeError, match="TensorrtExecutionProvider was requested"):
        tui_onnx._select_providers("tensorrt", ["CUDAExecutionProvider", "CPUExecutionProvider"])


def test_tui_tensorrt_device_selects_tensorrt_first() -> None:
    assert tui_onnx._select_providers(
        "tensorrt",
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    ) == ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


def test_tui_app_can_advance_model_vs_model_game() -> None:
    if not tui_app.TEXTUAL_AVAILABLE:
        pytest.skip("Textual is not installed")

    class UniformEvaluator:
        provider = "FakeExecutionProvider"

        def evaluate(self, states):
            priors = np.zeros((len(states), states[0].action_size), dtype=np.float32)
            values = np.zeros((len(states),), dtype=np.float32)
            for index, state in enumerate(states):
                legal = state.legal_moves()
                priors[index, legal] = 1.0
                priors[index] /= max(1.0, float(priors[index].sum()))
            return priors, values

    async def run_app() -> None:
        config = tui_app.TuiConfig(
            model_paths=tui_app.ModelPaths(Path("self.onnx"), Path("self.onnx"), "single"),
            board_size=9,
            device="cpu",
            simulations=1,
            leaves_per_batch=1,
            c_puct=1.0,
            temperature=0.0,
            move_delay=0.0,
            exactly_five=False,
            max_moves=2,
            seed=0,
            paused=False,
            debug_lines=8,
            infinite=False,
        )
        evaluator = UniformEvaluator()
        players = {
            tui_app.BLACK: tui_app.PlayerRuntime(tui_app.BLACK, Path("black.onnx"), evaluator),
            tui_app.WHITE: tui_app.PlayerRuntime(tui_app.WHITE, Path("white.onnx"), evaluator),
        }
        textual_app = tui_app.OmokTuiApp(config, players)
        async with textual_app.run_test(size=(100, 40)) as pilot:
            before = textual_app.side_fraction
            await pilot.press("ctrl+right")
            await pilot.pause(0.1)
            assert textual_app.side_fraction > before
            textual_app._drag_panel_to(70)
            assert 0.22 <= textual_app.side_fraction <= 0.6
            await pilot.pause(0.6)
        assert textual_app.state.move_count == 2

    asyncio.run(run_app())


def test_tui_infinite_records_score_and_starts_new_game() -> None:
    if not tui_app.TEXTUAL_AVAILABLE:
        pytest.skip("Textual is not installed")

    class UniformEvaluator:
        provider = "FakeExecutionProvider"

        def evaluate(self, states):
            priors = np.zeros((len(states), states[0].action_size), dtype=np.float32)
            values = np.zeros((len(states),), dtype=np.float32)
            for index, state in enumerate(states):
                legal = state.legal_moves()
                priors[index, legal] = 1.0
                priors[index] /= max(1.0, float(priors[index].sum()))
            return priors, values

    async def run_app() -> None:
        config = tui_app.TuiConfig(
            model_paths=tui_app.ModelPaths(Path("self.onnx"), Path("self.onnx"), "single"),
            board_size=9,
            device="cpu",
            simulations=1,
            leaves_per_batch=1,
            c_puct=1.0,
            temperature=0.0,
            move_delay=0.0,
            exactly_five=False,
            max_moves=None,
            seed=None,
            paused=True,
            debug_lines=8,
            infinite=True,
        )
        evaluator = UniformEvaluator()
        players = {
            tui_app.BLACK: tui_app.PlayerRuntime(tui_app.BLACK, Path("black.onnx"), evaluator),
            tui_app.WHITE: tui_app.PlayerRuntime(tui_app.WHITE, Path("white.onnx"), evaluator),
        }
        textual_app = tui_app.OmokTuiApp(config, players)
        async with textual_app.run_test(size=(100, 40)) as pilot:
            first_seed = textual_app.current_seed
            for action in (0, 9, 1, 10, 2, 11, 3, 12, 4):
                textual_app.state.apply_action(action)
            textual_app._finish_game()

            assert textual_app.score.black_wins == 1
            assert textual_app.result_hold_until is not None

            await pilot.pause(0.7)

            assert textual_app.game_index == 2
            assert textual_app.state.move_count == 0
            assert not textual_app.state.terminal
            assert textual_app.current_seed is not None
            assert textual_app.current_seed != first_seed

    asyncio.run(run_app())
