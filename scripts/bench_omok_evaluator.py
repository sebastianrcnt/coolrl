from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from tinygrad import Tensor

from coolrl.omok.config import load_config
from coolrl.omok.device import configure_device
from coolrl.omok.network import PolicyValueNet


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def summarize(name: str, values: list[float]) -> str:
    return (
        f"{name}: avg={statistics.fmean(values):.4f}s "
        f"p50={statistics.median(values):.4f}s "
        f"p90={percentile(values, 0.90):.4f}s "
        f"min={min(values):.4f}s max={max(values):.4f}s"
    )


def run_batch(model: PolicyValueNet, features: np.ndarray, device: str) -> dict[str, float]:
    times: dict[str, float] = {}

    t0 = time.perf_counter()
    contiguous = np.ascontiguousarray(features)
    times["contiguous"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tensor = Tensor(contiguous, device=device)
    times["tensor"] = time.perf_counter() - t0

    with Tensor.train(False):
        t0 = time.perf_counter()
        logits, values = model(tensor)
        times["forward_lazy"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        priors_t = logits.softmax(axis=1)
        times["softmax_lazy"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        priors = priors_t.realize().numpy()
        times["priors_numpy"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        value_np = values.realize().numpy()
        times["values_numpy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    priors.astype(np.float32, copy=False)
    value_np.astype(np.float32, copy=False)
    times["astype"] = time.perf_counter() - t0
    times["total"] = sum(times.values())
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Omok tinygrad evaluator batches.")
    parser.add_argument("--config", default="configs/omok_full_cuda.yaml")
    parser.add_argument("--device", default="CUDA")
    parser.add_argument("--batches", default="512,1024,2048")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = configure_device(args.device)
    rng = np.random.default_rng(args.seed)
    model = PolicyValueNet(cfg.rules.board_size, cfg.network)
    batch_sizes = [int(item.strip()) for item in args.batches.split(",") if item.strip()]

    print(
        "Omok evaluator benchmark: "
        f"device={device} channels={cfg.network.channels} blocks={cfg.network.blocks} "
        f"batches={batch_sizes} warmup={args.warmup} iters={args.iters}"
    )

    for batch_size in batch_sizes:
        features = rng.normal(size=(batch_size, 4, 9, 9)).astype(np.float32)
        for _ in range(args.warmup):
            run_batch(model, features, device)

        rows = [run_batch(model, features, device) for _ in range(args.iters)]
        keys = list(rows[0])
        print(f"\nBatch {batch_size}")
        for key in keys:
            values = [row[key] for row in rows]
            print("  " + summarize(key, values))


if __name__ == "__main__":
    main()
