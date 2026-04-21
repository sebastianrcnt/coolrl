# Tinygrad: Pad Dynamic Batches to Power-of-Two Buckets

## Why

Tinygrad JIT-compiles a **separate kernel for every unique (shape, dtype, device) combination** it sees. It does not behave like PyTorch eager mode, which dispatches to pre-built cuDNN/cuBLAS kernels that accept any batch size at runtime.

If model input shapes vary across calls, tinygrad compiles a fresh set of kernels each time a new shape appears. With NVRTC/LLVM this costs hundreds of ms to several seconds per shape and can make "iteration 1" of a training loop dramatically slower than iteration 2+.

In this repo the evaluator is called from MCTS with `features.shape[0] = N` where N equals the number of active games in a self-play batch. Omok games terminate at different move counts, so across a single iteration N can take almost any value from 1 up to `batch_size * leaves_per_batch`. Without bucketing, nearly every integer in that range triggers a new compile.

## The fix

Round N up to the next power of two before the forward pass, and slice the output back to N afterwards. See `src/coolrl/omok/evaluator.py::ModelEvaluator.evaluate_features`.

```python
n = features.shape[0]
bucket = 1 << (max(n, 1) - 1).bit_length()  # 1, 2, 4, 8, 16, 32, 64, 128, ...
if bucket > n:
    features = np.concatenate([features, np.zeros((bucket - n, *features.shape[1:]), dtype=features.dtype)], axis=0)
# forward pass on `features`
priors, values = priors[:n], values[:n]
```

Unique shapes collapse from O(batch_size) to O(log batch_size). Padding wastes at most ~2x compute on the last bucket boundary; the avoided compile stalls more than pay for it.

## Evaluator lifetime matters

Bucketing only reduces the number of unique shapes seen by a given evaluator/JIT
lifetime. If the training loop constructs a new `ModelEvaluator` for every
iteration, its local "seen buckets" state starts empty each time and logs such
as `ModelEvaluator JIT bucket: ... first use` will repeat on iteration 2, 3, and
so on.

For sequential CUDA self-play, prefer keeping one candidate evaluator alive
across iterations and replacing it only when the underlying model object is
replaced. The candidate model is normally updated in-place by the optimizer, so
its evaluator can be reused. The best-model evaluator should be recreated when
`best_model` is replaced after promotion or checkpoint restore.

Repeated first-use logs do not prove tinygrad recompiled every kernel; they
prove this `ModelEvaluator` instance had not seen the bucket before. If the log
is accompanied by the same long stalls on later iterations, the evaluator/JIT
lifetime is too short or the persistent tinygrad cache is not being used.

## When to apply this pattern

Any tinygrad code path where a **leading (batch) dimension varies at runtime** and the same function is called many times per iteration. Examples in this repo: model inference during self-play, arena evaluation. Training batches are already fixed by `optimization.batch_size`, so they compile once and are fine.

## What NOT to do

- **Do not** assume tinygrad behaves like PyTorch eager. It does not. Variable shapes are expensive.
- **Do not** "fix" the stall by disabling JIT. You lose the whole performance model.
- **Do not** pad to a large fixed maximum (e.g., always 1024) unless you have measured that the compute overhead is smaller than the compile overhead. For small networks it usually is not — power-of-two bucketing is the balanced default.
- **Do not** add bucketing to shapes that are already stable (training loop, fixed-size utilities). It adds pointless padding.

## Cache

Setting `CACHELEVEL=2` persists compiled kernels to `~/.cache/tinygrad/` so subsequent process starts skip the compile step. This is complementary to bucketing, not a replacement — bucketing reduces the number of *unique* kernels that have to be cached in the first place.
