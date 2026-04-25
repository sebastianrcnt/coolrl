# Lost Cities Deep CFR Worker Benchmark Notes

## Purpose

This note records short CPU-only traversal benchmarks for the Lost Cities Deep CFR `random_rollout` cutoff experiment. The goal was to choose a practical multiprocessing worker count before running longer training.

The benchmark used `benchmark-traversal --mode mp` so it only measured the multiprocessing traversal path and skipped the much slower single-process comparison.

## Short benchmark setup

A temporary benchmark config was used with:

```yaml
traversal:
  traversals_per_player: 8
  max_depth: 12
  max_nodes_per_traversal: 10000
  cutoff_value_mode: random_rollout
  cutoff_rollouts: 1
  cutoff_rollout_policy: random
  cutoff_rollout_max_steps: 10000
  traversal_worker_chunk_size: 1
  progress_every_traversals: 0
```

This produces 16 traversal batches:

```text
num_batches = 2 players * ceil(8 / 1) = 16
```

So requests up to 16 workers can be fully used. Requests above 16 are capped at `effective_workers=16`.

Example command:

```bash
uv run python -m coolrl.lost_cities.deep_cfr.cli benchmark-traversal \
  --config /tmp/lc_rollout_bench.yaml \
  --mp-workers 16 \
  --iteration 1 \
  --mode mp
```

## Ryzen 5 5600X

CPU: 6 cores / 12 threads

| requested_workers | effective_workers | batches | traversal_seconds | nodes/sec |
|---:|---:|---:|---:|---:|
| 4 | 4 | 16 | 88.1581 | 1814.9 |
| 6 | 6 | 16 | 74.0787 | 2159.9 |
| 8 | 8 | 16 | 69.7517 | 2293.9 |
| 12 | 12 | 16 | 70.5381 | 2268.3 |

Recommended setting:

```yaml
traversal:
  num_workers: 8
  traversal_worker_chunk_size: 1
```

If system responsiveness matters more than peak throughput, `num_workers: 6` is also reasonable.

## Ryzen 9 7950X

CPU: 16 cores / 32 threads

| requested_workers | effective_workers | batches | traversal_seconds | nodes/sec |
|---:|---:|---:|---:|---:|
| 8 | 8 | 16 | 37.7034 | 4243.6 |
| 12 | 12 | 16 | 37.5020 | 4266.4 |
| 16 | 16 | 16 | 27.3821 | 5843.2 |
| 20 | 16 | 16 | 27.3302 | 5854.3 |

Recommended setting:

```yaml
traversal:
  num_workers: 16
  traversal_worker_chunk_size: 1
```

The 20-worker run was capped to 16 effective workers because the benchmark only had 16 batches.

## Cross-machine result

Best observed result:

```text
5600X: 69.7517s at 8 workers
7950X: 27.3821s at 16 workers
```

The 7950X was about 2.55x faster on this rollout-heavy benchmark.

## Auto worker policy

A simple `num_workers: auto` policy that uses all logical CPUs is not ideal for this workload.

Reasons:

- Effective workers are capped by the number of traversal batches.
- Python multiprocessing has process, IPC, and result merge overhead.
- More workers than physical cores can be slower, as seen on the 5600X where 8 workers slightly beat 12.
- The 7950X result matched physical-core scale better than logical-thread scale.

A safer future auto policy would be:

```text
effective_auto_workers = min(logical_cpus // 2, num_batches)
```

This approximates physical cores on common SMT CPUs and avoids requesting workers that cannot receive batches. It would choose about 6 workers on a 5600X and 16 workers on a 7950X.

For maximum throughput on a known machine, explicit values are still useful:

- 5600X: `num_workers: 8`
- 7950X: `num_workers: 16`

## Conclusion

For the current `random_rollout` cutoff experiment:

- Use `num_workers: 8`, `traversal_worker_chunk_size: 1` on 5600X.
- Use `num_workers: 16`, `traversal_worker_chunk_size: 1` on 7950X.
- Consider changing `num_workers: auto` later to cap by both CPU capacity and `num_batches`.
