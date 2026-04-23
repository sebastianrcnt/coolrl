# Tinygrad: 동적 Batches를 Power-of-Two Buckets로 패딩하기

## 이유

Tinygrad는 **자신이 보는 모든 unique (shape, dtype, device) 조합에 대해 별도의 kernel을 JIT-컴파일**합니다. 이는 런타임에 모든 배치 크기를 허용하는 미리 빌드된 cuDNN/cuBLAS kernels에 dispatch하는 PyTorch eager mode처럼 동작하지 않습니다.

모델 input 형태가 호출 간 변하면, tinygrad는 새로운 형태가 나타날 때마다 새로운 kernel 세트를 컴파일합니다. NVRTC/LLVM을 사용하면 형태당 수백 ms에서 몇 초의 비용이 들 수 있으며 training loop의 "iteration 1"을 iteration 2+보다 훨씬 느리게 만들 수 있습니다.

이 repo에서 evaluator는 MCTS에서 `features.shape[0] = N`으로 호출됩니다. 여기서 N은 self-play 배치의 활성 게임 수와 같습니다. Omok 게임은 다른 move count에서 종료되므로, 단일 iteration에서 N은 1부터 `batch_size * leaves_per_batch`까지 거의 모든 값을 가질 수 있습니다. Bucketing 없이는 그 범위의 거의 모든 정수가 새로운 compile을 트리거합니다.

## 수정

forward pass 전에 N을 다음 power-of-two로 올림하고, 그 후 output을 N으로 slice합니다. `src/coolrl/omok/evaluator.py::ModelEvaluator.evaluate_features`를 참조하세요.

```python
n = features.shape[0]
bucket = 1 << (max(n, 1) - 1).bit_length()  # 1, 2, 4, 8, 16, 32, 64, 128, ...
if bucket > n:
    features = np.concatenate([features, np.zeros((bucket - n, *features.shape[1:]), dtype=features.dtype)], axis=0)
# forward pass on `features`
priors, values = priors[:n], values[:n]
```

Unique shapes는 O(batch_size)에서 O(log batch_size)로 축소됩니다. Padding은 마지막 bucket boundary에서 최대 ~2배의 compute를 낭비합니다. 회피된 compile stalls이 이를 더 이상 보상합니다.

## Evaluator lifetime이 중요합니다

Bucketing은 주어진 evaluator/JIT lifetime에서 보이는 unique shapes의 수를 줄입니다. Training loop이 매 iteration마다 새로운 `ModelEvaluator`를 구성하면, 그 local "seen buckets" 상태는 매번 비어 시작하고 `ModelEvaluator JIT bucket: ... first use`와 같은 logs가 iteration 2, 3 등에서 반복될 것입니다.

순차적 CUDA self-play의 경우, 한 candidate evaluator를 iteration 간에 활성화된 상태로 유지하고 기본 model 객체가 교체될 때만 교체하는 것을 선호합니다. Candidate model은 일반적으로 optimizer에 의해 in-place로 업데이트되므로, 그 evaluator를 재사용할 수 있습니다. Best-model evaluator는 promotion 후나 checkpoint restore 후 `best_model`이 교체될 때 재생성되어야 합니다.

반복되는 first-use logs는 tinygrad가 모든 kernel을 재컴파일했다는 증거가 아닙니다. 이 `ModelEvaluator` instance가 그 bucket을 전에 보지 못했다는 증거입니다. Log가 이후 iterations에서 동일한 긴 stalls을 동반한다면, evaluator/JIT lifetime이 너무 짧거나 persistent tinygrad cache가 사용되지 않고 있습니다.

## 이 패턴을 적용할 때

runtime에 **leading (batch) 차원이 변하는** tinygrad code path와 iteration당 같은 함수가 많이 호출되는 곳. 이 repo의 예: self-play 중 model inference, arena evaluation. Training batches은 이미 `optimization.batch_size`로 고정되어 있으므로 한 번만 컴파일되고 괜찮습니다.

## 하면 안 되는 것

- **하지 마세요**: tinygrad가 PyTorch eager처럼 동작한다고 가정. 그렇지 않습니다. Variable shapes는 비쌉니다.
- **하지 마세요**: JIT를 비활성화하여 stall을 "고치세요". 전체 성능 모델을 잃게 됩니다.
- **하지 마세요**: 계산 오버헤드가 compile 오버헤드보다 작다고 measured되지 않은 한, 큰 고정 최대값(예: 항상 1024)으로 pad하세요. 작은 networks에서는 보통 그렇지 않습니다 — power-of-two bucketing이 균형잡힌 기본값입니다.
- **하지 마세요**: 이미 안정적인 shapes(training loop, fixed-size utilities)에 bucketing을 추가합니다. 무의미한 padding을 추가합니다.

## Cache

`CACHELEVEL=2`를 설정하면 컴파일된 kernels을 `~/.cache/tinygrad/`에 유지하므로 이후 프로세스 시작은 compile 단계를 건너뜁니다. 이는 bucketing을 대체하는 것이 아니라 보완합니다 — bucketing은 처음부터 캐시되어야 하는 *unique* kernels의 수를 줄입니다.
