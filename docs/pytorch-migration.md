# 오목 PyTorch 전용 마이그레이션 계획

이 문서에서는 Omok의 나머지 Tinygrad 교육 경로를 교체하는 방법을 설명합니다.
PyTorch와 함께. 목표는 두 개의 훈련 백엔드를 지원하는 것이 아닙니다. 목표는 다음과 같습니다
권장 경로에서 Tinygrad를 삭제하고 PyTorch를 단일 소스로 만듭니다.
모델 정의, 학습, 체크포인트 및 내보내기에 대한 진실입니다.

## 마이그레이션 정책

- `training.backend`를 추가하지 마세요.
- `TrainingBackend` 추상화를 추가하지 마세요.
- 타이니그라드와 성화훈련을 나란히 두지 마세요.
- 훈련 상태에 대해 PyTorch `.pt` 체크포인트를 사용합니다.
- 기존 Tinygrad 체크포인트를 레거시 입력 아티팩트로만 처리합니다.
- Tinygrad 최적화 프로그램 상태는 이식 가능하지 않으며 마이그레이션되지 않습니다.
- MCTS 인터페이스를 변경하지 않고 유지합니다. MCTS는 계속해서
  평가자 인터페이스.
- 토치 전용 경로가 작동한 후에만 검색 설정을 다시 조정하세요.

구현에서는 호환성 레이어보다 직접 교체를 선호해야 합니다.
호환성 코드는 이전 Tinygrad 모델을 로드하는 데 필요한 경우에만 허용됩니다.
마이그레이션 중 가중치.

## 현재 상태

| 면적 | 현재 백엔드 | 마이그레이션 대상 |
|---|---|---|
| 셀프 플레이 평가자 | CUDA 프로필의 PyTorch(기본적으로 Tinygrad) | PyTorch 전용 |
| 경기장 평가자 | CUDA 프로필의 PyTorch(기본적으로 Tinygrad) | PyTorch 전용 |
| 훈련 모델 | `network.py`의 Tinygrad `PolicyValueNet` | PyTorch 모델 |
| 최적화 | `train.py`의 Tinygrad `AdamW` | `torch.optim.AdamW` |
| 재생 텐서 | Tinygrad 텐서 | 토치 텐서 |
| 검문소 | Tinygrad 세이프텐서 | 토치`.pt` |
| ONNX 내보내기 | PyTorch 미러 복제 | 공유 PyTorch 모델 |
| C/파이썬 MCTS | 백엔드 중립 평가자 인터페이스 | 변함없이 |

PyTorch 평가자는 이미 네트워크를 표현할 수 있음을 보여줍니다.
정확하게:

| 장치 | 정책 최대 절대비 차이 | 가치 최대 절대비 차이 |
|---|---:|---:|
| CPU, 배치 7 | ~2.8e-9 | ~1.0e-7 |
| CUDA, 배치 64 | ~4.0e-7 | ~1.1e-5 |
| CUDA, 패딩 배치 93 | ~7.6e-7 | ~5.4e-6 |

나머지 위험은 교육 의미: BatchNorm 상태, 최적화 프로그램입니다.
하이퍼파라미터, 체크포인트 재개 동작, 재생 텐서 변환 및
전체 루프 학습 동작.

## 대상 아키텍처

최종 의도된 모듈 소유권:

| 모듈 | 책임 |
|---|---|
| `torch_network.py` | 표준 PyTorch `PolicyValueNet` 및 잔여 블록 |
| `torch_evaluator.py` | `torch_network.py` 주위의 평가자 래퍼 |
| `train.py` | PyTorch 훈련 루프, 최적화 단계, 체크포인트 호출 |
| `checkpoint.py` | 토치 `.pt` 저장/로드 및 레거시 Tinygrad 가중치 가져오기 |
| `replay.py` | numpy 재생 저장소와 토치 일괄 샘플링 |
| `export_onnx.py` | `torch_network.py`에서 ONNX 내보내기 |
| `network.py` | 레거시 가중치 가져오기 후 제거됨 더 이상 필요하지 않음 |

구성 형태는 단순하게 유지되어야 합니다.
```yaml
optimization:
  batch_size: 256
  updates_per_iteration: 96
  learning_rate: 0.0005
  weight_decay: 0.0001

selfplay:
  evaluator_backend: torch
```
이 과정의 일부로 `최적화:`를 새로운 `훈련:` 섹션으로 이동하지 마세요.
마이그레이션. 구성 스키마 마이그레이션은 토치 전용 지원 없이 이탈을 추가합니다.
목표.

이전 구성이 유지되는 동안 `selfplay.evaluator_backend`가 일시적으로 남아 있을 수 있습니다.
업데이트되었지만 최종 권장 구성은 다음과 같이 PyTorch 평가자를 사용해야 합니다.
기본값이며 Tinygrad를 일반 옵션으로 제공해서는 안 됩니다.

## 1단계: PyTorch 네트워크 통합

목표: 평가자, 훈련, 패리티 테스트에 사용되는 하나의 PyTorch 네트워크 정의,
ONNX 내보내기.

일:

- `torch_network.py`를 생성합니다.
- `TorchPolicyValueNet`, `TorchResidualBlock` 및 `TorchSEBlock`을 다음에서 이동합니다.
  `torch_evaluator.py`를 `torch_network.py`로 변환합니다.
- 공유 모델을 가져오려면 'torch_evaluator.py'를 업데이트하세요.
- 'export_onnx.py'를 업데이트하여 공유 모델을 유지하는 대신 공유 모델을 가져옵니다.
  중복 토치 구현.
- 작은 그래드 모양의 가중치를 토치 모델에 로드하기 위한 공유 도우미를 추가합니다.
- 기존 Tinygrad 체크포인트와 호환되는 매개변수 이름을 유지합니다.
  실용적:
  -`stem_conv.weight`
  -`tower.N.conv1.weight`
  -`tower.N.se.fc1.weight`
  - `policy_fc.weight`
  - `value_fc.weight`

유효성 검사 명령:
```bash
uv run --extra omok python scripts/check_omok_torch_parity.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --batches 7,64,93
```
패리티 스크립트에는 다음이 포함되어야 합니다.

- 고정 배치에 대한 평가 모드 순방향 패리티
- 하나의 고정 배치에 대한 열차 모드 순방향 패리티
- BatchNorm `running_mean` / `running_var` 모양 및 로드 후 dtype
  작은그라드 검문소
- 여전히 공유 모델을 사용하여 ONNX 내보내기

승인 기준:

- PyTorch 평가자 패리티는 문서화된 허용 범위 내에서 유지됩니다.
- 훈련 모드 패리티에는 문서화된 허용 오차가 있습니다.
- ONNX 내보내기가 계속 작동합니다.
- 아직 훈련 행동 변화는 없습니다.

## 2단계: Tinygrad 교육을 PyTorch로 대체

목표: `train.py`가 Tinygrad 대신 PyTorch를 직접 사용하도록 합니다. 이것은
두 번째 백엔드가 아닌 교체입니다.

일:

- `train.py`의 Tinygrad 모델 구성을 다음으로 대체합니다.
  `torch_network.PolicyValueNet` 또는 1단계에서 선택한 클래스 이름.
- Tinygrad `AdamW`를 `torch.optim.AdamW`로 교체하세요.
- 현재 Tinygrad 호출 사이트에서 정확한 최적화 하이퍼파라미터를 포팅합니다.
  학습률, 베타, EPS, 가중치 감소 및 그래디언트 클리핑(있는 경우).
- 재생 배치 텐서 생성을 토치 텐서로 대체합니다.
- 리플레이 스토리지를 numpy/deque로 유지하세요. 샘플링된 배치만 토치가 되어야 합니다.
  텐서.
- 최적화 프로그램 업데이트 중에 `model.train()`을 사용하세요.
- 셀프 플레이 및 경기장 평가에는 `model.eval()`을 사용하세요.
- 기존 손실 계산을 유지합니다.
```text
policy_loss = -(target_policy * log_softmax(logits)).sum(axis=1).mean()
value_loss = mean((value - target_value) ** 2)
loss = policy_weight * policy_loss + value_weight * value_loss
```
- 다음과 같은 기존 동작을 정확하게 유지합니다.
  -`최신_온도`
  - 대칭 확대
  - 정책 목표 형태
  - 가치 목표 형태
  - `값_할인`
  - 승격/경기장 게이트 로직
  - MCTS 평가자 인터페이스

재생 API 대상:
```python
ReplayBuffer.sample_batch_numpy(...)
ReplayBuffer.sample_batch_torch(...)
```
`ReplayBuffer.sample_batch(...)`는 직접 토치 경로가 될 수도 있고
통화 사이트가 업데이트된 후 제거되었습니다. 그렇지 않은 경우에는 Tinygrad 샘플링 경로를 유지하지 마십시오.
임시 레거시 가져오기 스크립트에 필요합니다.

이 단계의 범위를 명시적으로 벗어났습니다.

-`torch.compile`
- AMP / 자동 전송 / GradScaler
- `채널_마지막`
- `torch.backends.cudnn.benchmark = True`
- MCTS 재조정
-tinygrad의 최적화 상태 변환

승인 기준:

- 토치 훈련을 통해 빠른 구성이 처음부터 실행됩니다.
- 전체 CUDA 구성은 토치 텐서 및 토치 최적화 프로그램을 사용하여 훈련을 시작합니다.
- 측정항목에는 여전히 단계 타이밍 및 손실 필드가 포함됩니다.
- 훈련 로그에는 PyTorch가 훈련 구현으로 명확하게 식별됩니다.

## 3단계: 토치 체크포인트 및 레거시 중량 가져오기

목표: 오래된 Tinygrad를 허용하면서 `.pt`를 사용하여 토치 훈련을 저장하고 재개합니다.
토치 실행을 시드하기 위한 모델 가중치.

토치 체크포인트 형식:
```python
{
    "checkpoint_format": "coolrl.omok.torch.v1",
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "metadata": {
        "torch_version": "...",
        "iteration": ...,
        "config": ...,
    },
}
```
규칙:

- 새로운 훈련 체크포인트는 `.pt`를 사용합니다.
- `latest.pt`와 `best.pt`는 일반적인 런타임 체크포인트입니다.
- Torch `.pt` 체크포인트는 모델, 최적화 프로그램, 스케줄러, 반복 및
  메타데이터.
- 기존 Tinygrad `.safetensors` 체크포인트는 모델로만 로드될 수 있습니다.
  무게.
-tinygrad 가중치를 로드하면 항상 새로운 토치 최적화 프로그램이 생성됩니다.
- 최적화 프로그램 상태가 복원되지 않은 경우 로그에 명시적으로 표시되어야 합니다.
-tinygrad 최적화 변환을 구현하지 마십시오.
- 새로운 Tinygrad 체크포인트 저장을 지원하지 않습니다.

선택적 도우미 명령:
```bash
uv run --extra omok python -m coolrl.omok.convert_checkpoint \
  --input checkpoints/omok_full_cuda/latest.safetensors \
  --output checkpoints/omok_full_cuda_torch/latest.pt
```
도우미는 유용하지만 `train.py`를 직접 초기화할 수 있는 경우에는 필요하지 않습니다.
레거시 가중치에서.

승인 기준:

- 토치`latest.pt`를 저장하세요.
- 토치`best.pt`를 저장하세요.
- `latest.pt`에서 토치 실행을 재개합니다.
- 오래된 Tinygrad 모델 체크포인트에서 새로운 토치 실행을 초기화합니다.
  최적화 상태.

## 4단계: 일반 경로에서 Tinygrad 제거

목표: 토치 훈련 및 체크포인트가 완료된 후 Tinygrad 코드를 삭제하거나 격리합니다.
일하고 있습니다.

일:

- PyTorch 평가기를 사용하도록 권장 구성을 업데이트합니다.
- 권장 구성에서 Tinygrad 평가자 선택을 제거합니다.
- `train.py`에서 Tinygrad 교육 가져오기를 제거합니다.
- Tinygrad 재생 텐서 생성을 제거합니다.
- Tinygrad 최적화 체크포인트 저장/로드를 제거합니다.
- 권장 설정 경로에서 중복된 Tinygrad 관련 문서를 제거하세요.
- 이전 모델 가중치를 가져오는 데 필요한 최소한의 레거시 코드만 유지합니다.
- 레거시 가져오기가 여전히 `network.py`에 의존하는 경우 해당 종속성을 뒤에 격리합니다.
  명확하게 이름이 지정된 함수 또는 스크립트.

금속 정책:

- Metal에만 Tinygrad를 두지 마십시오.
- 간단하다면 토치 MPS를 사용해 보세요.
- 토치 MPS가 안정적이지 않은 경우 별도로 금속 교육을 지원되지 않음으로 표시합니다.
  고정.
- CPU 훈련은 필요한 경우 토치 CPU를 사용할 수 있습니다.

승인 기준:

- 일반 교육 경로에서는 Tinygrad 모듈을 가져오지 않습니다.
- 권장되는 CUDA 구성은 토치 평가자와 토치 훈련을 사용합니다.
- 기존 MCTS 코드에는 API 변경이 필요하지 않습니다.
- 문서에서는 PyTorch를 여러 백엔드 중 하나가 아닌 트레이너로 설명합니다.

## 5단계: 벤치마크 후 재조정

목표: 먼저 토치 전용 기준선을 측정한 후 저장된 재투자 대상을 결정합니다.
시간.

기준 벤치마크:
```bash
rm -rf checkpoints/omok_full_cuda_torch
uv run --extra omok python -m coolrl.omok.train \
  --config configs/omok_full_cuda.yaml \
  --max-iterations 4
```
트랙:

| 미터법 | 왜 |
|---|---|
| `selfplay_seconds` | 평가자 비용 |
| `train_seconds` | 훈련 비용 |
| `train_optimizer_seconds` | 옵티마이저 병목 현상 검사 |
| `train_samples_per_second` | 벽 시간 잡음과 무관한 처리량 |
| `arena_seconds` | 프로모션 게이트 비용 |
| `duration_seconds` | 전체 루프 비용 |
| `정책_손실` / `값_손실` | 온전한 학습 |
| 경기장 승률 | 붕괴 감지 |
| 가능한 경우 최대 GPU 메모리 | 추후 최적화를 위한 여유 공간 |

첫 번째 토치 전용 벤치마크 중에는 검색 설정을 변경하지 마세요.

기준선이 안정된 후 레이어 토치별 최적화가 한 번에 하나씩 수행됩니다.

1.`torch.backends.cudnn.benchmark = True`
2. `채널_마지막`
3. `토치.컴파일`
4. AMP/오토캐스트

측정된 후에만 MCTS/검색 설정을 스윕합니다.
```text
leaves_per_batch: 8, 16, 32, 64
simulation_schedule: current, then higher final simulations if iteration time falls
arena.games: current, then higher if arena becomes cheap
```
절약된 재투자 주문:

1. 동일한 MCTS 예산을 유지하고 더 많거나 더 빠른 반복이 정상적으로 작동하는지 확인합니다.
2. 점진적으로 토치 최적화 활성화
3. `leaves_per_batch`를 다시 조정합니다.
4. 더 높은 수준의 시뮬레이션이나 더 큰 경기장 게이트를 고려하세요.

## 주요 위험

| 위험 | 완화 |
|---|---|
| BatchNorm 동작 변경 | 평가 및 훈련 모드에서 패리티 검사; 실행 통계 보존 |
| AdamW 드리프트 | Tinygrad 하이퍼파라미터를 명시적으로 포팅합니다. 토치 기본값에 의존하지 마세요 |
| 최적화 프로그램 상태는 이식 가능하지 않습니다 | Tinygrad 체크포인트는 가중치만 로드합니다. 새로운 최적화 프로그램이 예상됩니다 |
| 체크포인트 혼란 | 모든 새로운 훈련 체크포인트에 `.pt`를 사용하세요 |
| 재생 대상 모양 드리프트 | 재생 저장소를 변경하지 않고 유지하고 샘플링된 배치만 변환 |
| 학습 행동 변화 | 손실, 경기장 게이트 및 고정 체크포인트 경기 비교 |
| 금속 회귀 | 토치 MPS 또는 지원되지 않음; Metal만을 위해 Tinygrad를 보존하지 마세요 |
| 조기 최적화 | 컴파일/AMP/channels_last 전에 fp32 토치 기준선을 지정 |
| 검색 재조정으로 백엔드 버그 숨김 | 기본 학습 행동이 정상화된 후에만 재조정 |

## 권장 커밋 순서

1. `공유된 PyTorch Omok 네트워크 추출`
   - `torch_network.py`를 추가하세요.
   - `torch_evaluator.py` 업데이트
   - `export_onnx.py` 업데이트
   - 패리티 스크립트 추가

2. `훈련 모델과 옵티마이저를 토치로 대체`
   - `train.py` 업데이트
   - 토치 재생 일괄 샘플링 추가
   - 현재 손실 수학 및 최적화 하이퍼파라미터 보존

3. `토치 체크포인트 추가 저장 및 재개`
   - `.pt` 체크포인트 작성
   - `.pt`에서 모델 및 최적화 프로그램 재개
   - 로그 체크포인트 형식 및 토치 버전

4. `레거시 Tinygrad 가중치 가져오기 추가`
   - 이전 모델 가중치를 토치 모델에 로드
   - 항상 새로운 토치 옵티마이저를 시작하세요.
   - 이 경로를 격리된 상태로 유지하세요.

5. `일반 훈련 경로에서 Tinygrad 제거`
   - 트레이너에서 Tinygrad 가져오기를 제거합니다.
   - Tinygrad 재생 텐서를 제거합니다.
   - 권장 구성 및 문서 업데이트

6. '벤치마크 토치 전용 CUDA 베이스라인'
   - 고정 반복 벤치마크 실행
   -`docs/omok_cuda_tuning.md`의 문서 타이밍
   - 아직 검색을 재조정하지 마세요

7. `토치 최적화 및 재조정 검색 활성화`
   - 각 최적화를 독립적으로 측정
   -`leaves_per_batch`를 청소합니다.
   - 측정 결과 후에만 CUDA 구성을 업데이트합니다.

## 결정

**PyTorch는 유일한 교육 백엔드입니다.** 백엔드 선택을 도입하거나
병렬 트레이너 추상화.

**훈련 체크포인트는 `.pt`를 사용합니다.** `.safetensors`는 레거시 입력이거나 가능합니다.
새로운 훈련 형식이 아닌 향후 가중치 전용 내보내기 아티팩트입니다.

**Tinygrad 최적화 상태는 삭제됩니다.** 이전 체크포인트는 모델을 시드할 수 있습니다.
가중치가 있지만 토치는 새로운 최적화 프로그램으로 시작됩니다.

**금속은tinygrad를 유지하는 것을 정당화하지 않습니다.** 실행 가능한 경우 토치 MPS를 사용하십시오. 그렇지 않으면
별도로 고정될 때까지 금속 훈련이 지원되지 않음을 표시합니다.

**MCTS는 백엔드 중립성을 유지합니다.** 평가자 인터페이스이면 충분합니다. 하지 마십시오
이 마이그레이션의 일부로 MCTS를 변경합니다.

**마이그레이션 후 재조정이 발생합니다.** 먼저 안정적인 토치 전용 fp32를 구입하세요.
기준선을 정한 다음 최적화하고 다시 조정하세요.

## 아직 열려있습니다

- MCTS 재조정 전에 정확한 "기본 학습이 정상입니다" 신호를 정의합니다.
  실제 기본값은 손실 폭발 없음, 경기장 동작이 명확하지 않음입니다.
  접혀 있고 이전과 유사한 하나 이상의 고정 체크포인트 비교
  동일한 반복 횟수에 대한 기준선.
