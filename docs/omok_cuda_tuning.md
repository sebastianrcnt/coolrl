> 역사적 참고 사항: 이 문서는 이전의 Tinygrad에서 PyTorch로의 전환에서 CUDA 튜닝 측정을 기록합니다. 현재 Omok 런타임은 PyTorch 전용입니다. 훈련 체크포인트는 '.pt'이고, 일반 평가자는 PyTorch이며,tinygrad는 더 이상 런타임 종속성이 아닙니다. 아래의tinygrad 관련 섹션을 현재 설정 지침이 아닌 과거 기준 컨텍스트로 처리하세요.

# 오목 CUDA 셀프 플레이 튜닝 노트

이 노트는 RTX 3090 CUDA 프로필의 측정값을 캡처합니다.
(`configs/omok_full_cuda.yaml`) C MCTS 백엔드를 추가한 후.

## 현재 권장 사항

별도의 NVIDIA GPU에서 CUDA 자체 플레이를 하려면:
```yaml
selfplay:
  mcts_backend: c
  evaluator_backend: torch
  num_workers: 0
  batch_size: 64
  leaves_per_batch: 64
  search_threads: auto
```
CUDA 프로필에 대해 `num_workers: 0`을 유지합니다. 다중 프로세스 작업자 경로 핀
`selfplay_worker.py`에서 CPU에 대한 Tinygrad 추론으로 `num_workers: auto`가 이동합니다.
GPU에서 자체 재생 신경망 추론.

Ryzen 5 5600X 테스트 시스템에서 'search_threads: auto'는 12로 확인됩니다.
논리적 CPU. 이는 약속이 아닌 C 리프 수집 단계의 최대값입니다.
전체 학습 과정에서 12개의 CPU 코어가 계속 사용됩니다.

`evaluator_backend: torch`는 셀프 플레이와 경기장 추론에만 영향을 미칩니다. 훈련,
최적화 상태 및 체크포인트는 여전히 Tinygrad `PolicyValueNet`을 사용합니다. 는
PyTorch 평가자는 최적화 이후 현재의 Tinygrad 가중치에서 재구축되었습니다.
업데이트 및 최고의 모델 승격 이후에는 드롭인 추론으로 유지됩니다.
전체 교육 마이그레이션이 아닌 백엔드.

## 왜 '배치당 나뭇잎 개수: 64'인가요?

C MCTS를 사용하면 트리 순회가 이전 Python 트리 탐색보다 빠릅니다. 는
이전 `leaves_per_batch: 8`은 다음의 최대 자체 재생 추론 배치를 생성했습니다.
```text
64 active games * 8 leaves = 512 positions
```
16으로 올리면 CUDA는 다음을 볼 수 있습니다:
```text
64 active games * 16 leaves = 1024 positions
```
64로 더 높이면 평가자 호출 수가 줄어듭니다.
```text
simulations=256, leaves_per_batch=16 -> 16 eval rounds per search_batch
simulations=256, leaves_per_batch=64 ->  4 eval rounds per search_batch
```
현재 트레이너의 혼합 셀프 플레이에서는 각 소스가 일반적으로 절반을 소유합니다.
전역 배치이므로 관찰된 가장 큰 초기 버킷은 일반적으로 다음과 같습니다.
```text
32 active games * 64 leaves = 2048 positions
```
전체 CUDA 프로필에서 관찰된 셀프 플레이 타이밍:

| 설정 | 반복 | 자기 플레이 시간 | 메모 |
|---|---:|---:|---|
| `배치당 나뭇잎 수: 8` | 2 | ~44초 | 셀프 플레이 JIT 반복 없음 |
| `배치당 나뭇잎 수: 8` | 3 | ~53초 | 더 긴 게임 |
| `배치당 나뭇잎 수: 16` | 2 | ~31초 | 최대 버킷 1024 |
| `배치당 나뭇잎 수: 16` | 3 | ~37초 | 더 긴 게임 |
| `배치당 나뭇잎 수: 64` | 2 | ~20대 | 최대 버킷 2048, 새로운 4회 반복 실행 |
| `배치당 나뭇잎 수: 64` | 3 | ~47초 | 시뮬레이션=160, 혼합 소스 |
| `배치당 나뭇잎 수: 64` | 4 | ~47초 | 시뮬레이션=160, 혼합 소스 |

8에서 16으로 이동하면 이전에는 셀프 플레이 시간이 약 30~40% 감소했습니다.
달린다. 16에서 64로 이동하면 처리량도 향상되었지만 총 반복 시간은
이제 셀프 플레이, 훈련, 경기장으로 분할되어 셀프 플레이 전용 기능이 추가되었습니다.
튜닝은 전체 반복 시간에 미치는 영향이 제한적입니다.

ROCm 또는 Metal 프로필에 'leaves_per_batch: 64'를 맹목적으로 일반화하지 마세요.
백엔드 커널 동작, JIT 동작 및 검색 품질 균형은 다릅니다.
CUDA가 아닌 프로필의 경우 8/16/32/64를 스윕하고 기간과 경기장을 비교합니다.
품질 지표.

## 선택적 TensorRT 평가자

TensorRT 평가자는 내부의 신경망 추론만 가속화합니다.
셀프 플레이 및 아레나 MCTS. PyTorch 교육, 최적화 도구를 대체하지 않습니다.
업데이트, 재생 샘플링 또는 MCTS 트리 탐색.

다음을 사용하여 NVIDIA CUDA 시스템에서 명시적으로 활성화합니다.
```yaml
selfplay:
  evaluator_backend: tensorrt
```
또는 TensorRT가 설치된 경우 CUDA 전용 자동 선택을 허용합니다.
```yaml
selfplay:
  evaluator_backend: auto
```
'auto'는 CUDA 또는 TensorRT를 사용할 수 없을 때 토치 평가기로 대체됩니다.
Apple Silicon 및 Metal/MPS 실행은 토치 평가기를 계속 사용해야 합니다. 텐서RT
Metal 백엔드가 아니며 CUDA가 아닌 경로로 가져오지 않습니다.

다음을 사용하여 선택적 종속성을 설치합니다.
```bash
uv sync --extra omok-tensorrt
```
NVIDIA의 pip 패키지는 기본적으로 지원되는 최신 CUDA 주요 변형으로 설정됩니다.
텐서RT. 머신에 특정 CUDA 주요 버전이 필요한 경우 일치하는 버전을 설치하십시오.
NVIDIA 패키지를 수동으로 설치하세요(예: 'tensorrt-cu12' 또는 'tensorrt-cu13').

유용한 환경 손잡이:

| 변수 | 기본값 | 의미 |
|---|---:|---|
| `COOLRL_TENSORRT_MAX_BATCH` | `4096` | 최대 동적 프로필 배치 |
| `COOLRL_TENSORRT_OPT_BATCH` | `384` | 최적화 프로필 배치 |
| `COOLRL_TENSORRT_FP16` | '1' | GPU가 지원하는 경우 FP16 전술 활성화 |
| `COOLRL_TENSORRT_WORKSPACE_MB` | `2048` | TensorRT 빌더 작업공간 제한 |
| `COOLRL_TENSORRT_CACHE` | `~/.cache/coolrl/tensorrt` | 엔진 캐시 디렉토리; 임시 캐시에 `0`을 설정 |

후보 모델 엔진은 후보 가중치가 변경되므로 비용이 많이 들 수 있습니다.
모든 최적화 단계 후에. 최고의 모델 엔진은 최고의 엔진으로 인해 더 잘 상각됩니다.
모델은 프로모션 시에만 변경됩니다.

## `num_workers: auto`가 아닌 이유

짧은 벤치마크 구성으로 CUDA 단일 프로세스 자체 플레이를 비교했습니다.
동일한 네트워크 크기, 16개 게임, 32개 시뮬레이션을 사용하는 ProcessPool 작업자:

| 경로 | `노동자 수` | 추론 장치 | 기간 | 샘플 |
|---|---:|---|---:|---:|
| CUDA 단일 프로세스 | `0` | 쿠다 | 14.924초 | 787 |
| ProcessPool 작업자 | '자동' | CPU | 44.060초 | 690 |

작업자 경로는 전체적으로 약 3배 느렸고 재생당 약 3.4배 느렸습니다.
샘플. 더 많은 CPU 코어를 사용하지만 각 작업자는 아주 작은 CPU 추론을 수행합니다.
이는 3090의 일괄 추론보다 느립니다.

공유 GPU를 피하는 CPU/Metal 스타일 프로필에 작업자 병렬 처리를 사용합니다.
컨텍스트가 목표입니다. CUDA 프로필의 경우 기본 프로세스에서 자체 재생을 유지하세요.

## 스레드 C MCTS

'search_threads'는 C 백엔드 내부의 트리 수준 병렬성을 제어합니다. 작품
활성 게임/트리로 분할되므로 각 `MctsTree`는 여전히 하나만 변경됩니다.
수집 라운드 중 스레드. 이는 셀프 플레이와 경기장에 적용됩니다.
두 경로 모두 동일한 `MCTS.search_batch(...)` 구현을 호출합니다.

이는 의도적으로 동일 트리 병렬 MCTS가 아닙니다. 가상을 사용하지 않습니다.
손실, 원자성 또는 하나의 트리 내부 잠금. 게임이 많을 때 활용도가 가장 좋습니다
배치에서 활성 상태입니다.

C 노드에 대한 트리 수준 스레딩 및 경기장 할당을 추가한 후 CPU 트리
순회는 더 이상 관찰된 CUDA 프로필 병목 현상이 아닙니다. 짧은 재개
다음을 사용하여 `checkpoints/omok_full_cuda`에서 실행합니다.
```bash
COOLRL_MCTS_TIMING=1 COOLRL_MCTS_TIMING_MAX_CALLS=40 \
uv run python -m coolrl.omok.train \
  --config configs/omok_full_cuda.yaml \
  --resume checkpoints/omok_full_cuda \
  --max-iterations 5
```
대표적인 `cmcts_wrapper.MCTS.search_batch(...)` 타이밍을 보여주었습니다.

| 세그먼트 | 예시 시간 | 의미 | 현황 |
|---|---:|---|---|
| `루트` / `노이즈` | ~0.0004초 | 뿌리 준비 및 Dirichlet 소음 | 병목 현상이 아닙니다 |
| '수집' | ~0.006초 | C 12개의 스레드를 사용하는 MCTS 리프 수집 | 병목 현상이 아닙니다 |
| `평가` | ~1.15초 | CUDA의 `ModelEvaluator.evaluate_features(...)` | 지배적인 병목 현상 |
| '피드' | ~0.009초 | C NN 평가 후 확장/역전파 | 병목 현상이 아닙니다 |
| `추출` / `샘플` | <0.001초 | 정책 추출 및 조치 샘플링 | 병목 현상이 아닙니다 |

일반적인 회선의 경우:
```text
states=32 sims=256 leaves_per_batch=16 threads=12 rounds=16 leaves=8192
collect=0.006s eval=1.15s feed=0.009s total=1.17s
```
`collect`는 측정된 `search_batch` 시간의 약 0.5%인 반면 `eval`은
98% 이상. 이는 CPU 사용률이 대략 1% 정도인 것처럼 보이는 이유를 설명합니다.
또는 `search_threads: auto`를 사용해도 사용 중인 코어가 2개 있습니다. 스레드 C 섹션은 다음과 같습니다.
매우 짧으며 대부분의 벽 시간은 CUDA/tinygrad 평가에 소요됩니다.

## 반복 수준 병목 현상 맵

| 단계 | 코드 경로 | 주요 자원 | 현재 병목 현상이 발생합니까? |
|---|---|---|---|
| 시작/이력서 | `Trainer.__init__`, `_restore_from_checkpoint` | 디스크/CPU | 아니 |
| 셀프 플레이 C잎 컬렉션 | `cmcts_wrapper` -> C `mcts_batch_collect_leaves_threaded` | CPU 스레드 | 아니 |
| 셀프 플레이 NN 평가 | 'ModelEvaluator.evaluate_features' | CUDA/tinygrad | 예 |
| 셀프 플레이 피드/백업 | C `mcts_batch_feed_leaves` | CPU | 아니 |
| 재생 삽입 | `ReplayBuffer.add_game` | CPU/램 | 아니 |
| 최적화 업데이트 | `Trainer.train_model` | CUDA/tinygrad | 여기서는 측정되지 않았습니다. 가능한 2차 병목 현상 |
| 아레나 MCTS | `Arena._advance_games` -> 백엔드 `search_batch` | CPU + 쿠다 | 동일한 평가 병목 현상 발생 가능성 |
| 체크포인트 저장 | `save_model_checkpoints`, `save_runtime_state` | 디스크 | 아니 |

이러한 측정을 고려할 때 영구 C 작업자 풀은 다음에 대한 ROI가 낮습니다.
쿠다 프로필. 모든 C 컬렉션 오버헤드를 제거하더라도 단지 비용만 절약됩니다.
측정된 'search_batch' 시간의 0.5%입니다. 다음 튜닝 대상은 다음과 같습니다.
`ModelEvaluator.evaluate_features`, Tinygrad CUDA 버킷 동작 및
평가자 호출의 수/형태.

## 평가자 마이크로벤치마크

Tinygrad 평가자 대기 시간을 분리하려면 `scripts/bench_omok_evaluator.py`를 사용하십시오.
```bash
uv run python scripts/bench_omok_evaluator.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --backend tinygrad \
  --batches 128,256,512,1024,2048 \
  --warmup 2 \
  --iters 5
```
RTX 3090 프로필의 대표적인 결과:

| 일괄 | 총 평균 | `priors_numpy` 평균 | 메모 |
|---:|---:|---:|---|
| 128 | ~0.059초 | ~0.017초 | 작은 게임 후반 양동이 |
| 256 | ~0.064초 | ~0.022초 | 소형/중형 버킷 |
| 512 | ~0.071초 | ~0.030초 | `leaves_per_batch=16`과 공통 |
| 1024 | ~0.101초 | ~0.047초 | 중간 양동이 |
| 2048 | ~0.115초 | ~0.073초 | `leaves_per_batch=64`를 사용하는 일반적인 초기 버킷 |

Tinygrad는 게으르기 때문에 벤치마크의 'forward_lazy'는 대부분 그래프입니다.
완전한 GPU 완성이 아닌 구축. 주요 동기화 지점이 나타납니다
`priors_numpy`에서 정책 로그가 구체화되어 C에 대해 다시 복사됩니다.
MCTS. 이는 더 큰 배치가 도움이 될 수 있는 이유를 설명합니다. 배치 2048은 약 1.6배에 불과합니다.
배치 512보다 느리지만 `leaves_per_batch=64`는 평가 라운드를 4배 줄입니다.
256개 시뮬레이션에서 16개에 비해.

## PyTorch 평가기 Microbenchmark

동일한 Omok 네트워크 형태를 사용하는 일회성 PyTorch 열성 벤치마크
(`channels=64`, `blocks=6`)은 `uv run --with torch`를 통해 실행되었습니다.
PyTorch를 프로젝트 종속성으로 추가합니다. 환경:
```text
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.11.0+cu130
CUDA: available
```
벤치마크에서는 동등한 PyTorch 모듈을 구성하고 CUDA를 측정했습니다.
명시적 동기화를 통한 순방향, 소프트맥스 및 numpy 구체화.
가중치는 무작위였습니다. 이는 대기 시간 비교이지, 패리티 테스트가 아닙니다.
체크포인트를 저장했습니다.

이제 통합 벤치마크 스크립트에서 토치 평가기를 직접 실행할 수 있습니다.
```bash
uv run --extra omok python scripts/bench_omok_evaluator.py \
  --config configs/omok_full_cuda.yaml \
  --device CUDA \
  --backend torch \
  --batches 128,256,512,1024,2048 \
  --warmup 5 \
  --iters 20
```
| 일괄 | tinygrad 총 평균 | PyTorch 열망 총 평균 | 속도 향상 |
|---:|---:|---:|---:|
| 128 | ~0.0592초 | ~0.0017초 | ~35배 |
| 256 | ~0.0631초 | ~0.0019초 | ~33배 |
| 512 | ~0.0711초 | ~0.0035초 | ~20배 |
| 1024 | ~0.0998초 | ~0.0060초 | ~17배 |
| 2048 | ~0.1162초 | ~0.0116초 | ~10배 |

이는 이전의 대략적인 추정치인 2~4배보다 훨씬 더 큰 격차입니다. 왜냐하면
이전 C MCTS 타이밍에서는 'ModelEvaluator.evaluate_features(...)'가 더 많았습니다.
`search_batch(...)`의 98% 이상, 평가자만 교체하면 그럴듯하게 가능
훈련에 비해 셀프 플레이 및 경기장 평가 비용이 거의 사라집니다.

대략적인 반복 추정이 업데이트되었습니다.

| 시나리오 | 셀프 플레이 | 훈련 | 아레나 | 반복 | 200회 반복 |
|---|---:|---:|---:|---:|---:|
| 현재 작은그라드 | ~47초 | ~48초 | ~48초 | ~135~143초 | ~7.5-8.0h |
| PyTorch 평가자 전용 | ~3~5초 | ~48초 | ~3~5초 | ~55~60대 | ~3.0-3.4h |

정확한 숫자는 여전히 전체 루프 검증이 필요합니다. 마이크로벤치마크는 그렇지 않습니다.
체크포인트 가중치 변환 포함, 평가 후 C MCTS 피드/수집 오버헤드
감소, Python 루프 오버헤드 또는 게임 길이 변경. 충분히 강하고,
그러나 PyTorch 평가기 백엔드를 다음으로 가장 높은 ROI 변화로 만들기 위해.

통합된 PyTorch 평가자는 여전히 2의 거듭제곱 배치 버킷에 패딩됩니다. 안
패딩되지 않은 연기 실행은 많은 고유한 CUDA 모양을 생성하고 '13.3초'를 소비했습니다.
반복을 위한 자체 재생 평가 1. 2의 거듭제곱 패딩으로 모양 세트가 축소되고
동일한 실행을 평가 시간 '3.29초'로 줄였습니다.

PyTorch 평가가 통합되면 `leaves_per_batch`를 다시 스윕해야 합니다. 는
현재 `64` 값은 주로 값비싼 Tinygrad 평가기를 줄이기 위해 선택되었습니다.
전화. PyTorch 평가가 호출 비용을 낮추면 '8', '16' 또는
'32'는 신경 평가를 더 많이 공급/백업하여 검색 품질을 회복할 수 있습니다.
벽에 시간을 많이 들이지 않고 자주.

구현 상태: 이제 CUDA 프로필은 `selfplay.evaluator_backend를 사용합니다.
토치`. Tinygrad 평가기에 대한 CPU 및 CUDA 패리티 검사는 다음 범위 내에 있었습니다.
일반 부동 소수점 허용오차:

| 장치 | 정책 최대 절대비 차이 | 가치 최대 절대비 차이 |
|---|---:|---:|
| CPU, 배치 7 | ~2.8e-9 | ~1.0e-7 |
| CUDA, 배치 64 | ~4.0e-7 | ~1.1e-5 |

8개의 게임과 8개의 시뮬레이션으로 실행되는 작은 C MCTS + CUDA 연기
토치 평가자:
```text
selfplay_seconds=1.184s
eval_selfplay_candidate_seconds=1.141s
eval_selfplay_candidate_calls=68
eval_selfplay_candidate_avg_seconds=0.0168s
```
64개의 게임과 96개의 시뮬레이션을 사용하여 전체 CUDA 반복-1 준비 실행을 수행했지만
훈련/경기장, '25.42s'의 이전 작은 등급 기준에서 삭제되었습니다.
자기 플레이 시간:
```text
selfplay_seconds=3.591s
eval_selfplay_candidate_seconds=3.290s
eval_selfplay_candidate_calls=158
eval_selfplay_candidate_avg_seconds=0.0208s
eval_selfplay_candidate_max_bucket=4096
eval_selfplay_candidate_pad_ratio=1.1685
```
`uv run --extra omok ...`를 사용하거나 PyTorch를 실행하기 전에 설치하십시오.
쿠다 프로필. PyTorch가 설치되어 있지 않으면 `evaluator_backend: torch`가 빠르게 실패합니다.
설치 힌트와 함께.

## 4-반복 단계 벤치마크

`metrics.jsonl`에 단계 타이밍 필드를 추가한 후 빈 데이터베이스에서 새로 실행합니다.
`leaves_per_batch: 64`를 포함한 `checkpoints/omok_full_cuda` 및
`search_threads: auto`가 생성되었습니다:
```bash
rm -rf checkpoints/omok_full_cuda
uv run python -m coolrl.omok.train \
  --config configs/omok_full_cuda.yaml \
  --max-iterations 4
```
| 반복 | 심즈 | 합계 | 셀프 플레이 | 기차 | 아레나 | 체크포인트 | 평균 이동수 | 아레나 WR | 수락됨 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 96 | 25.42초 | 25.42초 | 0.00초 | 0.00초 | 0.84초 | 56.23 | - | - |
| 2 | 96 | 118.47초 | 20.47초 | 57.95초 | 39.56초 | 1.03초 | 53.81 | 0.5000 | 사실 |
| 3 | 160 | 143.83초 | 47.14초 | 42.59초 | 54.09초 | 1.03초 | 54.36 | 0.4167 | 거짓 |
| 4 | 160 | 141.25초 | 47.04초 | 43.44초 | 50.77초 | 1.03초 | 54.41 | 0.4167 | 거짓 |

훈련된 반복 2-4의 경우 평균 단계 시간은 다음과 같습니다.

| 단계 | 평균 |
|---|---:|
| 합계 | 134.52초 |
| 자기 플레이 | 38.22초 |
| 훈련 | 47.99초 |
| 경기장 | 48.14초 |
| 검문소 | 1.03초 |

따라서 현재 CUDA 프로필은 더 이상 하나의 CPU MCTS에 의해 지배되지 않습니다.
섹션. 160개 시뮬레이션에서는 셀프 플레이, 훈련 업데이트, 경기장이 모두 제공됩니다.
물질적 기여자. 향후 최적화는 다음에만 집중하는 것을 피해야 합니다.
자체 플레이 MCTS 탐색.

## 내장된 훈련 지표

이제 트레이너는 일반적인 오목을 진단하는 데 충분한 반복당 필드를 작성합니다.
별도의 마이크로벤치마크를 먼저 실행하지 않고도 CUDA 병목 현상이 발생합니다.

최상위 단계 타이밍:

| 필드 | 의미 |
|---|---|
| `duration_seconds` | 체크포인트 저장을 포함한 전체 반복 벽 시간 |
| `selfplay_seconds` | 전체 셀프 플레이 단계 벽 시간 |
| `train_seconds` | 최적화 프로그램 업데이트 단계 벽 시간 |
| `arena_seconds` | 후보 대 최고의 경기장 벽 시간 |
| `체크포인트_초` | 체크포인트 및 런타임 상태로 시간 절약 |

MCTS 검색 타이밍은 단계별로 그룹화됩니다. 예를 들어
`search_selfplay_candidate_*`, `search_selfplay_best_*`,
`search_arena_candidate_*` 및 `search_arena_best_*`:

| 접미사 | 의미 |
|---|---|
| `_calls` | `MCTS.search_batch(...)` 호출 횟수 |
| `_초` | 평가자 통화를 포함한 총 검색 시간 |
| `_avg_seconds` | 평균 검색 통화 대기 시간 |
| `_states` / `_avg_states` | 검색 호출로 본 활성 게임 상태 |
| `_requested_leaves` | MCTS 리프 방문을 요청했으며 대략적으로 * 시뮬레이션 |
| `_max_states` | 해당 단계에서 확인된 가장 큰 활성 배치 |
| `_max_simulations` | 해당 단계에서 사용된 최대 시뮬레이션 수 |
| `_max_leaves_per_batch` | 해당 단계에서 사용되는 가장 큰 구성 리프 배치 |

평가자 타이밍은 `eval_*` 필드와 동일한 단계 이름으로 그룹화됩니다.

| 접미사 | 의미 |
|---|---|
| `_calls` | 신경 평가자 호출 횟수 |
| `_초` | 총 평가자 실제 시간 |
| `_avg_seconds` | 평균 평가자 호출 대기 시간 |
| `_위치` | 패딩되지 않은 보드 위치 평가 |
| `_pended_positions` | 2의 거듭제곱 Tinygrad 버킷 패딩 후 위치 |
| `_pad_ratio` | 패딩_위치/위치 |
| `_avg_batch` / `_max_batch` | 실제 평가자 배치 크기 |
| `_max_bucket` | 가장 큰 패딩된 Tinygrad 버킷 |
| `_bucket_counts` | 패딩된 버킷 크기당 호출 수 |

교육 업데이트 시기:

| 필드 | 의미 |
|---|---|
| `train_metric_updates` | 측정된 최적화 프로그램 업데이트 |
| `train_metric_samples` | 샘플링된 총 재생 행 |
| `train_sample_seconds` | 샘플 및 텐서 생성 시간 재생 |
| `train_forward_seconds` | 모델 순방향 그래프 생성 시간 |
| `train_loss_seconds` | 정책/가치 손실 그래프 구축 시간 |
| `train_backward_seconds` | 역방향 그래프 생성 시간 |
| `train_optimizer_seconds` | 옵티마이저 단계 그래프 생성/실행 시간 |
| `train_sync_seconds` | 손실 구체화 및 동기화 시간 |
| `train_measured_seconds` | 측정된 훈련 하위 단계의 합계 |

평가자를 격리할 때만 별도의 `scripts/bench_omok_evaluator.py`를 사용하세요.
합성 배치 크기별 대기 시간 또는 Tinygrad를 다른 추론과 비교
백엔드. 일반적인 훈련 실행은 먼저 'metrics.jsonl'에 의존해야 합니다.

## 현재 핸드오프 참고 사항

- C MCTS 순회는 트리 수준 스레딩 및 노드 경기장 이후 빠릅니다.
  할당.
- 'search_threads: auto'는 한계로 유용하지만 CPU 사용률이 높아야 합니다.
  C 컬렉션은 벽 시간의 작은 부분이기 때문에 예상할 수 없습니다.
- `leaves_per_batch: 64`는 평가자를 줄여 초기 CUDA 처리량을 향상시켰습니다.
  Tinygrad에서 전화합니다. PyTorch 평가판이 통합된 후 `8/16/32/64`를 다시 스윕합니다.
  '64'는 더 이상 최고의 품질/속도 균형을 이루지 못할 수 있습니다.
- 가장 유망한 다음 실험은 다음과 같습니다.
  - 셀프 플레이 및 경기장을 위한 PyTorch 평가자 백엔드를 추가합니다.
  - 사용하기 전에tinygrad-to-PyTorch 체크포인트 가중치 패리티를 확인하세요.
    훈련 실행;
  - PyTorch eval 및 `leaves_per_batch: 64`를 사용하여 전체 루프 CUDA 벤치마크를 실행합니다.
  - PyTorch 평가 후 `leaves_per_batch`를 청소합니다.
  - eval이 더 이상 실행되지 않으면 훈련 업데이트가 아주 작은 수준으로 제한되는지 여부를 측정합니다.
    지배적;
  - PyTorch 평가가 완료된 후에만 ONNX Runtime 또는 TensorRT를 고려하십시오.
    훈련은 남아있는 주요 병목 현상이 될 것으로 예상됩니다.
  - '수집'이 의미 있는 경우에만 영구 C 작업자 풀을 다시 방문하세요.
    다시 `search_batch` 시간을 공유합니다.

## 현재 사용되지 않는 참조 필드

이러한 필드는 참조 구성과의 호환성을 위해 구문 분석되지만
현재 C 백엔드는 비동기 추론 대기열을 구현하지 않습니다.
```yaml
selfplay:
  inference_batch_size: 512
  inference_wait_ms: 2.0
```
유용한 미래 방향:

- `inference_batch_size`: 여러 C를 수집하기 위한 대상/한도가 될 수 있습니다.
  평가자를 호출하기 전에 리프 배치를 작성합니다.
- `inference_wait_ms`: 비동기 추론 서버/큐에서만 의미가 있습니다.
