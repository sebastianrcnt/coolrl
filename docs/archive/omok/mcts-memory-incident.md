# Omok MCTS 메모리 사건

이 문서는 2026-04-21에 다음을 실행하면서 관찰된 메모리 폭증을 기록합니다:

```bash
uv run python -m coolrl.omok.train --config configs/omok15_full_cuda_hdd.yaml
```

프로세스는 64 GB의 시스템 RAM을 소비했고, 약 8 GB의 swap을 채웠으며, 종료되기 전 심각한 SSD I/O를 유발했습니다. Checkpoint 디렉토리는 startup 상태에만 도달했으므로, replay serialization이나 optimizer training 중이 아니라 첫 번째 15x15 self-play 단계에서 실패가 발생했습니다.

## 증상

- `replay.pkl`이 실질적으로 비어 있음.
- `trainer_state.json`은 `iteration: 0`과 `status: startup` 표시.
- 시스템 RAM이 swap이 소진될 때까지 증가.
- Disk I/O는 checkpoint가 크기 때문이 아니라 OS paging 때문에 급증.

## 근본 원인

직접적인 원인은 arena allocation으로 전환한 후 C MCTS tree lifetime 관리였습니다.

각 C `MctsTree`는 게임 중 할당된 모든 노드에 대한 arena 블록을 소유합니다. 수정 전, `mcts_tree_advance()`는 `tree->root`를 선택된 자식으로 이동했지만 이전 root와 모든 미선택 sibling 브랜치를 보유한 arena 블록을 해제하지 않았습니다. 이로 인해 이전 search 브랜치가 게임의 나머지 부분에서 활성화된 상태로 유지되었습니다.

일부 9x9 runs에서는 숨길 수 있을 정도였지만 15x15에서는 폭발적이 되었습니다:

```text
9x9 action_size  = 81
15x15 action_size = 225
```

초기 게임 expansion은 legal action당 하나의 자식을 생성합니다. 노드당 조밀한 child-pointer array는 broad shallow expansion에서 board-size 비용을 action space에 대해 대략 이차식으로 만듭니다. 15x15 full CUDA profile은 또한 9x9 full CUDA profile보다 더 많은 simulations로 시작하므로, 같은 lifetime 버그가 시스템 레벨 OOM이 되었습니다.

## 수정

C backend는 이제 두 가지를 합니다:

- unexpanded 노드는 더 이상 `children` pointer array를 할당하지 않음;
- `mcts_tree_advance()`는 선택된 자식 subtree만 새로운 arena에 복제한 후 이전 arena를 해제합니다.

이는 선택된 라인에 대한 tree reuse를 유지하면서 모든 이동 후 미선택 브랜치를 삭제합니다. 또한 깊은 tree reuse가 이후 search 결과에 영향을 미치는 기존 C/Python parity 동작도 유지합니다.

Rust backend는 이미 root가 `Box<TreeNode>`이고 선택된 자식이 `take()`로 이동되기 때문에 advance할 때 미선택 브랜치를 삭제했습니다. Rust는 여전히 unexpanded 노드에 대해 child vector를 lazy-allocate하도록 변경되었으므로 15x15 search는 expansion 전 조밀한 child array에 대한 비용을 지불하지 않습니다.

## 운영 지침

- 이전 C backend build로 큰 15x15 profiles를 실행하지 마세요.
- C memory behavior가 긴 self-play runs에서 profiled될 때까지 15x15 full CUDA configs을 `mcts_backend: rust`에 유지하세요.
- RAM이 증가하고 `replay.pkl`이 작으면, replay persistence를 의심하기 전에 MCTS tree lifetime이나 search-batch 메모리를 의심하세요.
- RSS가 주로 iteration boundaries에서 증가하고 `replay.pkl`이 크면, 대신 replay capacity와 checkpoint serialization을 검사하세요.

## 검증

수정은 전체 training을 실행하지 않고도 검증되었습니다:

```bash
uv run python setup.py build_ext --inplace
cargo fmt --manifest-path src/coolrl/omok/rmcts/Cargo.toml --check
cargo test --locked --manifest-path src/coolrl/omok/rmcts/Cargo.toml
uv run --with pytest pytest tests/omok/test_mcts_backends_parity.py tests/omok/test_board_size.py
```

수정 당시 예상 결과:

```text
143 passed, 3 skipped
```
