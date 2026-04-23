# Omok Web Client — iOS Safari 메모리 조사

이 문서는 15×15 Omok("쿨파고") web client를 iOS Safari에서 실행하는 동안 발생한 두 개의 별도 tab-death 사건과 각 수정 뒤의 진단 추론을 기록합니다. 두 사건은 사용자 관점에서는 동일해 보이지만("게임이 잠시 동안 실행되다가 탭이 죽음") 완전히 다른 root causes와 완전히 다른 메모리 축을 가집니다. 메모리에 이들을 별도로 유지하면 향후 regression을 bisect하기가 더 쉬워집니다.

관련 commits:

- `5687c62` — Omok web UI for iOS Safari 안정화 (WASM era, compositor pressure)
- `d7fd988` — iOS Safari를 위한 WebGPU 메모리 압력 감소 (WebGPU era, inference pressure)
- Phase 3 — 모든 idle release에서 worker.terminate 전 graceful session teardown (첫 번째 시도의 "skip idle-release on GPU" 부분이 rolled back되었음 — 아래 참조)

## Phase 1 — WASM era: WebContent/compositor GPU pressure

Phase 1 당시 클라이언트는 ONNX Runtime Web WASM backend만 사용했습니다. 증상은 다음과 같습니다: 페이지를 열고, 몇 게임을 플레이한 후 탭을 몇 번 새로고침 — 그러면 iOS Safari에서 탭이 조용히 죽습니다. 데스크톱 Chrome은 영향받지 않았습니다.

### What the profile showed

![WASM-era Chrome DevTools profile](assets/omok-web-wasm-profile.png)

DevTools 성능 기록이 보여준 것:

- **JS heap이 flat함.** MCTS tree나 replay-buffer growth가 없음. C-side MCTS 메모리 문제 (참조: `omok-mcts-memory.md`)는 브라우저가 아닌 호스트의 Python training process에 대한 것입니다 — 브라우저의 MCTS는 TS에서 살았고, 각 search마다 할당되고 깔끔하게 tear down되었습니다.
- **GPU 메모리가 올라갔음** 앱이 GPU compute를 하지 않았음에도 불구하고: ORT는 WASM에 있었고, inference는 전적으로 CPU에서 실행되었습니다. GPU 압력은 inference가 아니라 **브라우저 자체 compositor**에서 나왔습니다.
- Frames는 idle 중 지속적인 re-paint activity를 보였습니다. 이것이 힌트입니다.

### Root cause

iOS Safari의 WebContent process는 tight한 메모리 budget을 공유합니다 (대략 1–1.5 GB, compositor와 함께 차지). 페이지는 이 budget에 여러 방식으로 hostile했는데 desktop Chrome에서는 보이지 않습니다:

1. **`backdrop-filter: saturate(180%) blur(22px)`** 모든 glass surface (board card, player cards, turn pill, icon buttons, settings sheet)에 적용. 각 blur는 device pixel ratio에서 full-viewport offscreen pass입니다. iPhone DPR 3 devices with high-resolution viewport에서 이는 compositor가 layer당 tens of megabytes의 texture atlas 비용을 초래하고, frames 전반에 retained됩니다.
2. **A 2.05 s infinite blur-sweep animation** `.turn-pill.thinking`에 적용되어, compositor가 다른 일이 없을 때도 매 frame마다 textures를 invalidate하고 re-upload하도록 했습니다.
3. **Per-frame canvas size sync** `getBoundingClientRect`를 다시 확인하고 rounding이 shift할 때마다 `canvas.width/height`를 재할당하여, 매 frame마다 backing store의 reallocation을 강제했습니다.
4. **`devicePixelRatio`-unlimited canvas** DPR 3 iPhones에서 15×15 board에 대해 9배 oversized backing store를 생성했는데, 이는 board가 필요로 하는 pixels의 roughly an order of magnitude입니다.
5. **Debug panel polling `setInterval(render, 500 ms)`** `<details>` panel이 열려 있는지 여부와 관계없이 실행되어, 백그라운드에서 continuous layout reads + DOM writes을 했습니다.
6. **Retained model `ArrayBuffer` copies** — 매 evaluator reload마다 ~12 MB `best.onnx` bytes의 extra copy를 JS heap에 "redo" convenience를 위해 유지했습니다.

각각 개별적으로는 desktop에서 cheap하고 Android Chrome에서는 invisible합니다. 하지만 iOS Safari에서 함께 작용하면, 두 번째나 세 번째 tab reload에서 WebContent process의 high-water mark를 limit 위로 밀어올렸습니다 — 그 시점에서 Safari는 조용히 tab을 jettison합니다.

### Fix (commit `5687c62`)

모바일/iOS에만 defensively 적용되어 desktop을 pristine하게 유지합니다:

- 모든 `backdrop-filter: blur(...)` 제거 (flat fill로 대체).
- blur-sweep animation 제거 (pseudo에 `display: none`).
- canvas DPR을 iOS에서 1.5로, 다른 모바일에서 2로 capped.
- per-frame canvas size sync 중단; resize 시에만 sync.
- panel이 닫혔을 때 debug panel polling 중단.
- reload 시 browser cache에서 default model을 refetch 하여 ArrayBuffer copy 대신.
- worker feature buffer를 inference calls 전반에 재사용하여 WASM inference가 Float32Arrays를 계속 allocate하지 않도록 함.

이러한 mitigations는 Svelte + Vite + TS refactor를 intact하게 살아남습니다 — 참조: `util/device.ts` (`canvasPixelRatio`, `isLowMemoryMode`), `omok-controller.ts` (`applyDeviceDefaults`, `releaseEvaluatorForIdle`, `MOBILE_MCTS_MAX_CHILDREN`), 및 `evaluator/worker.ts` (`featureBuffer` reuse).

## Phase 2 — WebGPU era: inference-side GPU pressure

ORT Web 1.24 upgrade 후 user-selectable WebGPU backend 추가 (commits `cb108eb`, `7fcdad6`, `770fd16`)로, iOS Safari에서 새로운 tab-death pattern이 나타났습니다 — 하지만 user가 명시적으로 WebGPU를 선택했을 때만. "Auto"는 이미 mobile을 WASM으로 routes합니다. WebGPU가 선택되었을 때, tab은 몇 게임 정도는 fine하게 survive하다가 mid-search에 죽습니다.

### What the profile showed

![WebGPU-era Chrome DevTools profile](assets/omok-web-webgpu-profile.png)

Windows Chrome에서 가져온 profile이지만, shape이 중요합니다:

- **JS heap**은 전체 recording 동안 7–24 MB 근처에서 stable합니다. Normal GC로 flat-ish합니다. **Not** a JS leak.
- **GPU memory** (bottom chart, blue line)은 MCTS bursts와 lockstep으로 sawtoothed합니다 — 모든 search는 GPU memory를 증가시키고, 다음 search 전에 대부분을 release합니다. 중요한 것은, **desktop Chrome은 다음 peak 전에 buffers를 release합니다**, 따라서 high-water mark는 bounded로 유지됩니다.
- Main thread는 MCTS에서 예상대로 idle gaps로 분리된 dense compute bursts를 보였습니다.

Desktop profile은 healthy했습니다. 문제는 iOS가 같은 pattern에서 왜 죽는지였습니다.

### Root cause

Phase 1과 달리, 여기서의 pressure는 compositor side가 아니라 **inference side**에 있습니다. 세 가지 GPU-resident lifetimes이 high-water mark에 contribute합니다:

1. **Per-run input tensor.** `worker.ts`는 모든 `evaluate()` call에서 `new ort.Tensor("float32", features, [batch, 4, N, N])`을 allocate합니다. WebGPU EP와 함께, tensor를 생성하면 GPUBuffer를 allocate합니다. `OrtTensor` JS object는 GC가 해제할 때까지 그 buffer에 대한 reference를 유지합니다. Desktop V8은 이것들을 GC-ing하는 데 aggressive합니다; iOS Safari의 JSC는 훨씬 lazier이므로, last search의 input buffer는 종종 다음 search의 allocation을 outlive합니다.

2. **Per-run output tensors.** ORT WebGPU EP는 session configuration에 따라 outputs을 GPU-resident tensors로 return할 수 있습니다. Output이 GPU-resident로 유지되면, `.data` access는 readback을 강제하지만, GPU copy는 JS `OrtTensor` object가 GC될 때까지 disposed되지 않습니다.

3. **Session-resident state.** Model weights, intermediate activations, 및 scratch buffers는 session의 lifetime 동안 GPU에 살아있습니다. App이 할 수 있는 것은 session을 terminate하는 것 뿐입니다.

iOS Safari는 desktop Chrome과 비교하여 모든 세 가지를 amplify합니다:

- iOS WebGPU는 17.4+에서만 available하고 buffer reclamation이 Chrome의 WebGPU implementation보다 느리고 덜 eager합니다. Buffers는 JS owner가 unreachable해진 후에도 더 오래 hang around합니다.
- WebContent process memory budget은 phase 1과 같은 tight한 1–1.5 GB이며, 이제 실제 WebGPU allocations과 공유됩니다.
- 여러 searches worth의 per-run buffers은 GC가 fire하기 전에 GPU memory에 coexist할 수 있고, cumulative allocation은 budget을 crosses합니다. WebContent는 jettison됩니다. Tab dies합니다.

### Fix (commit `d7fd988`)

세 가지 변경으로 위의 세 lifetime을 각각 공격합니다:

1. **모든 `run()`에서 input + output tensors를 dispose합니다.** `evaluate()`와 `warmUp()`은 이제 호출을 `try/finally`로 wrap하고 input과 모든 returned output tensor에서 `tensor.dispose?.()`을 call합니다. WASM에서는 `dispose`가 effectively a no-op이므로 desktop과 non-iOS behavior는 unchanged입니다. WebGPU에서는 GC를 기다리지 않고 deterministically GPU buffers를 release합니다.

2. **WebGPU/WebNN sessions에서 `preferredOutputLocation: "cpu"`입니다.** ORT에게 `run()`이 resolve하기 전에 outputs을 CPU memory로 copy하도록 tell하므로, GPU-side output buffer는 immediately reuse/release 대상이 됩니다. 위의 output dispose와 함께, 이는 output lifetime을 single `run()`에 pin합니다.

3. **모바일에서 WebGPU/WebNN select options를 disable합니다.** `auto`는 이미 mobile을 WASM으로 유지합니다 (`resolveBackendAttempts(choice, gpu, isMobile)`), 하지만 explicit selection은 여전히 clickable했습니다. 이제 options은 mobile에서 `disabled`이고 "(모바일 불안정)"로 labeled됩니다. Visible (transparency를 위해) 하지만 un-selectable입니다 (unstable path로부터 users를 보호하기 위해).

Session-resident state (cause list의 item 3)은 여기서 다루지 않습니다 (app layer에서 할 수 없으므로). 기존 idle-release path (`releaseEvaluatorForIdle` → mobile의 `terminateEvaluator`)는 이미 user의 turn이 prolonged되거나 game이 끝날 때 worker를 terminate하여 이를 handle합니다 — 하지만 GPU backends에서 그 help가 어떻게 harm으로 변하는지는 아래 phase 3을 참조하세요.

## Phase 3 — WebGPU era, revisited: idle-release churn

Phase 2 mitigations가 shipped 후, iOS Safari에서 명시적으로 WebGPU를 선택한 users는 여전히 mid-game failure를 보고했습니다: 어떤 수의 searches가 진행되면, 빨간 "모델 로드 실패"-class chip이 나타나고 몇 초 후 tab이 죽습니다. Pattern은 phase 2와 distinct했습니다 (smooth cumulative death였던 반면); 이것은 sharp cliff를 hit했습니다.

### Root cause

Phase 1 idle-release (`releaseEvaluatorForIdle` → mobile의 `terminateEvaluator`, 모든 AI move가 끝난 후 400 ms)는 WASM 주위로 설계되었습니다. 그 목표는 user가 생각하는 동안 retained Float32Array buffers와 worker의 JS heap을 drop하는 것이었습니다. WASM에는 harmless하지만, **WebGPU에는 actively harmful합니다**:

1. 모든 AI turn은 `scheduleEvaluatorIdleRelease()`로 끝나 400 ms timer를 queue합니다. Mobile에서는 timer가 `worker.terminate()`을 fire합니다 — `session.release()` call이 없는 hard kill이므로, ORT의 GPU resource cleanup은 driver의 lazy GC path에 fall합니다 (iOS Safari에서는 느림).
2. Human의 다음 move → AI turn → `ensureEvaluatorReady`가 worker를 scratch에서 rebuild합니다. 이는 다음을 의미합니다:
   - 12 MB default model을 `force-cache`를 통해 refetch합니다 (iOS memory pressure 하에서는 miss할 수 있고, `fetchBufferFor`에서 "로드 실패" chip을 produce합니다).
   - WebGPU device를 re-create하고 모든 model weights을 새로운 GPUBuffers에 reupload합니다.
   - 모든 operator의 compute shader pipeline을 recompile합니다.
3. Old session의 GPU state가 finalized되지 않았고 새로운 session이 이미 allocated된 brief window — **peak GPU footprint는 roughly doubles** per turn ~100 ms.
4. Game당 20+ times를 repeat합니다. iOS Safari의 WebContent process는 turn 5와 turn 20 사이 어딘가에서 race를 lose합니다.

Chip ("WebGPU 준비 실패", "로드 실패 [...]" — phrasing은 reinit pipeline의 어디서 failure가 land하는지에 따라 varies합니다)은 오직 첫 번째 visible symptom입니다. 뒤따르는 tab death는 메모리 pressure가 이미 limit을 past했을 때 OS가 WebContent process를 jettison하는 것입니다 몇 초 후.

### Fix (first attempt, rolled back — see below)

- `releaseEvaluatorForIdle`는 WebGPU/WebNN sessions에서 early-return되어 evaluator가 turns 전체에서 alive로 유지되고 iOS는 오직 **one** session lifetime만 survive하면 되었고 twenty는 아닙니다. 이것은 mid-game crash pattern을 stop했습니다.

- Error chips는 이제 underlying error의 첫 ~50 chars를 append합니다 (helper `errorChipDetail`), 따라서 향후 iOS incidents는 actual failure mode를 hide하는 generic "WebGPU 준비 실패" 대신 crumb trail을 남깁니다. (Kept.)

### Follow-up rollback — alive WebGPU session은 iOS에서 CPU를 burn합니다

Fix가 shipped된 직후, iOS profiles는 새로운 문제를 보였습니다: user가 AI move 후 생각하는 동안, tab은 close-to-100% CPU에 머물렀고 DevTools timeline은 *continuous* `Styles recalculated` → `Layout` → `Composite` → `Paint`를 보였습니다. microtask/timer events는 매 few milliseconds마다 fire합니다. WASM sessions는 unaffected했습니다; symptom은 live WebGPU session을 hold한 backends에 unique했습니다. Net result: "WebGPU 가 오히려 느려졌어."

Best-fit hypothesis (우리 code에서 fix하기 어려움): iOS Safari의 WebGPU-to-Metal path는 alive `GPUDevice`를 매 vsync마다 compositor/paint pipeline을 hot으로 유지하는 이유로 treats합니다. 그 keepalive의 일부는 DOM style-recalc로도 leak합니다. Desktop Chrome은 이를 하지 않습니다; WASM은 이를 trigger하지 않습니다 (no `GPUDevice`). 우리 control 하에 있는 유일한 trigger는 "WebGPU session은 turns 전체에서 alive로 유지됩니다" — 즉, 정확히 위의 fix입니다.

Traded-off: far tail에서의 crash vs. WebGPU의 모든 user에 대한 perpetual CPU burn. 후자는 모두, 매 turn에 hit합니다. 우리는 skip을 rolled back했습니다.

### Fix (current)

- `releaseEvaluatorForIdle` once again tears down the evaluator 400 ms
  after every AI move on mobile, **including WebGPU/WebNN sessions**.
  So between turns the GPUDevice dies, iOS's hot pipeline quiets, and
  the CPU burn goes away.

- The teardown now goes through the **graceful dispose path** from the
  follow-up section below. That is the actually-substantive part of
  phase 3 and is kept. The worker awaits `InferenceSession.release()`
  before terminating, so GPU buffers are freed deterministically
  instead of getting stranded in the driver's lazy GC. This shortens
  the "old session's GPU state is still around while the new session
  is being allocated" window that caused the original phase 3 crash
  — the phase 3 crash mode is less likely to recur on top of this
  cleaner teardown, even without the skip.

- Error chip improvements (`errorChipDetail`) are retained.

### Residual cost

Every AI turn still pays a WebGPU re-init cost (~500 ms on iOS: model
weights back to GPU, shader pipelines recompiled). Mitigations from
earlier phases (small MCTS `maxChildren`, short per-search simulation
count, per-run tensor dispose) cap the per-turn GPU peak so the
re-init cost doesn't stack. If consecutive fast turns ever become a
real UX problem, the right follow-up is a **longer idle-release
delay** (e.g. 5-10 s) so quick consecutive moves reuse the session
but genuine long-thinking moves still release. Not implemented today.

### Follow-up: graceful session teardown

User-initiated teardown paths (backend switch, file pick, re-init after
an idle release on WASM) now go through `WorkerEvaluator.dispose()`,
which posts a new `dispose` protocol message to the worker. The worker
calls `InferenceSession.release()` (awaiting GPU resource release) and
only then acks; the main thread then calls `worker.terminate()`. A
3-second timeout falls back to a hard terminate if `release()` hangs
(iOS Safari with a lost WebGPU device can do this).

The idle-release path for WASM sessions still uses the sync
`terminate()` — there's no GPU state to release, and the sync path
keeps the code simple. Page-unmount (`OmokController.dispose`) also
stays sync because the caller is leaving and we don't want to hold up
teardown on a release() that might never answer.

## Phase 4 — missed DOM/layout path during search progress

Later Safari profiling showed a different symptom from the tab-death
incidents above: Chrome spent very little time in rendering, while
Safari repeatedly cycled through `Layout` -> `Styles recalculated` ->
`Composite`, often against the full viewport, with the bundled app JS
and `updateInfo`-adjacent UI code showing up as initiators.

This does not invalidate the earlier memory/GPU conclusions, but it
does correct an important blind spot in the phase 3 reasoning. The
phase 3 note blamed continuous style/layout work primarily on the live
WebGPU session keeping WebKit's compositor pipeline hot. That may still
be a contributing factor, but the app also had a direct DOM mutation
path tied to the inference/search loop.

### What we missed

The suspicious path is not only `updateInfo()` itself. The hotter path
is:

1. `MCTS.run(..., { onProgress })` yields roughly every
   `yieldEveryMs` (default ~14 ms).
2. `OmokController.aiMove()` / `showHint()` update candidates and call
   `statusPresenter.setThinking(...)` from that progress callback.
3. `StatusPresenter.setThinking()` writes the turn-pill class and text
   immediately.

That means a long search can push DOM text/class updates at close to
frame cadence. Chrome/Blink tends to batch this well enough that the
profile looks compute-heavy rather than rendering-heavy. Safari/WebKit
is much more eager to invalidate style/layout for text and class
changes, especially in a compact flex layout where a status pill can
affect surrounding layout.

There was also a separate forced synchronous layout pattern in
`TurnPillText.set()` for animated status changes:

- read `pill.getBoundingClientRect().width`
- write `pill.style.width`
- force flush with `pill.offsetWidth`
- update text
- set width to `auto`
- read `getBoundingClientRect()` again
- write width again inside `requestAnimationFrame`

That pattern did not run for every thinking-progress tick because
`setThinking(..., { animate: false })` bypassed it, but it did run for
normal status renders, flashes, win/error states, and fallback paths.
It is exactly the kind of write -> read -> write sequence that Safari
turns into repeated full-layout work.

The debug panel remains a secondary trigger when open: its render path
reads `canvas.getBoundingClientRect()` and writes `innerHTML`. The
earlier phase 1 fix correctly stopped polling while the panel is
closed, so this is not a default idle-loop problem anymore.

### Fix direction

Keep inference/search progress and DOM writes decoupled:

- Store the latest progress text/candidates in JS state.
- Coalesce status DOM writes through a separate timer/rAF path.
- Throttle progress text updates to human-visible cadence
  (for example ~100-150 ms), not every MCTS yield.
- Avoid layout reads in the same turn as DOM writes. In particular,
  do not animate the status pill by measuring its current and target
  width.
- Prefer opacity/transform text animation, or no animation, over
  width animation for status changes.
- Add containment/compositing hints around the small status UI and the
  canvas (`contain`, `transform: translateZ(0)`, `will-change`) so
  Safari has less reason to invalidate the whole viewport.

This should be treated as an app-level Safari performance bug, separate
from ONNX Runtime memory pressure. If a future profile shows layout
storms while Chrome remains fine, inspect DOM writes made from MCTS
progress callbacks before assuming another ORT/WebGPU leak.

## 계층화된 완화 조치 — 활성 위치별 정리

| 카테고리 | Desktop | Mobile (non-iOS) | iOS Safari |
|---|---|---|---|
| backdrop-filter/blur | 활성화됨 | 비활성화됨 (`ios-low-memory` class not match) | **비활성화됨** |
| Canvas DPR cap | 제한없음 | ≤ 2 | **≤ 1.5** |
| Debug panel polling | `<details>` 열릴 때만 활성화 | 동일 | 동일 |
| MCTS `maxChildren` | Infinity | **48** | **48** |
| Auto backend | WebGPU → WASM fallback | **WASM only** | **WASM only** |
| WebGPU/WebNN option clickable | 가능 | **불가능** | **불가능** |
| Tensor dispose on run | 가능 (WASM에서 no-op) | 가능 | 가능 |
| preferredOutputLocation=cpu | WebGPU/WebNN only | n/a (WASM) | n/a (WASM) |
| Evaluator idle release (WASM session) | 비활성화 | on (AI move 후 400 ms) | on (AI move 후 400 ms) |
| Evaluator idle release (WebGPU/WebNN session) | 비활성화 | **on** (AI move 후 400 ms, graceful dispose 통해) | **on** (AI move 후 400 ms, graceful dispose 통해) |
| Error chip includes exception detail | 가능 | 가능 | 가능 |
| Graceful `session.release()` on teardown | 가능 (async paths only) | 가능 (async paths only) | 가능 (async paths only) |

## 남은 위험 사항 / 미해결 항목

- **ORT Web 버전 드리프트.** Worker는 `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.webgpu.min.js`를 jsdelivr에서 `importScripts`로 로드합니다. 고정된 URL이므로 자동 컨텐츠 드리프트는 없지만, 버전 업그레이드 시 loader의 예상 wasm-glue 파일명이 바뀔 수 있습니다 (`.jsep.*` → `.asyncify.*`는 1.23 → 1.24에서 발생 — commits `cb108eb` / `770fd16` 참조). `ort.env.wasm.wasmPaths = ORT_CDN_BASE` (base URL only)를 설정하여 완화하므로 loader가 자체 파일명을 선택합니다. 이 종류의 버그가 반복되면, `onnxruntime-web`을 `package.json` 의존성으로 전환하고 ESM entry (`onnxruntime-web/webgpu` → `ort.webgpu.bundle.min.mjs`)를 번들링하면서 wasm binary는 CDN에 유지하는 것을 고려하세요.

- **iOS desktop-class iPads.** `isIos` check는 UA와 `MacIntel + maxTouchPoints > 1`로 작동합니다. 현재는 작동하지만 iPadOS의 Safari UA는 불안정합니다.

- **Mobile Chrome (Android)의 WebGPU.** 현재 mobile guard에서 비활성화됩니다. Android WebGPU가 입증되면 다시 검토할 수 있습니다 — guard는 `isMobileDevice`로, `isIos`가 아니므로 현재는 필요한 것보다 더 대규모입니다.

- **Desktop의 Session lifetime.** Desktop은 현재 `releaseEvaluatorForIdle`을 호출하지 않으므로 model weights는 게임 사이에 GPU에 남아있습니다. 노트북에서는 수용할 수 있지만 배터리 모드에서는 잠재적 문제입니다.
