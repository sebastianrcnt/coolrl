<script lang="ts">
  import { onMount } from "svelte";
  import { OmokController, type DomRefs } from "./app/omok-controller";
  import { logDebug, logInfo } from "./util/logger";

  let canvas: HTMLCanvasElement;
  let boardCard: HTMLElement;
  let fileInput: HTMLInputElement;
  let fileName: HTMLElement;
  let colorSelect: HTMLSelectElement;
  let backendSelect: HTMLSelectElement;
  let simsSelect: HTMLSelectElement;
  let btnReset: HTMLButtonElement;
  let btnSettings: HTMLButtonElement;
  let btnSheetClose: HTMLButtonElement;
  let btnUndo: HTMLButtonElement;
  let btnAi: HTMLButtonElement;
  let btnConfirm: HTMLButtonElement;
  let cardBlack: HTMLElement;
  let cardWhite: HTMLElement;
  let blackName: HTMLElement;
  let whiteName: HTMLElement;
  let blackSub: HTMLElement;
  let whiteSub: HTMLElement;
  let turnPill: HTMLElement;
  let turnText: HTMLElement;
  let sheet: HTMLElement;
  let sheetBackdrop: HTMLElement;
  let debugPanel: HTMLDetailsElement;
  let debugGrid: HTMLElement;

  onMount(() => {
    logInfo("OmokApp", "onMount");
    const dom: DomRefs = {
      canvas,
      boardCard,
      fileInput,
      fileName,
      colorSelect,
      backendSelect,
      simsSelect,
      btnReset,
      btnSettings,
      btnSheetClose,
      btnUndo,
      btnAi,
      btnConfirm,
      cardBlack,
      cardWhite,
      blackName,
      whiteName,
      blackSub,
      whiteSub,
      turnPill,
      turnText,
      sheet,
      sheetBackdrop,
      debugPanel,
      debugGrid,
    };
    const controller = new OmokController({ dom });
    controller.start();
    logDebug("OmokApp", "controllerStarted");
    return () => {
      logDebug("OmokApp", "controllerDisposed");
      controller.dispose();
    };
  });
</script>

<div class="app">
  <header class="topbar">
    <button class="icon-btn" bind:this={btnReset} title="새 대국" aria-label="새 대국">
      <svg viewBox="0 0 24 24">
        <path d="M3 12a9 9 0 1 0 3-6.7" />
        <path d="M3 4v5h5" />
      </svg>
    </button>
    <div class="title">15×15 오목 <span class="muted">쿨파고</span></div>
    <button class="icon-btn" bind:this={btnSettings} title="설정" aria-label="설정">
      <svg viewBox="0 0 24 24">
        <circle cx="5" cy="12" r="1.6" />
        <circle cx="12" cy="12" r="1.6" />
        <circle cx="19" cy="12" r="1.6" />
      </svg>
    </button>
  </header>

  <div class="players">
    <div class="player-card glass" bind:this={cardBlack} id="card-black">
      <span class="stone black"></span>
      <div class="meta">
        <span class="name" bind:this={blackName}>흑</span>
        <span class="sub" bind:this={blackSub}>-</span>
      </div>
    </div>
    <div class="player-card glass right" bind:this={cardWhite} id="card-white">
      <span class="stone white"></span>
      <div class="meta">
        <span class="name" bind:this={whiteName}>백</span>
        <span class="sub" bind:this={whiteSub}>-</span>
      </div>
    </div>
  </div>

  <div class="turn-row">
    <span class="turn-pill glass" bind:this={turnPill}>
      <span class="dot"></span>
      <span bind:this={turnText}>흑 차례</span>
    </span>
  </div>

  <div class="board-wrap">
    <div class="board-card" bind:this={boardCard} id="board-card">
      <canvas bind:this={canvas} id="board"></canvas>
    </div>
  </div>

  <div class="actions">
    <button class="action-btn" bind:this={btnUndo} title="무르기">
      <svg viewBox="0 0 24 24">
        <path d="M9 14 4 9l5-5" />
        <path d="M4 9h10a6 6 0 0 1 0 12h-3" />
      </svg>
      아.. 실수
    </button>
    <button class="action-btn" bind:this={btnAi} title="추천 수 보기">
      <svg viewBox="0 0 24 24">
        <path d="M12 3l2.4 5.4L20 10l-4.2 3.8L17 20l-5-3-5 3 1.2-6.2L4 10l5.6-1.6L12 3z" />
      </svg>
      <span>쿨파고에게<br>훈수 듣기</span>
    </button>
    <button class="action-btn primary" bind:this={btnConfirm} title="착수 확정" disabled>
      <svg viewBox="0 0 24 24">
        <path d="M5 12l5 5L20 7" />
      </svg>
      가즈아
    </button>
  </div>

  <details class="debug-panel glass-subtle" bind:this={debugPanel} open hidden aria-hidden="true">
    <summary>디버그</summary>
    <div class="debug-grid" bind:this={debugGrid}></div>
  </details>
</div>

<div class="sheet-backdrop" bind:this={sheetBackdrop}></div>
<div class="sheet" bind:this={sheet} role="dialog" aria-modal="true" aria-label="설정">
  <div class="sheet-handle"></div>
  <div class="sheet-title">설정</div>

  <div class="sheet-group">
    <div class="sheet-row">
      <span class="label">내 돌</span>
      <select bind:this={colorSelect} id="color-select">
        <option value="1" selected>흑</option>
        <option value="-1">백(어려움)</option>
      </select>
    </div>
    <div class="sheet-row">
      <span class="label">탐색 횟수</span>
      <select bind:this={simsSelect} id="sims">
        <option value="64">64 · 쉬움</option>
        <option value="96">96 · 중간</option>
        <option value="128" selected>128 · 어려움</option>
        <option value="256">256 · 알파고</option>
      </select>
    </div>
  </div>

  <details class="sheet-group advanced-settings" id="advanced-settings">
    <summary class="sheet-summary">
      <span class="label">고급 설정</span>
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m6 9 6 6 6-6" />
      </svg>
    </summary>
    <div class="sheet-row">
      <span class="label">추론 방식</span>
      <select bind:this={backendSelect} id="backend-select">
        <option value="auto" selected>자동 (권장)</option>
        <option value="wasm">WASM</option>
        <option value="webgpu">WebGPU</option>
        <option value="webnn">WebML (WebNN)</option>
      </select>
    </div>
    <label class="sheet-file" for="file-input">
      <svg viewBox="0 0 24 24">
        <path d="M12 3v12" />
        <path d="m7 8 5-5 5 5" />
        <path d="M5 21h14" />
      </svg>
      <span>ONNX 모델</span>
      <span class="file-name" bind:this={fileName}>기본 모델</span>
      <input type="file" id="file-input" bind:this={fileInput} accept=".onnx">
    </label>
  </details>

  <button class="sheet-close" bind:this={btnSheetClose}>완료</button>
</div>
