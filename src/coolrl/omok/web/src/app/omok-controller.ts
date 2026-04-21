import { GameState, type Player } from "../core/game-state";
import { MCTS, type TreeNode } from "../core/mcts";
import { WorkerEvaluator } from "../evaluator/worker-evaluator";
import {
  makeMetricsForCanvas,
  readBoardMarginRatio,
  readBoardThemeColors,
  renderBoard,
} from "../render/board-renderer";
import { pixelToCell, type CellMetrics } from "../render/board-geometry";
import { StoneAnimations } from "../ui/stone-animations";
import { ThinkingGhosts } from "../ui/thinking-ghosts";
import { TurnPillText } from "../ui/turn-pill-text";
import { StatusPresenter, type DefaultStatus } from "../ui/status-presenter";
import { formatSignedValue, formatDuration } from "../util/format";
import {
  backendChoiceLabel,
  normalizeBackendChoice,
  resolveBackendAttempts,
  type BackendChoice,
} from "../util/backend";
import { readDeviceEnvironment, type DeviceEnvironment } from "../util/device";
import { pickLookaheadTemplate, formatLookahead } from "../ui/lookahead-templates";
import {
  emptyModelSource,
  fetchBufferFor,
  hasModel,
  setFromDefault,
  setFromFile,
  type ModelSourceState,
} from "./model-source";
import { DebugPanel, type DebugMetricsSource } from "./debug-panel";
import { logDebug, logError, logInfo, logWarn } from "../util/logger";

const BOARD_SIZE = 15;
const DEFAULT_MODEL_URL = "./best.onnx";
const DEFAULT_SIMS = 128;
const MOBILE_MCTS_MAX_CHILDREN = 48;
const MOBILE_DEFAULT_SIMS = 96;

export interface DomRefs {
  canvas: HTMLCanvasElement;
  boardCard: HTMLElement;
  fileInput: HTMLInputElement;
  fileName: HTMLElement;
  colorSelect: HTMLSelectElement;
  backendSelect: HTMLSelectElement;
  simsSelect: HTMLSelectElement;
  btnReset: HTMLButtonElement;
  btnSettings: HTMLButtonElement;
  btnSheetClose: HTMLButtonElement;
  btnUndo: HTMLButtonElement;
  btnAi: HTMLButtonElement;
  btnConfirm: HTMLButtonElement;
  cardBlack: HTMLElement;
  cardWhite: HTMLElement;
  blackName: HTMLElement;
  whiteName: HTMLElement;
  blackSub: HTMLElement;
  whiteSub: HTMLElement;
  turnPill: HTMLElement;
  turnText: HTMLElement;
  sheet: HTMLElement;
  sheetBackdrop: HTMLElement;
  debugPanel: HTMLDetailsElement;
  debugGrid: HTMLElement;
}

export interface OmokControllerOptions {
  dom: DomRefs;
  boardSize?: number;
  defaultModelUrl?: string;
  env?: DeviceEnvironment;
}

type Cleanup = () => void;

// Turn a thrown value into a short suffix suitable for a status-chip error
// message. Empty string if the error has nothing useful to say. We cap the
// length aggressively because status chips get squeezed on narrow phones.
function errorChipDetail(err: unknown, maxChars = 50): string {
  const raw = err instanceof Error ? err.message : typeof err === "string" ? err : "";
  const trimmed = raw.trim().replace(/\s+/g, " ");
  if (!trimmed) return "";
  const clipped = trimmed.length > maxChars ? `${trimmed.slice(0, maxChars - 1)}…` : trimmed;
  return ` · ${clipped}`;
}

export class OmokController {
  private readonly dom: DomRefs;
  private readonly boardSize: number;
  private readonly defaultModelUrl: string;
  private readonly env: DeviceEnvironment;

  private game: GameState;
  private history: number[] = [];
  private evaluator: WorkerEvaluator | null = null;
  private mcts: MCTS | null = null;
  private aiSubtree: TreeNode | null = null;
  private modelSource: ModelSourceState = emptyModelSource();
  private pendingAction: number | null = null;
  private aiValue: number | null = null;
  private aiTimeMs: number | null = null;
  private aiProgress: string | null = null;
  private busy = false;
  private humanPlayer: Player = 1;
  private backendChoice: BackendChoice = "auto";
  private initialSetup = true;
  private defaultModelLoadStarted = false;
  private canvasCtx: CanvasRenderingContext2D | null = null;
  private idleReleaseTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly stoneAnimations: StoneAnimations;
  private readonly thinkingGhosts: ThinkingGhosts;
  private readonly turnPillText: TurnPillText;
  private readonly statusPresenter: StatusPresenter;
  private readonly debugPanel: DebugPanel;

  private readonly cleanups: Cleanup[] = [];
  private readonly startedAt = performance.now();

  constructor(options: OmokControllerOptions) {
    this.dom = options.dom;
    this.boardSize = options.boardSize ?? BOARD_SIZE;
    this.defaultModelUrl = options.defaultModelUrl ?? DEFAULT_MODEL_URL;
    this.env = options.env ?? readDeviceEnvironment();
    this.game = new GameState(this.boardSize);

    this.stoneAnimations = new StoneAnimations(() => this.redraw());
    this.thinkingGhosts = new ThinkingGhosts({
      getGame: () => this.game,
      isBusy: () => this.busy,
      onTick: () => this.redraw(),
    });
    this.turnPillText = new TurnPillText({
      pill: this.dom.turnPill,
      textNode: this.dom.turnText,
    });
    this.statusPresenter = new StatusPresenter({
      pill: this.dom.turnPill,
      turnPillText: this.turnPillText,
      recompute: () => this.computeDefaultStatus(),
    });
    this.debugPanel = new DebugPanel({
      panel: this.dom.debugPanel,
      grid: this.dom.debugGrid,
      startedAt: this.startedAt,
      source: this.makeDebugMetricsSource(),
    });
    logInfo("OmokController", "constructed", {
      boardSize: this.boardSize,
      defaultModelUrl: this.defaultModelUrl,
      lowMemory: this.env.isLowMemoryMode,
      isMobile: this.env.isMobile,
      isIos: this.env.isIos,
    });
  }

  start(): void {
    logInfo("OmokController", "start", {
      isIos: this.env.isIos,
      initialBoardSize: this.boardSize,
    });
    if (this.env.isIos) {
      document.documentElement.classList.add("ios-low-memory");
    }
    this.applyDeviceDefaults();
    this.bindEvents();

    this.syncCanvasSize();
    this.redraw();
    this.updateInfo();
    this.debugPanel.syncTimer();

    requestAnimationFrame(() => {
      if (this.syncCanvasSize()) this.redraw();
    });
    window.addEventListener("load", this.handleResize);

    this.dom.btnSheetClose.textContent = "시작";
    this.openSheet();
    logInfo("OmokController", "sheetOpened", {
      initialSetup: this.initialSetup,
    });
  }

  dispose(): void {
    logInfo("OmokController", "dispose");
    for (const cleanup of this.cleanups) cleanup();
    this.cleanups.length = 0;
    this.stoneAnimations.clear();
    this.thinkingGhosts.stop(false);
    this.statusPresenter.dispose();
    this.debugPanel.dispose();
    if (this.evaluator) {
      this.evaluator.terminate();
      this.evaluator = null;
    }
    if (this.idleReleaseTimer !== null) {
      clearTimeout(this.idleReleaseTimer);
      this.idleReleaseTimer = null;
    }
  }

  // -------------------------------------------------------------------------
  // Event binding
  // -------------------------------------------------------------------------

  private bindEvents(): void {
    logDebug("OmokController", "bindEvents");
    const on = (
      el: EventTarget,
      type: string,
      handler: EventListener,
      opts?: AddEventListenerOptions
    ): void => {
      el.addEventListener(type, handler, opts);
      this.cleanups.push(() => el.removeEventListener(type, handler, opts));
    };

    on(this.dom.fileInput, "change", (e) =>
      this.handleFileInput(e as Event & { target: HTMLInputElement })
    );
    on(this.dom.colorSelect, "change", () => {
      this.humanPlayer = parseInt(this.dom.colorSelect.value) as Player;
      this.resetGame();
    });
    on(this.dom.backendSelect, "change", () => {
      this.backendChoice = normalizeBackendChoice(this.dom.backendSelect.value);
      this.pendingAction = null;
      this.aiSubtree = null;
      this.aiValue = null;
      this.aiTimeMs = null;
      this.aiProgress = null;
      if (!this.initialSetup) this.reloadModelForBackend();
      else this.updateInfo();
    });
    on(this.dom.simsSelect, "change", () => this.debugPanel.render());
    on(this.dom.btnReset, "click", () => { if (!this.busy) this.resetGame(); });
    on(this.dom.btnUndo, "click", () => this.undo());
    on(this.dom.btnAi, "click", () => {
      if (this.hasModel() && !this.busy) this.showHint();
    });
    on(this.dom.btnConfirm, "click", () => this.confirmPending());
    on(this.dom.btnSettings, "click", () => this.openSheet());
    on(this.dom.btnSheetClose, "click", () => this.finishInitialSetup());
    on(this.dom.sheetBackdrop, "click", () => this.finishInitialSetup());
    on(this.dom.canvas, "click", (e) => this.handleCanvasClick(e as MouseEvent));
    on(this.dom.canvas, "dblclick", (e) => e.preventDefault());
    on(document, "dblclick", (e) => e.preventDefault(), { passive: false });
    on(window, "resize", () => this.handleResize());
    on(window, "orientationchange", () => this.handleResize());
    on(document, "visibilitychange", () => this.debugPanel.syncTimer());
    on(this.dom.debugPanel, "toggle", () => this.debugPanel.syncTimer());

    const themeObserver = new MutationObserver(() => this.redraw());
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    this.cleanups.push(() => themeObserver.disconnect());

    if (window.ResizeObserver) {
      const resizeObserver = new ResizeObserver(() => this.handleResize());
      resizeObserver.observe(this.dom.canvas);
      this.cleanups.push(() => resizeObserver.disconnect());
    }
  }

  // -------------------------------------------------------------------------
  // Model loading
  // -------------------------------------------------------------------------

  private hasModel(): boolean {
    return hasModel(this.modelSource, this.evaluator !== null);
  }

  private async createEvaluator(buf: ArrayBuffer): Promise<WorkerEvaluator> {
    logDebug("OmokController", "createEvaluator.start", {
      boardSize: this.boardSize,
      backendChoice: this.backendChoice,
      hasLowMemory: this.env.isLowMemoryMode,
    });
    const webgpuSupported =
      typeof navigator !== "undefined" &&
      !!(navigator as { gpu?: unknown }).gpu;
    const attempts = resolveBackendAttempts(
      this.backendChoice,
      webgpuSupported,
      this.env.isMobile
    );
    let lastError: unknown = null;
    for (const backend of attempts) {
      try {
        logDebug("OmokController", "createEvaluator.try", { backend });
        return await WorkerEvaluator.fromArrayBuffer(
          buf,
          this.boardSize,
          backend,
          this.env.isLowMemoryMode
        );
      } catch (err) {
        lastError = err;
        logWarn("OmokController", `createEvaluator.backendFailed`, {
          backend,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
    throw lastError ?? new Error("no backend available");
  }

  private async fetchModelBuffer(): Promise<ArrayBuffer> {
    return fetchBufferFor(this.modelSource, this.defaultModelUrl);
  }

  private makeMcts(): MCTS {
    if (!this.evaluator) throw new Error("evaluator not ready");
    return new MCTS({
      cPuct: 1.6,
      evaluator: this.evaluator,
      maxChildren: this.env.isLowMemoryMode ? MOBILE_MCTS_MAX_CHILDREN : Infinity,
    });
  }

  private async ensureEvaluatorReady(statusText = "쿨파고 준비 중"): Promise<boolean> {
    logDebug("OmokController", "ensureEvaluatorReady.start", { statusText });
    if (this.evaluator && this.mcts) return true;
    if (!this.hasModel()) return false;

    this.setBusy(true);
    this.statusPresenter.flash(
      `${statusText}… ${backendChoiceLabel(this.backendChoice)}`,
      "thinking",
      60_000
    );
    await this.disposeEvaluatorAsync();
    this.mcts = null;
    try {
      const buf = await this.fetchModelBuffer();
      this.evaluator = await this.createEvaluator(buf);
      this.mcts = this.makeMcts();
      this.aiSubtree = null;
      this.statusPresenter.clearOverride();
      this.setBusy(false);
      this.updateInfo();
      this.debugPanel.render();
      logInfo("OmokController", "ensureEvaluatorReady.ok");
      return true;
    } catch (err) {
      logError("OmokController", "ensureEvaluatorReady.failed", err);
      this.setBusy(false);
      this.statusPresenter.flash(
        `${backendChoiceLabel(this.backendChoice)} 준비 실패${errorChipDetail(err)}`,
        "error",
        8000
      );
      this.updateInfo();
      return false;
    }
  }

  private async releaseEvaluatorForIdle(): Promise<void> {
    logDebug("OmokController", "releaseEvaluatorForIdle.try");
    if (!this.env.isLowMemoryMode || !this.evaluator || this.busy) return;
    if (!this.initialSetup && !this.isHumanTurn() && !this.game.terminal) return;
    // Release the session gracefully (worker awaits InferenceSession.release
    // before terminate) so GPU buffers aren't stranded in the driver's lazy
    // GC path. For WASM sessions release is effectively a no-op but still
    // safe to call.
    await this.disposeEvaluatorAsync();
    // State may have shifted during the dispose await (user clicked, a new
    // evaluator was created). Only clear downstream state if nobody took
    // over while we were waiting.
    if (this.evaluator !== null) return;
    this.mcts = null;
    this.aiSubtree = null;
    this.updateInfo();
    this.debugPanel.render();
  }

  private terminateEvaluator(): void {
    logDebug("OmokController", "terminateEvaluator");
    const ev = this.evaluator;
    if (ev) ev.terminate();
    this.evaluator = null;
  }

  // Graceful async teardown: await the worker's InferenceSession.release()
  // before terminating. Used by user-initiated paths (file pick, backend
  // switch, pre-reinit) where we're already in an async handler and can
  // afford to wait ~100-200 ms for deterministic GPU resource release.
  private async disposeEvaluatorAsync(): Promise<void> {
    logDebug("OmokController", "disposeEvaluatorAsync");
    const ev = this.evaluator;
    this.evaluator = null;
    if (ev) await ev.dispose();
  }

  private scheduleEvaluatorIdleRelease(): void {
    logDebug("OmokController", "scheduleEvaluatorIdleRelease", {
      isLowMemoryMode: this.env.isLowMemoryMode,
    });
    if (this.idleReleaseTimer !== null) clearTimeout(this.idleReleaseTimer);
    this.idleReleaseTimer = null;
    if (!this.env.isLowMemoryMode) return;
    this.idleReleaseTimer = setTimeout(() => {
      this.idleReleaseTimer = null;
      void this.releaseEvaluatorForIdle().catch((err) => {
        logWarn("OmokController", "releaseEvaluatorForIdle.failed", {
          error: err instanceof Error ? err.message : String(err),
        });
      });
    }, 400);
  }

  private async handleFileInput(e: Event & { target: HTMLInputElement }): Promise<void> {
    const file = e.target.files?.[0];
    if (!file) return;
    logInfo("OmokController", "fileInput.selected", { fileName: file.name, size: file.size });
    this.setBusy(true);
    this.statusPresenter.flash(
      `모델 로딩 중… ${backendChoiceLabel(this.backendChoice)}`,
      "thinking",
      60_000
    );
    await this.disposeEvaluatorAsync();
    this.mcts = null;
    try {
      const buf = await file.arrayBuffer();
      this.evaluator = await this.createEvaluator(buf);
      this.mcts = this.makeMcts();
      this.modelSource = setFromFile(file, buf, this.env.isLowMemoryMode);
      this.dom.fileName.textContent = file.name;
      this.dom.fileName.title = file.name;
      this.aiSubtree = null;
      this.statusPresenter.clearOverride();
      this.setBusy(false);
      this.updateInfo();
      this.maybeAiMove();
      this.scheduleEvaluatorIdleRelease();
      logInfo("OmokController", "fileInput.loaded", {
        fileName: file.name,
        bufferBytes: buf.byteLength,
      });
    } catch (err) {
      logError("OmokController", "fileInput.load.failed", err);
      this.statusPresenter.flash(`로드 실패${errorChipDetail(err)}`, "error");
      this.setBusy(false);
      this.updateInfo();
    }
  }

  private async reloadModelForBackend(): Promise<void> {
    logInfo("OmokController", "reloadModelForBackend.start", {
      backendChoice: this.backendChoice,
    });
    if (this.busy) return;
    if (!this.hasModel()) {
      this.defaultModelLoadStarted = false;
      this.tryAutoLoadDefault();
      return;
    }
    this.setBusy(true);
    this.statusPresenter.flash(
      `추론 방식 변경 중… ${backendChoiceLabel(this.backendChoice)}`,
      "thinking",
      60_000
    );
    await this.disposeEvaluatorAsync();
    this.mcts = null;
    try {
      const buf = await this.fetchModelBuffer();
      this.evaluator = await this.createEvaluator(buf);
      this.mcts = this.makeMcts();
      this.aiSubtree = null;
      this.statusPresenter.clearOverride();
      this.setBusy(false);
      this.updateInfo();
      this.maybeAiMove();
      this.scheduleEvaluatorIdleRelease();
      logInfo("OmokController", "reloadModelForBackend.ok", {
        backendChoice: this.backendChoice,
      });
    } catch (err) {
      logError("OmokController", "reloadModelForBackend.failed", err);
      this.setBusy(false);
      this.statusPresenter.flash(
        `${backendChoiceLabel(this.backendChoice)} 사용 불가${errorChipDetail(err)}`,
        "error",
        8000
      );
      this.updateInfo();
    }
  }

  private async tryAutoLoadDefault(): Promise<void> {
    if (this.defaultModelLoadStarted || this.evaluator) return;
    logInfo("OmokController", "autoLoadDefault.start");
    this.defaultModelLoadStarted = true;
    if (location.protocol === "file:") {
      logWarn("OmokController", "autoLoadDefault.fileProtocol");
      this.statusPresenter.flash(
        "file:// 에서는 모델을 가져올 수 없습니다 — HTTP 서버로 열어주세요",
        "error",
        8000
      );
      return;
    }
    this.setBusy(true);
    this.statusPresenter.flash(
      `기본 모델 로딩 중… ${backendChoiceLabel(this.backendChoice)}`,
      "thinking",
      120_000
    );
    let stage = "다운로드";
    try {
      const resp = await fetch(this.defaultModelUrl, { cache: "force-cache" });
      if (!resp.ok)
        throw new Error(`HTTP ${resp.status} on ${this.defaultModelUrl}`);
      stage = "모델 읽기";
      const buf = await resp.arrayBuffer();
      stage = `추론 준비 (${(buf.byteLength / 1024 / 1024).toFixed(1)}MB)`;
      await this.disposeEvaluatorAsync();
      this.evaluator = await this.createEvaluator(buf);
      this.mcts = this.makeMcts();
      this.modelSource = setFromDefault(buf, this.env.isLowMemoryMode);
      const { name, title } = this.modelSource;
      this.dom.fileName.textContent = name ?? "";
      this.dom.fileName.title = title ?? "";
      this.aiSubtree = null;
      this.statusPresenter.clearOverride();
      this.setBusy(false);
      this.updateInfo();
      this.maybeAiMove();
      this.scheduleEvaluatorIdleRelease();
      logInfo("OmokController", "autoLoadDefault.ok", {
        modelUrl: this.defaultModelUrl,
        sizeMB: Number((buf.byteLength / 1024 / 1024).toFixed(1)),
      });
    } catch (err) {
      logError("OmokController", `autoLoadDefault.failed@${stage}`, err);
      this.setBusy(false);
      const msg =
        err instanceof Error ? err.message.slice(0, 80) : "알 수 없음";
      this.statusPresenter.flash(`로드 실패 [${stage}]: ${msg}`, "error", 8000);
      this.updateInfo();
    }
  }

  // -------------------------------------------------------------------------
  // Game actions
  // -------------------------------------------------------------------------

  private isHumanTurn(): boolean {
    if (this.game.terminal) return false;
    if (!this.hasModel()) return true;
    return this.game.toPlay === this.humanPlayer;
  }

  private resetGame(): void {
    logInfo("OmokController", "resetGame");
    this.thinkingGhosts.stop(false);
    this.game = new GameState(this.boardSize);
    this.history = [];
    this.aiSubtree = null;
    this.pendingAction = null;
    this.aiValue = null;
    this.aiTimeMs = null;
    this.aiProgress = null;
    this.statusPresenter.clearOverride();
    this.stoneAnimations.clear();
    this.updateInfo();
    this.redraw();
    this.maybeAiMove();
  }

  private undo(): void {
    logInfo("OmokController", "undo", {
      historyLength: this.history.length,
      pendingAction: this.pendingAction,
    });
    if (this.busy) return;
    this.thinkingGhosts.stop(false);
    if (this.pendingAction !== null) {
      this.pendingAction = null;
      this.updateInfo();
      this.redraw();
      return;
    }
    if (this.history.length === 0) return;
    const count = this.evaluator && this.history.length >= 2 ? 2 : 1;
    this.history.splice(-count, count);
    this.game = new GameState(this.boardSize);
    this.aiSubtree = null;
    this.stoneAnimations.clear();
    for (const action of this.history) this.game.applyAction(action);
    this.statusPresenter.clearOverride();
    this.updateInfo();
    this.redraw();
  }

  private handleCanvasClick(e: MouseEvent): void {
    logDebug("OmokController", "canvasClick", {
      x: e.clientX,
      y: e.clientY,
      busy: this.busy,
      terminal: this.game.terminal,
    });
    if (this.busy || this.game.terminal) return;
    if (!this.isHumanTurn()) return;
    const rect = this.dom.canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (this.dom.canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (this.dom.canvas.height / rect.height);
    const marginRatio = readBoardMarginRatio();
    const metrics = makeMetricsForCanvas(this.dom.canvas, this.boardSize, marginRatio);
    const cell = pixelToCell(metrics, this.boardSize, x, y);
    if (!cell) return;
    const action = cell.row * this.boardSize + cell.col;
    if (this.game.board[action] !== 0) return;
    this.pendingAction = this.pendingAction === action ? null : action;
    this.updateInfo();
    this.redraw();
  }

  private confirmPending(): void {
    logDebug("OmokController", "confirmPending");
    if (this.busy || this.pendingAction === null) return;
    const action = this.pendingAction;
    this.pendingAction = null;
    this.playMove(action);
  }

  private playMove(action: number): void {
    logInfo("OmokController", "playMove", {
      action,
      move: this.game.moveCount + 1,
      toPlay: this.game.toPlay,
    });
    this.game.applyAction(action);
    this.history.push(action);
    this.stoneAnimations.start(action);
    if (this.reuseSearchTree() && this.aiSubtree?.children.has(action)) {
      this.aiSubtree = this.aiSubtree!.children.get(action) ?? null;
    } else {
      this.aiSubtree = null;
    }
    this.updateInfo();
    this.redraw();
    if (!this.game.terminal) this.maybeAiMove();
  }

  private maybeAiMove(): void {
    logDebug("OmokController", "maybeAiMove.check", {
      initialSetup: this.initialSetup,
      hasModel: this.hasModel(),
      busy: this.busy,
      terminal: this.game.terminal,
      toPlay: this.game.toPlay,
      human: this.humanPlayer,
    });
    if (this.initialSetup) return;
    if (!this.hasModel() || this.busy || this.game.terminal) return;
    if (this.game.toPlay !== this.humanPlayer) this.aiMove();
  }

  private async aiMove(): Promise<void> {
    logInfo("OmokController", "aiMove.start");
    if (this.game.terminal || this.busy) return;
    if (!(await this.ensureEvaluatorReady("쿨파고 깨우는 중"))) return;
    if (this.game.terminal || this.game.toPlay === this.humanPlayer) {
      this.scheduleEvaluatorIdleRelease();
      return;
    }
    this.setBusy(true);
    this.thinkingGhosts.start(this.game.toPlay);
    const sims = this.readSimsValue();
    const template = pickLookaheadTemplate();
    this.aiProgress = formatLookahead(template, 1);
    this.statusPresenter.setThinking(this.aiProgress);
    this.updateInfo();
    const t0 = performance.now();
    try {
      const result = await this.mcts!.run(this.game, sims, {
        reuseRoot: this.reuseSearchTree() ? this.aiSubtree : null,
        onProgress: (done, _total, candidates) => {
          this.thinkingGhosts.updateCandidates(candidates);
          this.aiProgress = formatLookahead(template, done);
          this.statusPresenter.setThinking(this.aiProgress);
        },
      });
      this.aiTimeMs = performance.now() - t0;
      this.aiValue = result.rootValue;
      this.aiProgress = null;
      this.statusPresenter.clearOverride();
      this.thinkingGhosts.stop(false);
      this.game.applyAction(result.action);
      this.history.push(result.action);
      this.stoneAnimations.start(result.action);
      this.aiSubtree = this.reuseSearchTree() ? result.nextRoot : null;
      logInfo("OmokController", "aiMove.done", {
        action: result.action,
        rootValue: result.rootValue,
        thinkMs: this.aiTimeMs,
      });
      this.updateInfo();
      this.redraw();
    } catch (err) {
      logError("OmokController", "aiMove.failed", err);
      this.aiProgress = null;
      this.statusPresenter.clearOverride();
      this.thinkingGhosts.stop(false);
      this.statusPresenter.flash(`쿨파고 오류${errorChipDetail(err)}`, "error");
    }
    this.setBusy(false);
    this.updateInfo();
    this.scheduleEvaluatorIdleRelease();
  }

  private async showHint(): Promise<void> {
    logInfo("OmokController", "hint.start");
    if (this.game.terminal || this.busy || !this.isHumanTurn()) return;
    if (!(await this.ensureEvaluatorReady("훈수 준비 중"))) return;
    if (this.game.terminal || !this.isHumanTurn()) {
      this.scheduleEvaluatorIdleRelease();
      return;
    }
    const sims = this.readSimsValue();
    const template = pickLookaheadTemplate();
    this.statusPresenter.setThinking(formatLookahead(template, 1));
    this.setBusy(true);
    this.thinkingGhosts.start(this.game.toPlay);
    const t0 = performance.now();
    try {
      const result = await this.mcts!.run(this.game, sims, {
        reuseRoot: this.reuseSearchTree() ? this.aiSubtree : null,
        onProgress: (done, _total, candidates) => {
          this.thinkingGhosts.updateCandidates(candidates);
          this.statusPresenter.setThinking(formatLookahead(template, done));
        },
      });
      this.pendingAction = result.action;
      this.statusPresenter.clearOverride();
      this.thinkingGhosts.stop(false);
      this.setBusy(false);
      this.redraw();
      const elapsed = performance.now() - t0;
      logInfo("OmokController", "hint.done", {
        action: result.action,
        rootValue: result.rootValue,
        thinkMs: elapsed,
      });
      this.statusPresenter.flash(
        `추천 수 표시 · 형세 ${formatSignedValue(result.rootValue)} · ${formatDuration(elapsed)}`,
        "thinking",
        1800
      );
      this.scheduleEvaluatorIdleRelease();
    } catch (err) {
      logError("OmokController", "hint.failed", err);
      this.statusPresenter.clearOverride();
      this.thinkingGhosts.stop(false);
      this.setBusy(false);
      this.statusPresenter.flash(`힌트 계산 실패${errorChipDetail(err)}`, "error");
      this.scheduleEvaluatorIdleRelease();
    }
  }

  // -------------------------------------------------------------------------
  // UI helpers
  // -------------------------------------------------------------------------

  private setBusy(busy: boolean): void {
    logDebug("OmokController", "setBusy", { busy });
    this.busy = busy;
    this.dom.btnReset.disabled = busy;
    this.dom.btnSettings.disabled = busy;
    this.dom.backendSelect.disabled = busy;
    this.dom.btnUndo.disabled =
      busy || (this.history.length === 0 && this.pendingAction === null);
    this.dom.btnAi.disabled =
      busy || !this.hasModel() || this.game.terminal || !this.isHumanTurn();
    this.dom.btnConfirm.disabled = busy || this.pendingAction === null;
  }

  private openSheet(): void {
    this.dom.sheet.classList.add("open");
    this.dom.sheetBackdrop.classList.add("open");
  }

  private closeSheet(): void {
    this.dom.sheet.classList.remove("open");
    this.dom.sheetBackdrop.classList.remove("open");
  }

  private finishInitialSetup(): void {
    if (!this.initialSetup) {
      this.closeSheet();
      return;
    }
    this.backendChoice = normalizeBackendChoice(this.dom.backendSelect.value);
    this.initialSetup = false;
    this.dom.btnSheetClose.textContent = "완료";
    this.closeSheet();
    this.updateInfo();
    if (this.hasModel()) this.maybeAiMove();
    else this.tryAutoLoadDefault();
  }

  private applyDeviceDefaults(): void {
    if (this.env.isMobile && this.dom.simsSelect.value === String(DEFAULT_SIMS)) {
      this.dom.simsSelect.value = String(MOBILE_DEFAULT_SIMS);
    }
    if (this.env.isMobile) {
      // WebGPU/WebNN on iOS Safari OOM-kills the tab under sustained MCTS
      // load; keep the warning visible, but still honor explicit user choice.
      for (const option of Array.from(this.dom.backendSelect.options)) {
        if (option.value === "webgpu" || option.value === "webnn") {
          option.title = "모바일 브라우저에서는 불안정";
          if (!option.textContent?.includes("(모바일 불안정)")) {
            option.textContent = `${option.textContent} (모바일 불안정)`;
          }
        }
      }
    }
  }

  private readSimsValue(): number {
    return parseInt(this.dom.simsSelect.value) || DEFAULT_SIMS;
  }

  private reuseSearchTree(): boolean {
    return !this.env.isLowMemoryMode;
  }

  private computeDefaultStatus(): DefaultStatus {
    const toPlayIsBlack = this.game.toPlay === 1;
    if (this.game.terminal) {
      const text =
        this.game.winner === 0
          ? "무승부"
          : (this.game.winner === 1 ? "흑" : "백") + " 승리";
      return { text, cls: "win" };
    }
    if (this.initialSetup) return { text: "돌 색을 골라주세요", cls: "" };
    if (this.busy) return { text: "쿨파고가 생각하는 중입니다…", cls: "thinking" };
    if (!this.hasModel()) {
      return { text: toPlayIsBlack ? "흑 차례" : "백 차례", cls: "" };
    }
    if (this.isHumanTurn()) {
      return {
        text: this.pendingAction !== null ? "확정을 눌러주세요" : "내 차례",
        cls: "",
      };
    }
    return { text: "쿨파고 차례", cls: "" };
  }

  private updateInfo(): void {
    const toPlayIsBlack = this.game.toPlay === 1;
    const aiIsBlack = this.hasModel() ? this.humanPlayer === -1 : false;

    this.dom.blackName.textContent =
      this.humanPlayer === 1 ? "흑(나)" : "흑(쿨파고)";
    this.dom.whiteName.textContent =
      this.humanPlayer === -1 ? "백(나)" : "백(쿨파고)";

    this.dom.cardBlack.classList.toggle(
      "active",
      !this.game.terminal && toPlayIsBlack
    );
    this.dom.cardWhite.classList.toggle(
      "active",
      !this.game.terminal && !toPlayIsBlack
    );

    const humanMoves = `${this.game.moveCount}수`;
    let aiSubText: string;
    if (!this.hasModel()) {
      aiSubText = "수동";
    } else if (this.aiProgress) {
      aiSubText = "생각 중...";
    } else if (!this.evaluator) {
      aiSubText = "절전 대기";
    } else if (this.aiValue !== null) {
      const t = this.aiTimeMs !== null ? ` · ${formatDuration(this.aiTimeMs)}` : "";
      aiSubText = `형세 ${formatSignedValue(this.aiValue)}${t}`;
    } else {
      aiSubText = "대기";
    }

    if (this.hasModel()) {
      if (aiIsBlack) {
        this.dom.blackSub.textContent = aiSubText;
        this.dom.whiteSub.textContent = humanMoves;
      } else {
        this.dom.whiteSub.textContent = aiSubText;
        this.dom.blackSub.textContent = humanMoves;
      }
    } else {
      this.dom.blackSub.textContent = humanMoves;
      this.dom.whiteSub.textContent = humanMoves;
    }

    this.dom.btnUndo.disabled =
      this.busy || (this.history.length === 0 && this.pendingAction === null);
    this.dom.btnAi.disabled =
      this.busy || !this.hasModel() || this.game.terminal || !this.isHumanTurn();
    this.dom.btnConfirm.disabled = this.busy || this.pendingAction === null;

    if (!this.statusPresenter.hasOverride()) {
      this.statusPresenter.render();
    }
    this.debugPanel.render();
  }

  // -------------------------------------------------------------------------
  // Canvas / rendering
  // -------------------------------------------------------------------------

  private syncCanvasSize(): boolean {
    const canvas = this.dom.canvas;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    const dpr = this.env.canvasPixelRatio;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      return true;
    }
    return false;
  }

  private getCurrentMetrics(): CellMetrics | null {
    if (this.dom.canvas.width === 0) return null;
    return makeMetricsForCanvas(
      this.dom.canvas,
      this.boardSize,
      readBoardMarginRatio()
    );
  }

  private handleResize = (): void => {
    if (this.syncCanvasSize()) this.redraw();
    this.debugPanel.render();
  };

  private redraw(): void {
    const ctx =
      this.canvasCtx ??
      (this.canvasCtx = this.dom.canvas.getContext("2d")!);
    const metrics = this.getCurrentMetrics();
    if (!metrics) return;
    renderBoard({
      ctx,
      game: this.game,
      metrics,
      theme: readBoardThemeColors(),
      pendingAction: this.pendingAction,
      humanPlayer: this.humanPlayer,
      isHumansTurn: this.isHumanTurn(),
      stoneAnimAt: (action) => this.stoneAnimations.progress(action),
      ghostSlots: this.thinkingGhosts.slots,
      now: performance.now(),
    });
  }

  // -------------------------------------------------------------------------
  // Debug metrics source (read-only snapshot for DebugPanel)
  // -------------------------------------------------------------------------

  private makeDebugMetricsSource(): DebugMetricsSource {
    const ctrl = this;
    return {
      get boardSize() { return ctrl.boardSize; },
      get moveCount() { return ctrl.game.moveCount; },
      get actionSize() { return ctrl.game.actionSize; },
      get historyLength() { return ctrl.history.length; },
      get animCount() { return ctrl.stoneAnimations.size; },
      get animTicking() { return ctrl.stoneAnimations.isTicking; },
      get ghostSlotCount() { return ctrl.thinkingGhosts.slots.length; },
      get ghostCandidateCount() { return ctrl.thinkingGhosts.candidateCount; },
      get ghostTicking() { return ctrl.thinkingGhosts.isTicking; },
      get busy() { return ctrl.busy; },
      get initialSetup() { return ctrl.initialSetup; },
      get backendChoice() { return ctrl.backendChoice; },
      get evaluatorBackend() { return ctrl.evaluator?.backend ?? null; },
      get evaluatorActive() { return ctrl.evaluator !== null; },
      get modelName() { return ctrl.modelSource.name; },
      get modelBytes() { return ctrl.modelSource.bytes; },
      get defaultModelLoadStarted() { return ctrl.defaultModelLoadStarted; },
      get simsCount() { return ctrl.readSimsValue(); },
      get maxChildren() {
        return ctrl.env.isLowMemoryMode ? MOBILE_MCTS_MAX_CHILDREN : Infinity;
      },
      get reuseTree() { return ctrl.reuseSearchTree(); },
      get aiProgress() { return ctrl.aiProgress; },
      get aiTimeMs() { return ctrl.aiTimeMs; },
      get aiValue() { return ctrl.aiValue; },
      getCanvas() { return ctrl.dom.canvas; },
      getMetrics() { return ctrl.getCurrentMetrics(); },
    };
  }
}
