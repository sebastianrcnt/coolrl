import { formatBytes, escapeHtml } from "../util/format";
import { backendLabel, type InferenceBackend } from "../util/backend";
import type { CellMetrics } from "../render/board-geometry";

export interface DebugMetricsSource {
  readonly boardSize: number;
  readonly moveCount: number;
  readonly actionSize: number;
  readonly historyLength: number;
  readonly animCount: number;
  readonly animTicking: boolean;
  readonly ghostSlotCount: number;
  readonly ghostCandidateCount: number;
  readonly ghostTicking: boolean;
  readonly busy: boolean;
  readonly initialSetup: boolean;
  readonly backend: InferenceBackend;
  readonly evaluatorBackend: InferenceBackend | null;
  readonly evaluatorActive: boolean;
  readonly modelName: string | null;
  readonly modelBytes: number;
  readonly defaultModelLoadStarted: boolean;
  readonly simsCount: number;
  readonly maxChildren: number;
  readonly reuseTree: boolean;
  readonly aiProgress: string | null;
  readonly aiTimeMs: number | null;
  readonly aiValue: number | null;
  getCanvas(): HTMLCanvasElement;
  getMetrics(): CellMetrics | null;
}

export interface DebugPanelOptions {
  panel: HTMLDetailsElement;
  grid: HTMLElement;
  startedAt: number;
  source: DebugMetricsSource;
}

export class DebugPanel {
  private readonly panel: HTMLDetailsElement;
  private readonly grid: HTMLElement;
  private readonly startedAt: number;
  private readonly source: DebugMetricsSource;
  private refreshTimer: ReturnType<typeof setInterval> | null = null;

  constructor(options: DebugPanelOptions) {
    this.panel = options.panel;
    this.grid = options.grid;
    this.startedAt = options.startedAt;
    this.source = options.source;
  }

  syncTimer(): void {
    this.clearTimer();
    if (document.visibilityState !== "visible" || !this.panel.open) return;
    this.render();
    this.refreshTimer = setInterval(() => this.render(), 500);
  }

  render(): void {
    if (!this.panel.open || this.panel.hidden) return;
    const src = this.source;
    const canvas = src.getCanvas();
    const rect = canvas.getBoundingClientRect();
    const deviceDpr = window.devicePixelRatio || 1;
    const renderDpr = rect.width > 0 ? canvas.width / rect.width : deviceDpr;
    const supports = [
      (navigator as { gpu?: unknown }).gpu ? "WebGPU" : null,
      "ml" in navigator ? "WebNN" : null,
    ]
      .filter(Boolean)
      .join(" ") || "-";

    let boardMetric = "-";
    const metrics = src.getMetrics();
    if (metrics) {
      boardMetric = `간격 ${metrics.step.toFixed(1)} / 여백 ${metrics.margin.toFixed(1)}`;
    }

    const resolvedBackend =
      src.evaluatorBackend
        ? `${backendLabel(src.backend)}→${backendLabel(src.evaluatorBackend)}`
        : backendLabel(src.backend);
    const evaluatorState = src.evaluatorActive ? "활성" : src.modelName ? "절전" : "-";
    const model = src.modelName ? `${src.modelName} (${formatBytes(src.modelBytes)})` : "-";
    const state = src.initialSetup ? "처음 설정" : src.busy ? "계산/로딩" : "대기";
    const maxChildrenLabel = Number.isFinite(src.maxChildren) ? String(src.maxChildren) : "all";
    const ai =
      src.aiProgress ??
      (src.aiTimeMs !== null
        ? `${(src.aiTimeMs / 1000).toFixed(1)}초 ${src.aiValue !== null ? (src.aiValue >= 0 ? "+" : "") + src.aiValue.toFixed(2) : ""}`
        : "-");

    const nav = navigator as { deviceMemory?: number; hardwareConcurrency?: number };
    const rows: [string, string][] = [
      ["경과", `${((performance.now() - this.startedAt) / 1000).toFixed(0)}초 · ${document.visibilityState}`],
      ["상태", `${state} · 자동로드 ${src.defaultModelLoadStarted ? "시작" : "전"}`],
      ["백엔드", `${resolvedBackend} · ${evaluatorState}`],
      ["탐색", `${src.simsCount} · top ${maxChildrenLabel} · ${src.reuseTree ? "reuse" : "no-reuse"}`],
      ["모델", model],
      ["힙", formatHeap()],
      ["기기", `${nav.deviceMemory ? `${nav.deviceMemory}GB` : "mem n/a"} · dpr ${deviceDpr.toFixed(2)}→${renderDpr.toFixed(2)} · ${nav.hardwareConcurrency ?? "?"}코어`],
      ["지원", supports],
      ["화면", `${Math.round(window.innerWidth)}×${Math.round(window.innerHeight)} · ${Math.round(rect.width)}×${Math.round(rect.height)}css`],
      ["캔버스", `${canvas.width}×${canvas.height}px · ${boardMetric}`],
      ["게임", `${src.moveCount}/${src.actionSize}수 · 기록 ${src.historyLength}`],
      ["애니", `돌 ${src.animCount}/${src.animTicking ? "raf" : "-"} · 고스트 ${src.ghostSlotCount}/${src.ghostCandidateCount}/${src.ghostTicking ? "raf" : "-"}`],
      ["쿨파고", ai],
    ];

    const html = rows
      .map(
        ([key, value]) =>
          `<span class="debug-k">${escapeHtml(key)}</span><span class="debug-v" title="${escapeHtml(value)}">${escapeHtml(value)}</span>`
      )
      .join("");
    if (this.grid.innerHTML !== html) this.grid.innerHTML = html;
  }

  dispose(): void {
    this.clearTimer();
  }

  private clearTimer(): void {
    if (this.refreshTimer !== null) {
      clearInterval(this.refreshTimer);
      this.refreshTimer = null;
    }
  }
}

function formatHeap(): string {
  const mem = (performance as { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
  if (!mem) return "n/a";
  return `${formatBytes(mem.usedJSHeapSize)} / ${formatBytes(mem.totalJSHeapSize)} / ${formatBytes(mem.jsHeapSizeLimit)}`;
}
