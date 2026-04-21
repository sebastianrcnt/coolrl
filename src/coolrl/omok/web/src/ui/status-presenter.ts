import type { TurnPillText } from "./turn-pill-text";

export type StatusClass = "" | "thinking" | "error" | "win";

export interface DefaultStatus {
  text: string;
  cls: StatusClass;
}

export interface StatusPresenterOptions {
  pill: HTMLElement;
  turnPillText: TurnPillText;
  recompute: () => DefaultStatus;
}

interface Override {
  text: string;
  cls: StatusClass;
}

export class StatusPresenter {
  private static readonly THINKING_RENDER_INTERVAL_MS = 120;

  private readonly pill: HTMLElement;
  private readonly turnPillText: TurnPillText;
  private readonly recompute: () => DefaultStatus;
  private override: Override | null = null;
  private flashTimer: ReturnType<typeof setTimeout> | null = null;
  private thinkingTimer: ReturnType<typeof setTimeout> | null = null;
  private thinkingRaf = 0;
  private lastThinkingRender = 0;
  private pendingThinkingText: string | null = null;

  constructor(options: StatusPresenterOptions) {
    this.pill = options.pill;
    this.turnPillText = options.turnPillText;
    this.recompute = options.recompute;
  }

  hasOverride(): boolean {
    return this.override !== null;
  }

  render(): void {
    this.cancelThinkingRender();
    const status = this.override ?? this.recompute();
    this.applyClass(status.cls);
    this.turnPillText.set(status.text, { animate: true });
  }

  setThinking(text: string): void {
    this.override = { text, cls: "thinking" };
    this.applyClass("thinking");
    this.pendingThinkingText = text;

    const now = performance.now();
    if (now - this.lastThinkingRender >= StatusPresenter.THINKING_RENDER_INTERVAL_MS) {
      this.flushThinkingText(now);
      return;
    }

    if (this.thinkingTimer !== null || this.thinkingRaf) return;
    const delay = Math.max(
      0,
      StatusPresenter.THINKING_RENDER_INTERVAL_MS - (now - this.lastThinkingRender)
    );
    this.thinkingTimer = setTimeout(() => {
      this.thinkingTimer = null;
      this.thinkingRaf = requestAnimationFrame((timestamp) => {
        this.thinkingRaf = 0;
        this.flushThinkingText(timestamp);
      });
    }, delay);
  }

  flash(text: string, cls: StatusClass = "", ttlMs = 2400): void {
    this.override = { text, cls };
    this.render();
    this.clearFlashTimer();
    this.flashTimer = setTimeout(() => {
      this.override = null;
      this.flashTimer = null;
      this.render();
    }, ttlMs);
  }

  clearOverride(): void {
    this.override = null;
    this.cancelThinkingRender();
    this.clearFlashTimer();
  }

  dispose(): void {
    this.cancelThinkingRender();
    this.clearFlashTimer();
    this.override = null;
  }

  private clearFlashTimer(): void {
    if (this.flashTimer !== null) {
      clearTimeout(this.flashTimer);
      this.flashTimer = null;
    }
  }

  private applyClass(cls: StatusClass): void {
    const className = cls ? `turn-pill glass ${cls}` : "turn-pill glass";
    if (this.pill.className !== className) this.pill.className = className;
  }

  private flushThinkingText(timestamp = performance.now()): void {
    if (this.pendingThinkingText === null) return;
    const text = this.pendingThinkingText;
    this.pendingThinkingText = null;
    this.lastThinkingRender = timestamp;
    this.turnPillText.set(text, { animate: false });
  }

  private cancelThinkingRender(): void {
    if (this.thinkingTimer !== null) {
      clearTimeout(this.thinkingTimer);
      this.thinkingTimer = null;
    }
    if (this.thinkingRaf) {
      cancelAnimationFrame(this.thinkingRaf);
      this.thinkingRaf = 0;
    }
    this.pendingThinkingText = null;
  }
}
