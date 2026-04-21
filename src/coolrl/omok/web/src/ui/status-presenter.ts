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
  private readonly pill: HTMLElement;
  private readonly turnPillText: TurnPillText;
  private readonly recompute: () => DefaultStatus;
  private override: Override | null = null;
  private flashTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(options: StatusPresenterOptions) {
    this.pill = options.pill;
    this.turnPillText = options.turnPillText;
    this.recompute = options.recompute;
  }

  hasOverride(): boolean {
    return this.override !== null;
  }

  render(): void {
    const status = this.override ?? this.recompute();
    this.applyClass(status.cls);
    this.turnPillText.set(status.text, { animate: true });
  }

  setThinking(text: string): void {
    this.override = { text, cls: "thinking" };
    this.applyClass("thinking");
    this.turnPillText.set(text, { animate: false });
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
    this.clearFlashTimer();
  }

  dispose(): void {
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
    this.pill.className = cls ? `turn-pill glass ${cls}` : "turn-pill glass";
  }
}
