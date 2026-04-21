interface TurnPillTextOptions {
  pill: HTMLElement;
  textNode: HTMLElement;
  resetTimeoutMs?: number;
}

export class TurnPillText {
  private readonly pill: HTMLElement;
  private readonly textNode: HTMLElement;
  private readonly resetTimeoutMs: number;
  private resetTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(options: TurnPillTextOptions) {
    this.pill = options.pill;
    this.textNode = options.textNode;
    this.resetTimeoutMs = options.resetTimeoutMs ?? 320;
  }

  set(text: string, options: { animate?: boolean } = {}): void {
    if (this.textNode.textContent === text) return;
    const animate = options.animate ?? true;
    if (!animate) {
      this.cancelTimer();
      this.pill.style.width = "";
      this.textNode.textContent = text;
      return;
    }

    const oldWidth = this.pill.getBoundingClientRect().width;
    const reduceMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion || oldWidth <= 0) {
      this.textNode.textContent = text;
      this.pill.style.width = "";
      return;
    }

    this.cancelTimer();
    const fromWidth = Math.ceil(oldWidth);
    const previousTransition = this.pill.style.transition;
    this.pill.style.transition = "none";
    this.pill.style.width = `${fromWidth}px`;
    void this.pill.offsetWidth;

    this.textNode.textContent = text;
    this.pill.style.width = "auto";
    const toWidth = Math.ceil(this.pill.getBoundingClientRect().width);
    this.pill.style.width = `${fromWidth}px`;
    void this.pill.offsetWidth;
    this.pill.style.transition = previousTransition;

    requestAnimationFrame(() => {
      this.pill.style.width = `${toWidth}px`;
    });

    this.resetTimer = setTimeout(() => {
      if (this.textNode.textContent === text) this.pill.style.width = "";
    }, this.resetTimeoutMs);
  }

  private cancelTimer(): void {
    if (this.resetTimer !== null) {
      clearTimeout(this.resetTimer);
      this.resetTimer = null;
    }
  }
}
