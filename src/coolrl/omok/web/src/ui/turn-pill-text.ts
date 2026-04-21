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
    const reduceMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (!animate || reduceMotion) {
      this.cancelTimer();
      this.pill.style.width = "";
      this.textNode.textContent = text;
      return;
    }

    this.cancelTimer();
    this.pill.style.width = "";
    this.textNode.textContent = text;

    if (typeof this.textNode.animate === "function") {
      this.textNode.animate(
        [
          { opacity: 0.72, transform: "translateY(1px)" },
          { opacity: 1, transform: "translateY(0)" },
        ],
        { duration: 140, easing: "ease-out" }
      );
    }

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
