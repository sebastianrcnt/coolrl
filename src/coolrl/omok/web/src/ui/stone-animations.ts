export const STONE_ANIM_MS = 280;

interface StoneAnim {
  start: number;
}

export class StoneAnimations {
  private readonly anims = new Map<number, StoneAnim>();
  private rafHandle = 0;
  private readonly onTick: () => void;

  constructor(onTick: () => void) {
    this.onTick = onTick;
  }

  start(action: number): void {
    this.anims.set(action, { start: performance.now() });
    if (!this.rafHandle) this.tick();
  }

  has(action: number): boolean {
    return this.anims.has(action);
  }

  progress(action: number, now = performance.now()): number | null {
    const anim = this.anims.get(action);
    if (!anim) return null;
    return Math.min(1, (now - anim.start) / STONE_ANIM_MS);
  }

  clear(): void {
    this.anims.clear();
    if (this.rafHandle) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = 0;
    }
  }

  get size(): number {
    return this.anims.size;
  }

  get isTicking(): boolean {
    return this.rafHandle !== 0;
  }

  private tick = (): void => {
    this.rafHandle = 0;
    const now = performance.now();
    for (const [action, anim] of this.anims) {
      if (now - anim.start >= STONE_ANIM_MS) this.anims.delete(action);
    }
    this.onTick();
    if (this.anims.size > 0) {
      this.rafHandle = requestAnimationFrame(this.tick);
    }
  };
}
