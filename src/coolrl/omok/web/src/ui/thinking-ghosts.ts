import type { GameState, Player } from "../core/game-state";

export const THINKING_GHOST_INTERVAL_MS = 110;
export const THINKING_GHOST_LIFE_MS = 360;
export const THINKING_GHOST_MAX = 5;

export interface ThinkingGhostSlot {
  action: number;
  player: Player;
  start: number;
  life: number;
}

export interface ThinkingGhostsDeps {
  getGame(): GameState;
  isBusy(): boolean;
  onTick(): void;
}

export class ThinkingGhosts {
  private player: Player | 0 = 0;
  private candidates: number[] = [];
  private slotsInternal: ThinkingGhostSlot[] = [];
  private lastSpawn = 0;
  private rafHandle = 0;
  private readonly deps: ThinkingGhostsDeps;

  constructor(deps: ThinkingGhostsDeps) {
    this.deps = deps;
  }

  get slots(): readonly ThinkingGhostSlot[] {
    return this.slotsInternal;
  }

  get candidateCount(): number {
    return this.candidates.length;
  }

  get active(): boolean {
    return this.player !== 0;
  }

  get isTicking(): boolean {
    return this.rafHandle !== 0;
  }

  start(player: Player): void {
    this.player = player;
    this.candidates = [];
    this.slotsInternal = [];
    this.lastSpawn = 0;
    if (!this.rafHandle) {
      this.rafHandle = requestAnimationFrame(this.tick);
    }
  }

  stop(shouldTick = true): void {
    this.player = 0;
    this.candidates = [];
    this.slotsInternal = [];
    this.lastSpawn = 0;
    if (this.rafHandle) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = 0;
    }
    if (shouldTick) this.deps.onTick();
  }

  updateCandidates(actions: ReadonlyArray<number | { action: number }>): void {
    const game = this.deps.getGame();
    this.candidates = actions
      .map((item) => (typeof item === "number" ? item : item.action))
      .filter((action) => Number.isInteger(action) && game.board[action] === 0)
      .slice(0, 10);
  }

  private pickAction(): number | null {
    const game = this.deps.getGame();
    const active = new Set(this.slotsInternal.map((slot) => slot.action));
    let pool = this.candidates.filter(
      (action) => game.board[action] === 0 && !active.has(action)
    );
    if (pool.length === 0) {
      pool = game.legalIndices().filter((action) => !active.has(action));
    }
    if (pool.length === 0) return null;
    const upper = Math.min(pool.length, this.candidates.length ? 10 : pool.length);
    return pool[Math.floor(Math.random() * upper)] ?? null;
  }

  private spawn(now: number): void {
    if (this.player === 0) return;
    const game = this.deps.getGame();
    if (game.terminal) return;
    const action = this.pickAction();
    if (action === null) return;
    this.slotsInternal.push({
      action,
      player: this.player,
      start: now,
      life: THINKING_GHOST_LIFE_MS,
    });
    if (this.slotsInternal.length > THINKING_GHOST_MAX) {
      this.slotsInternal.splice(0, this.slotsInternal.length - THINKING_GHOST_MAX);
    }
  }

  private tick = (now: number): void => {
    this.rafHandle = 0;
    this.slotsInternal = this.slotsInternal.filter((slot) => now - slot.start < slot.life);

    const game = this.deps.getGame();
    if (this.player === 0 || game.terminal || !this.deps.isBusy()) {
      if (this.slotsInternal.length > 0) {
        this.deps.onTick();
        this.rafHandle = requestAnimationFrame(this.tick);
      }
      return;
    }

    if (
      now - this.lastSpawn >= THINKING_GHOST_INTERVAL_MS ||
      this.slotsInternal.length === 0
    ) {
      this.spawn(now);
      this.lastSpawn = now;
    }

    this.deps.onTick();
    this.rafHandle = requestAnimationFrame(this.tick);
  };
}
