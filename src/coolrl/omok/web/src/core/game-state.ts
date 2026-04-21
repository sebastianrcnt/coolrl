export type Player = 1 | -1;
export type Cell = 1 | -1 | 0;

const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [0, 1],
  [1, 1],
  [1, -1],
];

export class GameState {
  readonly boardSize: number;
  readonly actionSize: number;
  readonly board: Int8Array;
  toPlay: Player;
  moveCount: number;
  lastAction: number | null;
  winner: Player | 0;
  terminal: boolean;

  constructor(boardSize = 9) {
    this.boardSize = boardSize;
    this.actionSize = boardSize * boardSize;
    this.board = new Int8Array(this.actionSize);
    this.toPlay = 1;
    this.moveCount = 0;
    this.lastAction = null;
    this.winner = 0;
    this.terminal = false;
  }

  clone(): GameState {
    const copy = new GameState(this.boardSize);
    copy.board.set(this.board);
    copy.toPlay = this.toPlay;
    copy.moveCount = this.moveCount;
    copy.lastAction = this.lastAction;
    copy.winner = this.winner;
    copy.terminal = this.terminal;
    return copy;
  }

  legalIndices(): number[] {
    if (this.terminal) return [];
    const out: number[] = [];
    for (let i = 0; i < this.actionSize; i++) {
      if (this.board[i] === 0) out.push(i);
    }
    return out;
  }

  applyAction(action: number): void {
    const player = this.toPlay;
    this.board[action] = player;
    this.lastAction = action;
    this.moveCount++;
    const row = Math.floor(action / this.boardSize);
    const col = action % this.boardSize;
    if (this.isWinAt(row, col, player)) {
      this.winner = player;
      this.terminal = true;
    } else if (this.moveCount >= this.actionSize) {
      this.terminal = true;
    }
    this.toPlay = -this.toPlay as Player;
  }

  outcomeForPlayer(player: Player): number {
    if (!this.terminal) return 0;
    if (this.winner === 0) return 0;
    return this.winner === player ? 1.0 : -1.0;
  }

  private isWinAt(row: number, col: number, player: Player): boolean {
    for (const [dr, dc] of DIRECTIONS) {
      const count =
        1 +
        this.countDirection(row, col, dr, dc, player) +
        this.countDirection(row, col, -dr, -dc, player);
      if (count >= 5) return true;
    }
    return false;
  }

  private countDirection(
    row: number,
    col: number,
    dr: number,
    dc: number,
    player: Player
  ): number {
    let count = 0;
    let r = row + dr;
    let c = col + dc;
    while (
      r >= 0 &&
      r < this.boardSize &&
      c >= 0 &&
      c < this.boardSize &&
      this.board[r * this.boardSize + c] === player
    ) {
      count++;
      r += dr;
      c += dc;
    }
    return count;
  }
}
