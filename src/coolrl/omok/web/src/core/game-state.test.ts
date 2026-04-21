import { describe, it, expect } from "vitest";
import { GameState } from "./game-state";

function playSequence(game: GameState, actions: number[]): void {
  for (const a of actions) game.applyAction(a);
}

describe("GameState basics", () => {
  it("starts with empty board and black to play", () => {
    const g = new GameState(15);
    expect(g.toPlay).toBe(1);
    expect(g.moveCount).toBe(0);
    expect(g.terminal).toBe(false);
    expect(g.winner).toBe(0);
  });

  it("alternates players after each move", () => {
    const g = new GameState(15);
    g.applyAction(0);
    expect(g.toPlay).toBe(-1);
    g.applyAction(1);
    expect(g.toPlay).toBe(1);
  });

  it("legalIndices shrinks as moves are played", () => {
    const g = new GameState(15);
    expect(g.legalIndices()).toHaveLength(225);
    g.applyAction(0);
    expect(g.legalIndices()).toHaveLength(224);
  });

  it("legalIndices is empty when terminal", () => {
    const g = new GameState(15);
    // Force a horizontal 5-in-a-row for black (columns 0-4, row 0)
    playSequence(g, [
      0, 15,   // row0 col0 (b), row1 col0 (w)
      1, 16,
      2, 17,
      3, 18,
      4,       // black wins
    ]);
    expect(g.terminal).toBe(true);
    expect(g.legalIndices()).toHaveLength(0);
  });

  it("rejects moves after the game is terminal", () => {
    const g = new GameState(15);
    playSequence(g, [0, 15, 1, 16, 2, 17, 3, 18, 4]);
    expect(() => g.applyAction(5)).toThrow("cannot play on a terminal position");
  });

  it("rejects occupied cells", () => {
    const g = new GameState(15);
    g.applyAction(0);
    expect(() => g.applyAction(0)).toThrow("illegal move at (0, 0)");
  });

  it("rejects out-of-range or non-integer actions", () => {
    const g = new GameState(15);
    expect(() => g.applyAction(-1)).toThrow("action out of range: -1");
    expect(() => g.applyAction(g.actionSize)).toThrow(`action out of range: ${g.actionSize}`);
    expect(() => g.applyAction(1.5)).toThrow("action out of range: 1.5");
  });
});

describe("GameState win detection", () => {
  it("detects horizontal 5-in-a-row for black", () => {
    const g = new GameState(15);
    // black: 0,1,2,3,4  white: 15,16,17,18 (different row)
    playSequence(g, [0, 15, 1, 16, 2, 17, 3, 18, 4]);
    expect(g.terminal).toBe(true);
    expect(g.winner).toBe(1);
  });

  it("detects vertical 5-in-a-row for white", () => {
    const g = new GameState(15);
    // black avoids completing row 0; white: 15,30,45,60,75
    playSequence(g, [0, 15, 1, 30, 2, 45, 3, 60, 5, 75]);
    expect(g.terminal).toBe(true);
    expect(g.winner).toBe(-1);
  });

  it("detects diagonal 5-in-a-row", () => {
    const g = new GameState(15);
    // black: diagonal (0,0),(1,1),(2,2),(3,3),(4,4) = actions 0,16,32,48,64
    // white: column 14: 14,29,44,59
    playSequence(g, [0, 14, 16, 29, 32, 44, 48, 59, 64]);
    expect(g.terminal).toBe(true);
    expect(g.winner).toBe(1);
  });

  it("does not trigger on 4-in-a-row", () => {
    const g = new GameState(15);
    playSequence(g, [0, 15, 1, 16, 2, 17, 3]);
    expect(g.terminal).toBe(false);
  });
});

describe("GameState.clone", () => {
  it("produces an independent copy", () => {
    const g = new GameState(15);
    g.applyAction(0);
    const copy = g.clone();
    copy.applyAction(1);
    expect(g.moveCount).toBe(1);
    expect(copy.moveCount).toBe(2);
    expect(g.board[1]).toBe(0);
  });
});

describe("outcomeForPlayer", () => {
  it("returns 1 for the winner", () => {
    const g = new GameState(15);
    playSequence(g, [0, 15, 1, 16, 2, 17, 3, 18, 4]);
    expect(g.outcomeForPlayer(1)).toBe(1.0);
    expect(g.outcomeForPlayer(-1)).toBe(-1.0);
  });
  it("returns 0 for a draw", () => {
    const g = new GameState(3); // 3×3, all moves exhaust without 5-in-a-row
    for (let i = 0; i < 9; i++) g.applyAction(i);
    expect(g.terminal).toBe(true);
    expect(g.winner).toBe(0);
    expect(g.outcomeForPlayer(1)).toBe(0);
  });
});
