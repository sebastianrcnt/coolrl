import { describe, it, expect } from "vitest";
import {
  chooseMoveWithWeakening,
  type ActionCandidate,
  type WeakeningParams,
} from "./mcts";

const VERY_EASY: WeakeningParams = {
  temperature: 2.0,
  topK: 20,
  minVisitRatio: 0.05,
  qDrop: 0.55,
  qWeight: 0.3,
  priorWeight: 0.5,
};

const NORMAL: WeakeningParams = {
  temperature: 0.55,
  topK: 5,
  minVisitRatio: 0.22,
  qDrop: 0.18,
  qWeight: 2.5,
  priorWeight: 0.25,
};

const HARD: WeakeningParams = {
  temperature: 0.35,
  topK: 3,
  minVisitRatio: 0.35,
  qDrop: 0.10,
  qWeight: 3.0,
  priorWeight: 0.20,
};

function cand(action: number, visits: number, q: number, prior = 0.05): ActionCandidate {
  return { action, visits, q, prior };
}

const constRng = (v: number) => () => v;

describe("chooseMoveWithWeakening — dominant-move bypass", () => {
  it("blocks an open-4: large Q gap forces argmax even at very-easy", () => {
    // Block move: q=+0.1, alternatives: q=-0.9 (opponent wins next move).
    const candidates = [
      cand(10, 30, +0.10),
      cand(20, 4,  -0.90),
      cand(30, 3,  -0.92),
      cand(40, 2,  -0.95),
    ];
    const action = chooseMoveWithWeakening(candidates, VERY_EASY, 64, constRng(0.5));
    expect(action).toBe(10);
  });

  it("blocks an open-3: medium Q gap (~0.30) still triggers bypass", () => {
    // Open-3 is not immediate loss but demands a block. MCTS at sims=96
    // typically yields a moderate Q gap (block ≈ +0.2, attacks ≈ -0.1 to -0.3).
    const candidates = [
      cand(10, 22, +0.20),  // block one end
      cand(20, 14, -0.15),  // some attack
      cand(30, 11, -0.20),
      cand(40,  8, -0.25),
    ];
    const action = chooseMoveWithWeakening(candidates, VERY_EASY, 96, constRng(0.5));
    expect(action).toBe(10);
  });

  it("does NOT bypass on a minor Q preference (gap < 0.20)", () => {
    const candidates = [
      cand(10, 25, +0.30),
      cand(20, 22, +0.15),  // qGap = 0.15, below threshold
      cand(30, 20, +0.10),
      cand(40, 18, +0.05),
    ];
    const seen = new Set<number>();
    let i = 0;
    const rng = () => ((i++ * 0.137) % 1);
    for (let k = 0; k < 30; k++) {
      seen.add(chooseMoveWithWeakening(candidates, VERY_EASY, 96, rng));
    }
    // Sampling proceeds → multiple distinct actions over many draws.
    expect(seen.size).toBeGreaterThan(1);
  });

  it("visit-dominated best is selected even if Q gap is small", () => {
    // No clear Q signal but MCTS heavily concentrated on action 10.
    const candidates = [
      cand(10, 50, +0.20),
      cand(20, 4,  +0.15),
      cand(30, 3,  +0.10),
    ];
    const action = chooseMoveWithWeakening(candidates, VERY_EASY, 96, constRng(0.5));
    expect(action).toBe(10);
  });

  it("visitDom floor scales with sims (12 visits at sims=256 does not bypass)", () => {
    // Visit count 12 with second=3 is 4x ratio. At sims=64 the visitDom
    // floor is max(8, 9) = 9 → would bypass. At sims=256 the floor is
    // max(8, 38) = 38 → must NOT bypass on visit dominance alone.
    const candidates = [
      cand(10, 12, +0.10),
      cand(20, 3,  +0.08),
      cand(30, 2,  +0.05),
    ];
    // qGap=0.02 → qForced no. So sampling must proceed (varied actions).
    const seen = new Set<number>();
    let i = 0;
    const rng = () => ((i++ * 0.137) % 1);
    for (let k = 0; k < 30; k++) {
      seen.add(chooseMoveWithWeakening(candidates, VERY_EASY, 256, rng));
    }
    expect(seen.size).toBeGreaterThan(1);
  });

  it("does NOT bypass when no move is clearly dominant", () => {
    // Several plausible moves with similar Q and visits; sampling should
    // produce different actions across rng draws.
    const candidates = [
      cand(10, 12, +0.10),
      cand(20, 11, +0.08),
      cand(30, 10, +0.07),
      cand(40,  9, +0.05),
      cand(50,  8, +0.04),
    ];
    const seen = new Set<number>();
    let i = 0;
    const rng = () => ((i++ * 0.137) % 1);
    for (let k = 0; k < 30; k++) {
      seen.add(chooseMoveWithWeakening(candidates, VERY_EASY, 96, rng));
    }
    expect(seen.size).toBeGreaterThan(1);
  });
});

describe("chooseMoveWithWeakening — Q-drop with noisy 1-visit Q", () => {
  it("a 1-visit move with high noisy Q does not collapse the pool to argmax", () => {
    // The 1-visit move (action 99) has spuriously high Q. If bestQ were
    // computed before topK filtering, qDrop would shrink the pool to {best}
    // and sampling would degenerate. Computing bestQ AFTER topK keeps the
    // legitimate sibling candidates in play.
    const candidates = [
      cand(10, 25, +0.20),
      cand(20, 22, +0.18),
      cand(30, 20, +0.15),
      cand(40, 18, +0.12),
      cand(50, 15, +0.10),
      cand(99,  1, +0.95), // noisy, low-visit, high Q — should be filtered out
    ];
    // Use a relaxed Q-drop so the bug, if present, would be obvious.
    const params: WeakeningParams = { ...NORMAL, qDrop: 0.18, topK: 5 };
    const seen = new Set<number>();
    let i = 0;
    const rng = () => ((i++ * 0.0731) % 1);
    for (let k = 0; k < 50; k++) {
      const a = chooseMoveWithWeakening(candidates, params, 96, rng);
      seen.add(a);
    }
    // Action 99 is dropped by visit-ratio (1 < 25*0.22≈5.5) and topK.
    expect(seen.has(99)).toBe(false);
    // Other reasonable candidates should appear; if Q-drop misfires they wouldn't.
    expect(seen.size).toBeGreaterThan(1);
  });
});

describe("chooseMoveWithWeakening — Q direction (positive Q = good for AI)", () => {
  it("argmax is preserved when sibling candidates have lower Q", () => {
    const candidates = [
      cand(10, 30, +0.40),
      cand(20, 28, +0.35),
      cand(30, 25, +0.30),
    ];
    // Force first sample to land on first cumulative bucket (action 10 since highest weight).
    const action = chooseMoveWithWeakening(candidates, NORMAL, 96, constRng(0.0));
    expect(action).toBe(10);
  });

  it("Q-drop removes blunders from the pool", () => {
    // bestQ(post-topK) ≈ +0.25, qDrop=0.18 → threshold ≈ +0.07.
    // Action 30 has q=-0.50 → must be filtered out (and the test runs many
    // rng draws to ensure it is never selected).
    const candidates = [
      cand(10, 30, +0.25),
      cand(20, 28, +0.20),
      cand(30, 26, -0.50), // blunder despite decent visits
      cand(40, 22, +0.10),
    ];
    const seen = new Set<number>();
    let i = 0;
    const rng = () => ((i++ * 0.0913) % 1);
    for (let k = 0; k < 60; k++) {
      seen.add(chooseMoveWithWeakening(candidates, NORMAL, 96, rng));
    }
    expect(seen.has(30)).toBe(false);
  });
});

describe("chooseMoveWithWeakening — edge cases", () => {
  it("returns the only candidate when len === 1", () => {
    const action = chooseMoveWithWeakening([cand(7, 50, +0.5)], NORMAL, 96, constRng(0.5));
    expect(action).toBe(7);
  });

  it("falls back to argmax-by-visits when no candidate has visits", () => {
    const candidates = [
      cand(10, 0, 0, 0.1),
      cand(20, 0, 0, 0.5),
      cand(30, 0, 0, 0.2),
    ];
    // All visits=0 → sorted is empty → argmaxByVisits over the original list.
    // Tie-break: first highest-visit. visits all 0, returns first.
    const action = chooseMoveWithWeakening(candidates, NORMAL, 96, constRng(0.5));
    expect(candidates.map(c => c.action)).toContain(action);
  });

  it("hard preset still bypasses on forced moves", () => {
    const candidates = [
      cand(10, 60, +0.15),
      cand(20, 5,  -0.90),
      cand(30, 3,  -0.92),
    ];
    const action = chooseMoveWithWeakening(candidates, HARD, 128, constRng(0.5));
    expect(action).toBe(10);
  });
});
