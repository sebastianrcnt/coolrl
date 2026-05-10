// Difficulty presets for the omok AI weakening system.
//
// Each preset bundles two things:
//   - sims:      MCTS simulation count for this difficulty
//   - weakening: filtered-softmax parameters (null = pure argmax / strongest)
//
// Tuning notes
// ------------
// sims: keep in [96, 256]. Below 96 the visit-ratio / top-K filters lose
//   resolution; above 256 we hit mobile OOM.
// temperature: softmax flatness inside the candidate pool. 0 → argmax.
// topK: candidate pool cap. Larger = more variety.
// minVisitRatio: drop moves with N < ratio * maxN. Cuts the junk tail.
// qDrop: drop moves with Q < bestQ - qDrop. Bigger = lenient (allows blunders).
// qWeight: how much Q biases the softmax. Lower = AI cares less about quality.
// priorWeight: weight of policy-network prior in softmax. Sets human-like flavor.
//
// Forced-move protection (4-in-a-row blocks etc.) is handled in mcts.ts via
// the dominant-move bypass and applies at every difficulty level — so making
// presets weaker here cannot make the AI ignore obvious threats.

import type { WeakeningParams } from "../core/mcts";

export interface DifficultyPreset {
  readonly label: string;
  readonly sims: number;
  readonly weakening: WeakeningParams | null; // null = argmax (강함)
}

export const DIFFICULTY_PRESETS = {
  strong: {
    label: "강함",
    sims: 256,
    weakening: null,
  },
  hard: {
    label: "어려움",
    sims: 128,
    weakening: {
      temperature: 0.35,
      topK: 3,
      minVisitRatio: 0.35,
      qDrop: 0.10,
      qWeight: 3.0,
      priorWeight: 0.20,
    },
  },
  normal: {
    label: "보통",
    sims: 128,
    weakening: {
      temperature: 0.55,
      topK: 5,
      minVisitRatio: 0.22,
      qDrop: 0.18,
      qWeight: 2.5,
      priorWeight: 0.25,
    },
  },
  easy: {
    label: "쉬움",
    sims: 96,
    weakening: {
      temperature: 1.20,
      topK: 12,
      minVisitRatio: 0.10,
      qDrop: 0.35,
      qWeight: 1.0,
      priorWeight: 0.40,
    },
  },
  veryEasy: {
    label: "매우 쉬움",
    sims: 96,
    weakening: {
      temperature: 2.50,
      topK: 25,
      minVisitRatio: 0.04,
      qDrop: 0.65,
      qWeight: 0.2,
      priorWeight: 0.55,
    },
  },
} as const satisfies Record<string, DifficultyPreset>;

export type Difficulty = keyof typeof DIFFICULTY_PRESETS;

export const DEFAULT_DIFFICULTY: Difficulty = "normal";
export const MOBILE_DEFAULT_DIFFICULTY: Difficulty = "easy";

export function resolveDifficulty(value: string): Difficulty {
  return value in DIFFICULTY_PRESETS ? (value as Difficulty) : DEFAULT_DIFFICULTY;
}

export function getDifficultyPreset(value: string): DifficultyPreset {
  return DIFFICULTY_PRESETS[resolveDifficulty(value)];
}
