import { describe, it, expect } from "vitest";
import {
  LOOKAHEAD_TEMPLATES,
  pickLookaheadTemplate,
  formatLookahead,
} from "./lookahead-templates";

describe("pickLookaheadTemplate", () => {
  it("returns a string from the list", () => {
    const result = pickLookaheadTemplate();
    expect(LOOKAHEAD_TEMPLATES).toContain(result);
  });
  it("uses the injected RNG deterministically", () => {
    const first = pickLookaheadTemplate(() => 0);
    expect(first).toBe(LOOKAHEAD_TEMPLATES[0]);
    const last = pickLookaheadTemplate(() => 0.999);
    expect(last).toBe(LOOKAHEAD_TEMPLATES[LOOKAHEAD_TEMPLATES.length - 1]);
  });
});

describe("formatLookahead", () => {
  it("replaces {n} with the sim count", () => {
    const result = formatLookahead("뭔가 {n}번 중...", 42);
    expect(result).toBe("뭔가 42번 중...");
  });
  it("clamps count to minimum 1", () => {
    const result = formatLookahead("{n}번", 0);
    expect(result).toBe("1번");
  });
  it("works with negative by clamping to 1", () => {
    expect(formatLookahead("{n}", -5)).toBe("1");
  });
});
