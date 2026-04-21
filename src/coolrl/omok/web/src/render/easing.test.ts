import { describe, it, expect } from "vitest";
import { clamp01, easeOutCubic, easeInCubic, easeOutBack } from "./easing";

describe("clamp01", () => {
  it("clamps below 0 to 0", () => expect(clamp01(-5)).toBe(0));
  it("clamps above 1 to 1", () => expect(clamp01(2)).toBe(1));
  it("passes through midpoint", () => expect(clamp01(0.5)).toBe(0.5));
});

describe("easeOutCubic", () => {
  it("is 0 at t=0", () => expect(easeOutCubic(0)).toBe(0));
  it("is 1 at t=1", () => expect(easeOutCubic(1)).toBe(1));
  it("is concave (value > t for mid-range)", () => expect(easeOutCubic(0.5)).toBeGreaterThan(0.5));
});

describe("easeInCubic", () => {
  it("is 0 at t=0", () => expect(easeInCubic(0)).toBe(0));
  it("is 1 at t=1", () => expect(easeInCubic(1)).toBe(1));
  it("is convex (value < t for mid-range)", () => expect(easeInCubic(0.5)).toBeLessThan(0.5));
});

describe("easeOutBack", () => {
  it("is 1 at t=1", () => expect(easeOutBack(1)).toBeCloseTo(1));
  it("overshoots past 1 near the end", () => {
    const peak = easeOutBack(0.7);
    expect(peak).toBeGreaterThan(1);
  });
});
