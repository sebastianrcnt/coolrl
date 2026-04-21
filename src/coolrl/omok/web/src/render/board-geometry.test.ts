import { describe, it, expect } from "vitest";
import {
  starPoints,
  computeCellMetrics,
  cellX,
  cellY,
  pixelToCell,
  actionToRowCol,
} from "./board-geometry";

describe("starPoints", () => {
  it("returns 5 points for 15×15", () => {
    expect(starPoints(15)).toHaveLength(5);
  });
  it("includes the center for 15×15", () => {
    const points = starPoints(15);
    expect(points).toContainEqual([7, 7]);
  });
  it("returns only center for small board", () => {
    const points = starPoints(5);
    expect(points).toHaveLength(1);
    expect(points[0]).toEqual([2, 2]);
  });
});

describe("computeCellMetrics + cellX/cellY", () => {
  const metrics = computeCellMetrics(300, 300, 15, 0.04);

  it("step is positive and reasonable", () => {
    expect(metrics.step).toBeGreaterThan(0);
  });
  it("offsetX/Y are 0 for square canvas", () => {
    expect(metrics.offsetX).toBe(0);
    expect(metrics.offsetY).toBe(0);
  });
  it("cellX(0) equals margin", () => {
    expect(cellX(metrics, 0)).toBeCloseTo(metrics.margin);
  });
  it("cellX(14) equals width - margin", () => {
    expect(cellX(metrics, 14)).toBeCloseTo(metrics.width - metrics.margin);
  });
  it("cellY(7) is at the center", () => {
    const cy = cellY(metrics, 7);
    expect(cy).toBeCloseTo(metrics.height / 2, 0);
  });
});

describe("pixelToCell", () => {
  const metrics = computeCellMetrics(300, 300, 15, 0.04);

  it("round-trips pixel at intersection back to that cell", () => {
    const col = 5, row = 3;
    const x = cellX(metrics, col);
    const y = cellY(metrics, row);
    const cell = pixelToCell(metrics, 15, x, y);
    expect(cell).toEqual({ row, col });
  });

  it("returns null for pixel far outside the board", () => {
    expect(pixelToCell(metrics, 15, -50, -50)).toBeNull();
  });

  it("returns null for pixel outside the board margin", () => {
    // pixelToCell rounds to the nearest intersection, so only pixels that
    // round to an out-of-range column/row return null
    const x = cellX(metrics, 0) - metrics.step * 0.6; // rounds to col=-1
    const y = cellY(metrics, 0);
    expect(pixelToCell(metrics, 15, x, y)).toBeNull();
  });
});

describe("actionToRowCol", () => {
  it("converts action 0 to (0,0)", () => {
    expect(actionToRowCol(0, 15)).toEqual({ row: 0, col: 0 });
  });
  it("converts action 14 to (0,14)", () => {
    expect(actionToRowCol(14, 15)).toEqual({ row: 0, col: 14 });
  });
  it("converts action 15 to (1,0)", () => {
    expect(actionToRowCol(15, 15)).toEqual({ row: 1, col: 0 });
  });
});
