import { describe, it, expect } from "vitest";
import {
  emptyModelSource,
  setFromFile,
  setFromDefault,
  hasModel,
} from "./model-source";

describe("emptyModelSource", () => {
  it("has no model", () => {
    const src = emptyModelSource();
    expect(hasModel(src, false)).toBe(false);
    expect(src.origin).toBeNull();
    expect(src.name).toBeNull();
  });
});

describe("setFromFile", () => {
  const buf = new ArrayBuffer(1024);
  const file = new File([buf], "custom.onnx");

  it("sets origin to file", () => {
    const src = setFromFile(file, buf, false);
    expect(src.origin).toBe("file");
    expect(src.name).toBe("custom.onnx");
  });

  it("keeps a reusable buffer when not low-memory", () => {
    const src = setFromFile(file, buf, false);
    expect(src.buffer).not.toBeNull();
    expect(src.file).toBeNull();
  });

  it("stores the file reference in low-memory mode", () => {
    const src = setFromFile(file, buf, true);
    expect(src.buffer).toBeNull();
    expect(src.file).toBe(file);
  });
});

describe("setFromDefault", () => {
  const buf = new ArrayBuffer(2048);

  it("sets origin to default", () => {
    const src = setFromDefault(buf, false);
    expect(src.origin).toBe("default");
    expect(src.name).toContain("best.onnx");
  });

  it("reports correct byte count", () => {
    const src = setFromDefault(buf, false);
    expect(src.bytes).toBe(2048);
  });
});

describe("hasModel", () => {
  it("returns true when evaluator is active", () => {
    expect(hasModel(emptyModelSource(), true)).toBe(true);
  });
  it("returns false for empty source with no active evaluator", () => {
    expect(hasModel(emptyModelSource(), false)).toBe(false);
  });
  it("returns true when origin is default, even without active evaluator", () => {
    const src = setFromDefault(new ArrayBuffer(8), false);
    src.buffer = null; // simulate low-memory cleared buffer
    expect(hasModel(src, false)).toBe(true);
  });
});
