import { describe, it, expect } from "vitest";
import {
  isInferenceBackend,
  normalizeBackend,
  backendLabel,
} from "./backend";

describe("isInferenceBackend", () => {
  it("accepts wasm", () => expect(isInferenceBackend("wasm")).toBe(true));
  it("accepts webgpu", () => expect(isInferenceBackend("webgpu")).toBe(true));
  it("accepts webnn", () => expect(isInferenceBackend("webnn")).toBe(true));
  it("rejects unknown strings", () => expect(isInferenceBackend("cuda")).toBe(false));
});

describe("normalizeBackend", () => {
  it("passes through valid backends", () => {
    expect(normalizeBackend("webgpu")).toBe("webgpu");
  });
  it("falls back to wasm for unknown", () => {
    expect(normalizeBackend("cuda")).toBe("wasm");
  });
  it("falls back to wasm for null", () => {
    expect(normalizeBackend(null)).toBe("wasm");
  });
  it("falls back to wasm for undefined", () => {
    expect(normalizeBackend(undefined)).toBe("wasm");
  });
});

describe("backendLabel", () => {
  it("maps wasm to WASM", () => expect(backendLabel("wasm")).toBe("WASM"));
  it("maps webgpu to WebGPU", () => expect(backendLabel("webgpu")).toBe("WebGPU"));
  it("maps webnn to WebML", () => expect(backendLabel("webnn")).toBe("WebML"));
  it("falls back to WASM for unknown", () => expect(backendLabel("cuda")).toBe("WASM"));
});
