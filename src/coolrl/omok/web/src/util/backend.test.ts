import { describe, it, expect } from "vitest";
import {
  backendChoiceLabel,
  backendLabel,
  isBackendChoice,
  isInferenceBackend,
  normalizeBackend,
  normalizeBackendChoice,
  resolveBackendAttempts,
} from "./backend";

describe("isInferenceBackend", () => {
  it("accepts wasm", () => expect(isInferenceBackend("wasm")).toBe(true));
  it("accepts webgpu", () => expect(isInferenceBackend("webgpu")).toBe(true));
  it("accepts webnn", () => expect(isInferenceBackend("webnn")).toBe(true));
  it("rejects auto", () => expect(isInferenceBackend("auto")).toBe(false));
  it("rejects unknown strings", () => expect(isInferenceBackend("cuda")).toBe(false));
});

describe("isBackendChoice", () => {
  it("accepts auto", () => expect(isBackendChoice("auto")).toBe(true));
  it("accepts wasm", () => expect(isBackendChoice("wasm")).toBe(true));
  it("accepts webgpu", () => expect(isBackendChoice("webgpu")).toBe(true));
  it("accepts webnn", () => expect(isBackendChoice("webnn")).toBe(true));
  it("rejects unknown strings", () => expect(isBackendChoice("cuda")).toBe(false));
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
  it("falls back to wasm for auto", () => {
    expect(normalizeBackend("auto")).toBe("wasm");
  });
});

describe("normalizeBackendChoice", () => {
  it("passes through auto", () => {
    expect(normalizeBackendChoice("auto")).toBe("auto");
  });
  it("passes through valid backends", () => {
    expect(normalizeBackendChoice("webgpu")).toBe("webgpu");
  });
  it("defaults to auto for unknown", () => {
    expect(normalizeBackendChoice("cuda")).toBe("auto");
  });
  it("defaults to auto for null", () => {
    expect(normalizeBackendChoice(null)).toBe("auto");
  });
  it("defaults to auto for undefined", () => {
    expect(normalizeBackendChoice(undefined)).toBe("auto");
  });
});

describe("backendLabel", () => {
  it("maps wasm to WASM", () => expect(backendLabel("wasm")).toBe("WASM"));
  it("maps webgpu to WebGPU", () => expect(backendLabel("webgpu")).toBe("WebGPU"));
  it("maps webnn to WebNN", () => expect(backendLabel("webnn")).toBe("WebNN"));
  it("falls back to WASM for unknown", () => expect(backendLabel("cuda")).toBe("WASM"));
});

describe("backendChoiceLabel", () => {
  it("maps auto to 자동", () => expect(backendChoiceLabel("auto")).toBe("자동"));
  it("maps wasm to WASM", () => expect(backendChoiceLabel("wasm")).toBe("WASM"));
  it("maps webgpu to WebGPU", () => expect(backendChoiceLabel("webgpu")).toBe("WebGPU"));
  it("defaults to 자동 for unknown", () => expect(backendChoiceLabel("cuda")).toBe("자동"));
});

describe("resolveBackendAttempts", () => {
  it("auto with WebGPU support tries webgpu then wasm", () => {
    expect(resolveBackendAttempts("auto", true)).toEqual(["webgpu", "wasm"]);
  });
  it("auto on mobile uses wasm even with WebGPU support", () => {
    expect(resolveBackendAttempts("auto", true, true)).toEqual(["wasm"]);
  });
  it("auto without WebGPU support skips straight to wasm", () => {
    expect(resolveBackendAttempts("auto", false)).toEqual(["wasm"]);
  });
  it("explicit wasm stays on wasm with no fallback", () => {
    expect(resolveBackendAttempts("wasm", true)).toEqual(["wasm"]);
  });
  it("explicit webgpu stays on webgpu even when unsupported (surfaces real error)", () => {
    expect(resolveBackendAttempts("webgpu", false)).toEqual(["webgpu"]);
  });
  it("explicit webnn stays on webnn with no fallback", () => {
    expect(resolveBackendAttempts("webnn", true)).toEqual(["webnn"]);
  });
});
