export type InferenceBackend = "wasm" | "webgpu" | "webnn";

export const BACKENDS: readonly InferenceBackend[] = ["wasm", "webgpu", "webnn"];

const BACKEND_LABELS: Record<InferenceBackend, string> = {
  wasm: "WASM",
  webgpu: "WebGPU",
  webnn: "WebML",
};

export function isInferenceBackend(value: string): value is InferenceBackend {
  return (BACKENDS as readonly string[]).includes(value);
}

export function normalizeBackend(value: string | undefined | null): InferenceBackend {
  return value && isInferenceBackend(value) ? value : "wasm";
}

export function backendLabel(value: InferenceBackend | string | undefined | null): string {
  return BACKEND_LABELS[normalizeBackend(value ?? null)];
}
