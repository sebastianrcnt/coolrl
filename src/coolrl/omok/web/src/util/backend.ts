export type InferenceBackend = "wasm" | "webgpu" | "webnn";
export type BackendChoice = "auto" | InferenceBackend;

export const BACKENDS: readonly InferenceBackend[] = ["wasm", "webgpu", "webnn"];
export const BACKEND_CHOICES: readonly BackendChoice[] = [
  "auto",
  "wasm",
  "webgpu",
  "webnn",
];

const BACKEND_LABELS: Record<InferenceBackend, string> = {
  wasm: "WASM",
  webgpu: "WebGPU",
  webnn: "WebML",
};

const BACKEND_CHOICE_LABELS: Record<BackendChoice, string> = {
  auto: "자동",
  ...BACKEND_LABELS,
};

export function isInferenceBackend(value: string): value is InferenceBackend {
  return (BACKENDS as readonly string[]).includes(value);
}

export function isBackendChoice(value: string): value is BackendChoice {
  return (BACKEND_CHOICES as readonly string[]).includes(value);
}

export function normalizeBackend(value: string | undefined | null): InferenceBackend {
  return value && isInferenceBackend(value) ? value : "wasm";
}

export function normalizeBackendChoice(
  value: string | undefined | null
): BackendChoice {
  return value && isBackendChoice(value) ? value : "auto";
}

export function backendLabel(value: InferenceBackend | string | undefined | null): string {
  return BACKEND_LABELS[normalizeBackend(value ?? null)];
}

export function backendChoiceLabel(
  value: BackendChoice | string | undefined | null
): string {
  return BACKEND_CHOICE_LABELS[normalizeBackendChoice(value ?? null)];
}

// "auto" tries WebGPU first only on desktop and falls back to WASM; mobile
// devices stay on WASM because WebGPU/WebNN support is still uneven there.
// An explicit choice is honored strictly so the user can diagnose backend issues.
export function resolveBackendAttempts(
  choice: BackendChoice,
  webgpuSupported: boolean,
  isMobile = false
): InferenceBackend[] {
  if (choice === "auto") {
    if (isMobile) return ["wasm"];
    return webgpuSupported ? ["webgpu", "wasm"] : ["wasm"];
  }
  return [choice];
}
