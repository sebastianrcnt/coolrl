export type ModelOrigin = "default" | "file" | null;

export interface ModelSourceState {
  buffer: ArrayBuffer | null;
  file: File | null;
  origin: ModelOrigin;
  bytes: number;
  name: string | null;
  title: string | null;
}

export function emptyModelSource(): ModelSourceState {
  return { buffer: null, file: null, origin: null, bytes: 0, name: null, title: null };
}

export function setFromFile(file: File, buf: ArrayBuffer, lowMemory: boolean): ModelSourceState {
  return {
    buffer: lowMemory ? null : buf.slice(0),
    file: lowMemory ? file : null,
    origin: "file",
    bytes: file.size || buf.byteLength,
    name: file.name,
    title: file.name,
  };
}

export function setFromDefault(buf: ArrayBuffer, lowMemory: boolean): ModelSourceState {
  return {
    buffer: lowMemory ? null : buf.slice(0),
    file: null,
    origin: "default",
    bytes: buf.byteLength,
    name: "best.onnx (기본)",
    title: "best.onnx (기본 모델)",
  };
}

export function hasModel(state: ModelSourceState, evaluatorActive: boolean): boolean {
  return evaluatorActive || state.buffer !== null || state.file !== null || state.origin === "default";
}

export async function fetchBufferFor(
  state: ModelSourceState,
  defaultUrl: string
): Promise<ArrayBuffer> {
  if (state.buffer) return state.buffer.slice(0);
  if (state.file) return state.file.arrayBuffer();
  if (state.origin === "default") {
    const response = await fetch(defaultUrl, { cache: "force-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status} on ${defaultUrl}`);
    return response.arrayBuffer();
  }
  throw new Error("model buffer unavailable");
}
