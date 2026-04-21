import { describe, expect, it, vi } from "vitest";

vi.mock("./worker?worker", () => {
  class FakeWorker {
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: ErrorEvent) => void) | null = null;

    postMessage(message: { id: number; type: string; backend?: string }, transfer: Transferable[] = []): void {
      structuredClone(message, { transfer });
      queueMicrotask(() => {
        if (message.type === "init") {
          this.onmessage?.({
            data: { id: message.id, ok: true, backend: message.backend ?? "wasm" },
          } as MessageEvent);
        }
      });
    }

    terminate(): void {
      // no-op
    }
  }

  return { default: FakeWorker };
});

describe("WorkerEvaluator.fromArrayBuffer", () => {
  it("does not detach the caller-owned model buffer", async () => {
    const { WorkerEvaluator } = await import("./worker-evaluator");
    const buf = new ArrayBuffer(16);
    const evaluator = await WorkerEvaluator.fromArrayBuffer(buf, 15);

    expect(buf.byteLength).toBe(16);
    expect(() => buf.slice(0)).not.toThrow();

    evaluator.terminate();
  });
});
