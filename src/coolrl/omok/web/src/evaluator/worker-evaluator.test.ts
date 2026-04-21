import { describe, expect, it, vi } from "vitest";

// Shared fake worker that handles the full message protocol.
// The test file stores posted messages on an exported array so each test
// can assert what the evaluator sent over the wire.
const sent: Array<{ type: string; id: number }> = [];
vi.mock("./worker?worker", () => {
  class FakeWorker {
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: ErrorEvent) => void) | null = null;
    terminated = false;

    postMessage(
      message: { id: number; type: string; backend?: string },
      transfer: Transferable[] = []
    ): void {
      // structuredClone with transfer validates the transfer list the same
      // way a real worker boundary would (rejects detachment of buffers in
      // use, etc.) without actually crossing a thread.
      structuredClone(message, { transfer });
      sent.push({ type: message.type, id: message.id });
      queueMicrotask(() => {
        if (this.terminated) return;
        if (message.type === "init") {
          this.onmessage?.({
            data: { id: message.id, ok: true, backend: message.backend ?? "wasm" },
          } as MessageEvent);
        } else if (message.type === "evaluate") {
          this.onmessage?.({
            data: {
              id: message.id,
              ok: true,
              policy: new Float32Array([1]).buffer,
              values: new Float32Array([0]).buffer,
              batch: 1,
              actionSize: 1,
            },
          } as MessageEvent);
        } else if (message.type === "dispose") {
          this.onmessage?.({
            data: { id: message.id, ok: true, disposed: true },
          } as MessageEvent);
        }
      });
    }

    terminate(): void {
      this.terminated = true;
    }
  }

  return { default: FakeWorker };
});

describe("WorkerEvaluator.fromArrayBuffer", () => {
  it("does not detach the caller-owned model buffer", async () => {
    sent.length = 0;
    const { WorkerEvaluator } = await import("./worker-evaluator");
    const buf = new ArrayBuffer(16);
    const evaluator = await WorkerEvaluator.fromArrayBuffer(buf, 15);

    expect(buf.byteLength).toBe(16);
    expect(() => buf.slice(0)).not.toThrow();

    evaluator.terminate();
  });
});

describe("WorkerEvaluator.dispose", () => {
  it("sends a dispose message and then terminates the worker", async () => {
    sent.length = 0;
    const { WorkerEvaluator } = await import("./worker-evaluator");
    const buf = new ArrayBuffer(16);
    const evaluator = await WorkerEvaluator.fromArrayBuffer(buf, 15);

    sent.length = 0;
    await evaluator.dispose();

    const disposeMessages = sent.filter((m) => m.type === "dispose");
    expect(disposeMessages).toHaveLength(1);
    // A subsequent evaluate must reject because terminate() fired after
    // release completed.
    await expect(evaluator.evaluate([])).rejects.toThrow();
  });

  it("is a no-op when the worker is already gone", async () => {
    sent.length = 0;
    const { WorkerEvaluator } = await import("./worker-evaluator");
    const buf = new ArrayBuffer(16);
    const evaluator = await WorkerEvaluator.fromArrayBuffer(buf, 15);

    evaluator.terminate();
    sent.length = 0;
    await expect(evaluator.dispose()).resolves.toBeUndefined();
    expect(sent.filter((m) => m.type === "dispose")).toHaveLength(0);
  });
});

describe("WorkerEvaluator.healthCheck", () => {
  it("sends a lightweight evaluate request", async () => {
    sent.length = 0;
    const { WorkerEvaluator } = await import("./worker-evaluator");
    const buf = new ArrayBuffer(16);
    const evaluator = await WorkerEvaluator.fromArrayBuffer(buf, 15);

    sent.length = 0;
    await evaluator.healthCheck();

    expect(sent.filter((m) => m.type === "evaluate")).toHaveLength(1);
    await evaluator.dispose();
  });
});
