import EvaluatorWorker from "./worker?worker";
import type { Evaluator, PolicyValue } from "../core/mcts";
import type { GameState } from "../core/game-state";
import type { InferenceBackend } from "../util/backend";
import { logDebug, logError, logInfo, logWarn } from "../util/logger";
import type {
  EvaluateSuccess,
  InitSuccess,
  StateSnapshot,
  WorkerRequest,
  WorkerResponse,
} from "./protocol";

interface Pending {
  resolve(value: WorkerResponse): void;
  reject(error: Error): void;
}

function toSnapshot(state: GameState): StateSnapshot {
  return {
    boardSize: state.boardSize,
    board: state.board,
    toPlay: state.toPlay,
    lastAction: state.lastAction,
  };
}

export class WorkerEvaluator implements Evaluator {
  readonly boardSize: number;
  readonly actionSize: number;
  readonly lowMemory: boolean;
  backend: InferenceBackend;

  private worker: Worker | null;
  private readonly pending = new Map<number, Pending>();
  private nextId = 1;

  private constructor(boardSize: number, backend: InferenceBackend, lowMemory: boolean) {
    this.boardSize = boardSize;
    this.actionSize = boardSize * boardSize;
    this.backend = backend;
    this.lowMemory = lowMemory;
    this.worker = null;
  }

  static async fromArrayBuffer(
    buf: ArrayBuffer,
    boardSize: number,
    backend: InferenceBackend = "wasm",
    lowMemory = false
  ): Promise<WorkerEvaluator> {
    const started = performance.now();
    const evaluator = new WorkerEvaluator(boardSize, backend, lowMemory);
    const worker = new EvaluatorWorker();
    evaluator.worker = worker;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => evaluator.onMessage(event);
    worker.onerror = (event) => evaluator.onError(event);
    const workerBuffer = buf.slice(0);
    logDebug("WorkerEvaluator", "fromArrayBuffer.send", {
      boardSize,
      backend,
      lowMemory,
      bytes: buf.byteLength,
    });
    const response = (await evaluator.send(
      { type: "init", buf: workerBuffer, boardSize, backend, lowMemory },
      [workerBuffer]
    )) as InitSuccess;
    evaluator.backend = response.backend ?? backend;
    logInfo("WorkerEvaluator", "fromArrayBuffer.ready", {
      backend: evaluator.backend,
      elapsedMs: Number((performance.now() - started).toFixed(1)),
    });
    return evaluator;
  }

  terminate(): void {
    if (this.worker) {
      logWarn("WorkerEvaluator", "terminate", {
        pending: this.pending.size,
      });
      this.worker.terminate();
      this.worker = null;
    }
    for (const pending of this.pending.values()) {
      pending.reject(new Error("evaluator terminated"));
    }
    this.pending.clear();
  }

  // Graceful teardown: ask the worker to call InferenceSession.release() so
  // GPU resources are freed deterministically, then terminate the worker.
  // Falls back to a hard terminate if the worker doesn't answer within
  // `timeoutMs` (iOS Safari with a lost WebGPU device can hang on release).
  async dispose(timeoutMs = 3000): Promise<void> {
    if (!this.worker) return;
    logDebug("WorkerEvaluator", "dispose.start", {
      backend: this.backend,
      pending: this.pending.size,
    });
    const release = this.send({ type: "dispose" }).then(
      () => "released" as const,
      (err) => {
        logWarn("WorkerEvaluator", "dispose.releaseFailed", {
          error: err instanceof Error ? err.message : String(err),
        });
        return "failed" as const;
      }
    );
    const timeout = new Promise<"timeout">((resolve) => {
      setTimeout(() => resolve("timeout"), timeoutMs);
    });
    const outcome = await Promise.race([release, timeout]);
    if (outcome === "timeout") {
      logWarn("WorkerEvaluator", "dispose.timeout", { timeoutMs });
    }
    this.terminate();
  }

  async evaluate(states: GameState[]): Promise<PolicyValue> {
    if (!this.worker) throw new Error("evaluator not initialized");
    const t0 = performance.now();
    logDebug("WorkerEvaluator", "evaluate.start", { batch: states.length });
    const snapshots = states.map(toSnapshot);
    const response = (await this.send({
      type: "evaluate",
      states: snapshots,
    })) as EvaluateSuccess;
    const policyArr = new Float32Array(response.policy);
    const valuesArr = new Float32Array(response.values);
    const { batch, actionSize } = response;
    const policy: Float32Array[] = [];
    const value: number[] = [];
    for (let b = 0; b < batch; b++) {
      policy.push(policyArr.subarray(b * actionSize, (b + 1) * actionSize));
      value.push(valuesArr[b]!);
    }
    logDebug("WorkerEvaluator", "evaluate.done", {
      batch,
      actionSize,
      elapsedMs: Number((performance.now() - t0).toFixed(1)),
    });
    return { policy, value };
  }

  private send(
    request: { type: string } & Record<string, unknown>,
    transfer: Transferable[] = []
  ): Promise<WorkerResponse> {
    if (!this.worker) return Promise.reject(new Error("evaluator not initialized"));
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      logDebug("WorkerEvaluator", "send", {
        id,
        type: request.type,
        hasTransfer: transfer.length > 0,
      });
      this.worker!.postMessage({ ...request, id }, transfer);
    });
  }

  private onMessage(event: MessageEvent<WorkerResponse>): void {
    const data = event.data;
    const pending = this.pending.get(data.id);
    if (!pending) return;
    this.pending.delete(data.id);
    logDebug("WorkerEvaluator", "onMessage", {
      id: data.id,
      ok: data.ok,
    });
    if (data.ok) {
      pending.resolve(data);
    } else {
      pending.reject(new Error(data.error || "worker error"));
    }
  }

  private onError(event: ErrorEvent): void {
    const message = (event && event.message) || "worker crashed";
    logError("WorkerEvaluator", "onError", message);
    for (const pending of this.pending.values()) {
      pending.reject(new Error(message));
    }
    this.pending.clear();
  }
}
