import EvaluatorWorker from "./worker?worker";
import type { Evaluator, PolicyValue } from "../core/mcts";
import type { GameState } from "../core/game-state";
import type { InferenceBackend } from "../util/backend";
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
    const evaluator = new WorkerEvaluator(boardSize, backend, lowMemory);
    const worker = new EvaluatorWorker();
    evaluator.worker = worker;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => evaluator.onMessage(event);
    worker.onerror = (event) => evaluator.onError(event);
    const workerBuffer = buf.slice(0);
    const response = (await evaluator.send(
      { type: "init", buf: workerBuffer, boardSize, backend, lowMemory },
      [workerBuffer]
    )) as InitSuccess;
    evaluator.backend = response.backend ?? backend;
    return evaluator;
  }

  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    for (const pending of this.pending.values()) {
      pending.reject(new Error("evaluator terminated"));
    }
    this.pending.clear();
  }

  async evaluate(states: GameState[]): Promise<PolicyValue> {
    if (!this.worker) throw new Error("evaluator not initialized");
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
      this.worker!.postMessage({ ...request, id }, transfer);
    });
  }

  private onMessage(event: MessageEvent<WorkerResponse>): void {
    const data = event.data;
    const pending = this.pending.get(data.id);
    if (!pending) return;
    this.pending.delete(data.id);
    if (data.ok) {
      pending.resolve(data);
    } else {
      pending.reject(new Error(data.error || "worker error"));
    }
  }

  private onError(event: ErrorEvent): void {
    const message = (event && event.message) || "worker crashed";
    for (const pending of this.pending.values()) {
      pending.reject(new Error(message));
    }
    this.pending.clear();
  }
}
