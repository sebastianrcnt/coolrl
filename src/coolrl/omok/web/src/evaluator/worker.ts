/// <reference lib="webworker" />
//
// Worker-side ONNX evaluator. Runs on a DedicatedWorkerGlobalScope so UI
// rendering / input / animation keep flowing while inference is in-flight.
//
// ONNX Runtime is still fetched from the CDN at runtime via importScripts so
// the bundle stays lean and the user's browser cache is shared with other
// sites that use the same ort build. The Blob-from-string trick from the old
// WORKER_SRC is gone — Vite bundles this module separately (classic worker)
// and imports the main-thread glue via typed messages.
//
import { normalizeBackend, type InferenceBackend } from "../util/backend";
import type {
  EvaluateRequest,
  InitRequest,
  StateSnapshot,
  WorkerRequest,
  WorkerResponse,
} from "./protocol";
import type { OrtInferenceSession, OrtRuntime } from "./ort-types";

declare const self: DedicatedWorkerGlobalScope;
declare function importScripts(...urls: string[]): void;
declare const ort: OrtRuntime;

const ORT_CDN_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";

class EvaluatorSession {
  private session: OrtInferenceSession | null = null;
  private activeBackend: InferenceBackend = "wasm";
  private ortLoaded = false;
  private boardSize = 9;
  private planeSize = 81;
  private actionSize = 81;
  private featureBuffer: Float32Array | null = null;

  get backend(): InferenceBackend {
    return this.activeBackend;
  }

  async init(request: InitRequest): Promise<void> {
    this.setSize(request.boardSize);
    this.activeBackend = normalizeBackend(request.backend);
    this.loadOrt(this.activeBackend);
    const sessionOptions = this.buildSessionOptions(this.activeBackend, request.lowMemory);
    this.session = await ort.InferenceSession.create(request.buf, sessionOptions);
    await this.warmUp();
  }

  async evaluate(request: EvaluateRequest): Promise<{
    policy: ArrayBuffer;
    values: ArrayBuffer;
    batch: number;
    actionSize: number;
  }> {
    if (!this.session) throw new Error("evaluator not initialized");
    const batch = request.states.length;
    const features = this.encodeFeatures(request.states);
    const tensor = new ort.Tensor("float32", features, [batch, 4, this.boardSize, this.boardSize]);
    const output = await this.session.run({ input: tensor });
    const logits = output.policy_logits?.data;
    const values = output.value?.data;
    if (!logits || !values) throw new Error("missing policy_logits or value output");
    if (logits.length !== batch * this.actionSize) {
      throw new Error(
        `policy size ${logits.length / batch} != action size ${this.actionSize}`
      );
    }
    const policy = new Float32Array(logits.length);
    policy.set(logits);
    for (let b = 0; b < batch; b++) {
      softmaxInPlace(policy, b * this.actionSize, this.actionSize);
    }
    const valuesOut = new Float32Array(values);
    return {
      policy: policy.buffer,
      values: valuesOut.buffer,
      batch,
      actionSize: this.actionSize,
    };
  }

  private setSize(n: number): void {
    this.boardSize = n;
    this.planeSize = n * n;
    this.actionSize = n * n;
  }

  private loadOrt(backend: InferenceBackend): void {
    if (this.ortLoaded) return;
    const useJsep = backend === "webgpu" || backend === "webnn";
    importScripts(ORT_CDN_BASE + (useJsep ? "ort.webgpu.min.js" : "ort.wasm.min.js"));
    ort.env.wasm.wasmPaths = wasmPaths(backend);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    this.ortLoaded = true;
  }

  private buildSessionOptions(backend: InferenceBackend, lowMemory: boolean) {
    const options = {
      executionProviders: [backend],
      graphOptimizationLevel: lowMemory ? ("basic" as const) : ("all" as const),
    };
    if (backend !== "wasm") return options;
    return {
      ...options,
      enableCpuMemArena: !lowMemory,
      enableMemPattern: !lowMemory,
      executionMode: "sequential" as const,
    };
  }

  private async warmUp(): Promise<void> {
    const emptyFeat = new Float32Array(4 * this.planeSize);
    // Color plane set to 1 so the warm-up shape matches what evaluate() feeds.
    for (let i = 0; i < this.planeSize; i++) emptyFeat[3 * this.planeSize + i] = 1.0;
    const warm = new ort.Tensor("float32", emptyFeat, [1, 4, this.boardSize, this.boardSize]);
    await this.session!.run({ input: warm });
  }

  private encodeFeatures(states: StateSnapshot[]): Float32Array {
    const needed = states.length * 4 * this.planeSize;
    if (!this.featureBuffer || this.featureBuffer.length !== needed) {
      this.featureBuffer = new Float32Array(needed);
    } else {
      this.featureBuffer.fill(0);
    }
    const out = this.featureBuffer;
    for (let b = 0; b < states.length; b++) {
      const state = states[b]!;
      const base = b * 4 * this.planeSize;
      const toPlay = state.toPlay;
      const colorValue = toPlay === 1 ? 1.0 : 0.0;
      const board = state.board;
      for (let i = 0; i < this.planeSize; i++) {
        const v = board[i];
        out[base + i] = v === toPlay ? 1.0 : 0.0;
        out[base + this.planeSize + i] = v === -toPlay ? 1.0 : 0.0;
        out[base + 3 * this.planeSize + i] = colorValue;
      }
      if (state.lastAction !== null && state.lastAction >= 0) {
        out[base + 2 * this.planeSize + state.lastAction] = 1.0;
      }
    }
    return out;
  }
}

function wasmPaths(backend: InferenceBackend): Record<string, string> {
  const useJsep = backend === "webgpu" || backend === "webnn";
  const suffix = useJsep ? ".jsep" : "";
  return {
    wasm: `${ORT_CDN_BASE}ort-wasm-simd-threaded${suffix}.wasm`,
    mjs: `${ORT_CDN_BASE}ort-wasm-simd-threaded${suffix}.mjs`,
  };
}

function softmaxInPlace(arr: Float32Array, offset: number, len: number): void {
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = arr[offset + i]!;
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const exp = Math.exp(arr[offset + i]! - max);
    arr[offset + i] = exp;
    sum += exp;
  }
  if (sum > 0) {
    for (let i = 0; i < len; i++) arr[offset + i]! /= sum;
  }
}

const evaluator = new EvaluatorSession();

function post(response: WorkerResponse, transfer?: Transferable[]): void {
  self.postMessage(response, transfer ?? []);
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.type === "init") {
      await evaluator.init(msg);
      post({ id: msg.id, ok: true, backend: evaluator.backend });
      return;
    }
    if (msg.type === "evaluate") {
      const result = await evaluator.evaluate(msg);
      post(
        {
          id: msg.id,
          ok: true,
          policy: result.policy,
          values: result.values,
          batch: result.batch,
          actionSize: result.actionSize,
        },
        [result.policy, result.values]
      );
      return;
    }
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    post({ id: msg.id, ok: false, error });
  }
};
