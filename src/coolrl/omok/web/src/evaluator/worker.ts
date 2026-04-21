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
import { logDebug, logError, logInfo, logWarn } from "../util/logger";
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

  async dispose(): Promise<void> {
    const session = this.session;
    this.session = null;
    this.featureBuffer = null;
    if (!session) {
      logDebug("EvaluatorWorker", "dispose.noSession");
      return;
    }
    logInfo("EvaluatorWorker", "dispose.start", { backend: this.activeBackend });
    if (typeof session.release === "function") {
      await session.release();
    }
    logInfo("EvaluatorWorker", "dispose.done");
  }

  async init(request: InitRequest): Promise<void> {
    this.setSize(request.boardSize);
    this.activeBackend = normalizeBackend(request.backend);
    logInfo("EvaluatorWorker", "init.start", {
      boardSize: request.boardSize,
      backend: request.backend,
      lowMemory: request.lowMemory,
      bufBytes: request.buf.byteLength,
    });
    this.loadOrt(this.activeBackend);
    const sessionOptions = this.buildSessionOptions(this.activeBackend, request.lowMemory);
    this.session = await ort.InferenceSession.create(request.buf, sessionOptions);
    await this.warmUp();
    logInfo("EvaluatorWorker", "init.done", {
      boardSize: this.boardSize,
      backend: this.activeBackend,
    });
  }

  async evaluate(request: EvaluateRequest): Promise<{
    policy: ArrayBuffer;
    values: ArrayBuffer;
    batch: number;
    actionSize: number;
  }> {
    if (!this.session) throw new Error("evaluator not initialized");
    const t0 = performance.now();
    logDebug("EvaluatorWorker", "evaluate.start", {
      batch: request.states.length,
      boardSize: this.boardSize,
    });
    const batch = request.states.length;
    const features = this.encodeFeatures(request.states);
    logDebug("EvaluatorWorker", "evaluate.featuresPrepared", {
      batch,
      featureLength: features.length,
    });
    const tensor = new ort.Tensor("float32", features, [batch, 4, this.boardSize, this.boardSize]);
    let output: Record<string, OrtTensor> | null = null;
    try {
      output = await this.session.run({ input: tensor });
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
    } finally {
      // Explicitly release GPU-backed buffers on each run. On WASM these are
      // no-ops, but iOS Safari's WebGPU EP holds tensor resources much longer
      // than desktop Chrome and accumulated inputs/outputs cause tab death.
      tensor.dispose?.();
      if (output) {
        for (const key of Object.keys(output)) output[key]?.dispose?.();
      }
    }
  }

  private setSize(n: number): void {
    const prev = this.boardSize;
    this.boardSize = n;
    this.planeSize = n * n;
    this.actionSize = n * n;
    if (prev !== n) {
      logDebug("EvaluatorWorker", "setSize", { prev, next: n });
    }
  }

  private loadOrt(backend: InferenceBackend): void {
    logDebug("EvaluatorWorker", "loadOrt", { backend, alreadyLoaded: this.ortLoaded });
    if (this.ortLoaded) return;
    const useWebGpuLoader = backend === "webgpu" || backend === "webnn";
    importScripts(ORT_CDN_BASE + (useWebGpuLoader ? "ort.webgpu.min.js" : "ort.wasm.min.js"));
    // Hand the loader only the CDN base URL and let it construct its own
    // filename (e.g. ort-wasm-simd-threaded.asyncify.mjs for the native
    // WebGPU/WebNN EPs in 1.24+). Pinning explicit filenames couples us to
    // the internal wasm-glue naming, which has changed each minor release.
    ort.env.wasm.wasmPaths = ORT_CDN_BASE;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    this.ortLoaded = true;
  }

  private buildSessionOptions(backend: InferenceBackend, lowMemory: boolean) {
    const options = {
      executionProviders: [backend],
      graphOptimizationLevel: "all" as const,
    };
    if (backend === "wasm") {
      return {
        ...options,
        enableCpuMemArena: !lowMemory,
        enableMemPattern: !lowMemory,
        executionMode: "sequential" as const,
      };
    }
    // For WebGPU/WebNN, ask ORT to copy outputs back to CPU so the GPU-side
    // output buffers are eligible for release the moment run() resolves.
    return {
      ...options,
      preferredOutputLocation: "cpu" as const,
    };
  }

  private async warmUp(): Promise<void> {
    logDebug("EvaluatorWorker", "warmUp.start");
    const emptyFeat = new Float32Array(4 * this.planeSize);
    // Color plane set to 1 so the warm-up shape matches what evaluate() feeds.
    for (let i = 0; i < this.planeSize; i++) emptyFeat[3 * this.planeSize + i] = 1.0;
    const warm = new ort.Tensor("float32", emptyFeat, [1, 4, this.boardSize, this.boardSize]);
    let warmOut: Record<string, OrtTensor> | null = null;
    try {
      warmOut = await this.session!.run({ input: warm });
    } finally {
      warm.dispose?.();
      if (warmOut) {
        for (const key of Object.keys(warmOut)) warmOut[key]?.dispose?.();
      }
    }
    logDebug("EvaluatorWorker", "warmUp.done");
  }

  private encodeFeatures(states: StateSnapshot[]): Float32Array {
    logDebug("EvaluatorWorker", "encodeFeatures", { batch: states.length });
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
  logDebug("EvaluatorWorker", "post", {
    id: response.id,
    ok: response.ok,
  });
  self.postMessage(response, transfer ?? []);
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  logDebug("EvaluatorWorker", "onmessage", {
    type: msg?.type,
    id: msg?.id,
  });
  try {
    if (msg.type === "init") {
      await evaluator.init(msg);
      logDebug("EvaluatorWorker", "onmessage.initDone", { id: msg.id });
      post({ id: msg.id, ok: true, backend: evaluator.backend });
      return;
    }
    if (msg.type === "evaluate") {
      const result = await evaluator.evaluate(msg);
      logDebug("EvaluatorWorker", "onmessage.evaluateDone", {
        id: msg.id,
        batch: result.batch,
      });
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
    if (msg.type === "dispose") {
      await evaluator.dispose();
      logDebug("EvaluatorWorker", "onmessage.disposeDone", { id: msg.id });
      post({ id: msg.id, ok: true, disposed: true });
      return;
    }
  } catch (err) {
    logError("EvaluatorWorker", "onmessage.error", {
      id: msg?.id,
      message: err instanceof Error ? err.message : String(err),
    });
    const error = err instanceof Error ? err.message : String(err);
    post({ id: msg.id, ok: false, error });
  }
};
