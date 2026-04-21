export interface OrtTensor {
  readonly data: Float32Array;
  readonly dims: readonly number[];
  dispose?(): void;
}

export interface OrtTensorConstructor {
  new (type: "float32", data: Float32Array, dims: readonly number[]): OrtTensor;
}

export interface OrtInferenceSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

export interface OrtSessionOptions {
  executionProviders: string[];
  graphOptimizationLevel?: "basic" | "all";
  enableCpuMemArena?: boolean;
  enableMemPattern?: boolean;
  executionMode?: "sequential" | "parallel";
  preferredOutputLocation?: "cpu" | "gpu-buffer" | Record<string, "cpu" | "gpu-buffer">;
}

export interface OrtInferenceSessionConstructor {
  create(buf: ArrayBuffer, options: OrtSessionOptions): Promise<OrtInferenceSession>;
}

export interface OrtRuntime {
  Tensor: OrtTensorConstructor;
  InferenceSession: OrtInferenceSessionConstructor;
  env: {
    wasm: {
      wasmPaths: string | Record<string, string>;
      numThreads: number;
      proxy: boolean;
    };
  };
}
