import type { InferenceBackend } from "../util/backend";

export interface StateSnapshot {
  boardSize: number;
  board: Int8Array;
  toPlay: number;
  lastAction: number | null;
}

export interface InitRequest {
  type: "init";
  id: number;
  buf: ArrayBuffer;
  boardSize: number;
  backend: InferenceBackend;
  lowMemory: boolean;
}

export interface EvaluateRequest {
  type: "evaluate";
  id: number;
  states: StateSnapshot[];
}

export interface DisposeRequest {
  type: "dispose";
  id: number;
}

export type WorkerRequest = InitRequest | EvaluateRequest | DisposeRequest;

export interface InitSuccess {
  id: number;
  ok: true;
  backend: InferenceBackend;
}

export interface EvaluateSuccess {
  id: number;
  ok: true;
  policy: ArrayBuffer;
  values: ArrayBuffer;
  batch: number;
  actionSize: number;
}

export interface DisposeSuccess {
  id: number;
  ok: true;
  disposed: true;
}

export interface ErrorResponse {
  id: number;
  ok: false;
  error: string;
}

export type WorkerResponse =
  | InitSuccess
  | EvaluateSuccess
  | DisposeSuccess
  | ErrorResponse;
