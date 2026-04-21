export type LogLevel = "debug" | "info" | "warn" | "error";

const LOG_PREFIX = "[omok-web]";
const APP_BOOT_MS = typeof performance === "undefined" ? 0 : performance.now();
let debugEnabled: boolean | null = null;

type LogMethod = "debug" | "log";

function hasConsole(): globalThis is Window & Console {
  return typeof globalThis !== "undefined" && "console" in globalThis;
}

function nowStamp(): string {
  if (typeof performance === "undefined") return "t=0.0ms";
  return `+${(performance.now() - APP_BOOT_MS).toFixed(1)}ms`;
}

function resolveDebugEnabled(): boolean {
  if (debugEnabled !== null) return debugEnabled;
  try {
    const global: unknown = globalThis as unknown;
    if (global && typeof global === "object") {
      const anyGlobal = global as {
        process?: { env?: { NODE_ENV?: string } };
        location?: { search?: string };
        localStorage?: Storage;
      };
      if (anyGlobal.process?.env?.NODE_ENV === "development") {
        debugEnabled = true;
        return true;
      }
      if (anyGlobal.location && typeof anyGlobal.location.search === "string") {
        const params = new URLSearchParams(anyGlobal.location.search);
        const flag = params.get("omokLog");
        if (flag === "1" || (flag && flag.toLowerCase() === "true")) {
          debugEnabled = true;
          return true;
        }
      }
      const storage = anyGlobal.localStorage;
      if (storage && typeof storage.getItem === "function") {
        const stored = storage.getItem("OMOK_WEB_DEBUG");
        if (stored === "1" || (stored && stored.toLowerCase() === "true")) {
          debugEnabled = true;
          return true;
        }
      }
    }
  } catch {
    // no-op
  }
  debugEnabled = false;
  return false;
}

function emit(level: LogLevel, component: string, event: string, details?: unknown): void {
  if (!hasConsole()) return;
  if (level === "debug" && !resolveDebugEnabled()) return;

  const method: LogMethod = level === "debug" ? "debug" : "log";
  const prefix = `${LOG_PREFIX} ${nowStamp()} [${level.toUpperCase()}] [${component}]`;
  const line = `${prefix} ${event}`;
  if (details === undefined) {
    console[method](line);
    return;
  }
  console[method](line, details);
}

export function setDebugLogging(enabled: boolean): void {
  debugEnabled = enabled;
}

export function logDebug(component: string, event: string, details?: unknown): void {
  emit("debug", component, event, details);
}

export function logInfo(component: string, event: string, details?: unknown): void {
  emit("info", component, event, details);
}

export function logWarn(component: string, event: string, details?: unknown): void {
  emit("warn", component, event, details);
}

export function logError(component: string, event: string, details?: unknown): void {
  emit("error", component, event, details);
}
