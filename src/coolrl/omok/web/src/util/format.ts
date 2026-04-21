export function formatSignedValue(value: number | null | undefined): string | null {
  if (value === null || value === undefined) return null;
  const sign = value >= 0 ? "+" : "";
  return sign + value.toFixed(2);
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "";
  const digits = ms < 10_000 ? 1 : 0;
  return `${(ms / 1000).toFixed(digits)}초`;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "n/a";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  const digits = unit === 0 ? 0 : value < 10 ? 1 : 0;
  return `${value.toFixed(digits)}${units[unit]}`;
}

export function escapeHtml(value: unknown): string {
  const replacements: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };
  return String(value).replace(/[&<>"']/g, (ch) => replacements[ch] ?? ch);
}
