export function clamp01(t: number): number {
  return Math.max(0, Math.min(1, t));
}

export function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function easeInCubic(t: number): number {
  return t * t * t;
}

export function easeOutBack(t: number): number {
  const c = 1.70158;
  const s = c + 1;
  return 1 + s * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2);
}
