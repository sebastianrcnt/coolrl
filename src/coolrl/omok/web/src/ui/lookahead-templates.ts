export const LOOKAHEAD_TEMPLATES: readonly string[] = [
  "땀 {n}방울 흘리는 중...",
  "머리 {n}번 때리는 중...",
  "{n}수 앞을 보는 척 하는 중...",
  "{n}번 시뮬레이션 돌리는 중...",
  "예의상 {n}수 고민하는 척 하는 중...",
  "{n}가지 잡생각 하는 중...",
  "{n}번 짱구 굴리는 중...",
  "네 수를 {n}번 비웃는 중...",
  "인간은 왜이렇게 멍청한지 {n}번 비웃는 중...",
];

export function pickLookaheadTemplate(
  rng: () => number = Math.random
): string {
  const idx = Math.floor(rng() * LOOKAHEAD_TEMPLATES.length);
  return LOOKAHEAD_TEMPLATES[idx] ?? LOOKAHEAD_TEMPLATES[0]!;
}

export function formatLookahead(template: string, simsDone: number): string {
  return template.replace("{n}", String(Math.max(1, simsDone)));
}
