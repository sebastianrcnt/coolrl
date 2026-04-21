export interface CellMetrics {
  width: number;
  height: number;
  square: number;
  offsetX: number;
  offsetY: number;
  margin: number;
  step: number;
}

export interface Cell {
  row: number;
  col: number;
}

export function starPoints(boardSize: number): Array<[number, number]> {
  if (boardSize < 9) {
    const center = Math.floor(boardSize / 2);
    return [[center, center]];
  }
  const edge = boardSize >= 13 ? 3 : 2;
  const center = Math.floor(boardSize / 2);
  return [
    [edge, edge],
    [edge, boardSize - 1 - edge],
    [center, center],
    [boardSize - 1 - edge, edge],
    [boardSize - 1 - edge, boardSize - 1 - edge],
  ];
}

export function computeCellMetrics(
  canvasWidth: number,
  canvasHeight: number,
  boardSize: number,
  marginRatio: number
): CellMetrics {
  const square = Math.min(canvasWidth, canvasHeight);
  const offsetX = (canvasWidth - square) / 2;
  const offsetY = (canvasHeight - square) / 2;
  const margin = square * marginRatio;
  const step = (square - 2 * margin) / (boardSize - 1);
  return { width: canvasWidth, height: canvasHeight, square, offsetX, offsetY, margin, step };
}

export function cellX(metrics: CellMetrics, col: number): number {
  return metrics.offsetX + metrics.margin + col * metrics.step;
}

export function cellY(metrics: CellMetrics, row: number): number {
  return metrics.offsetY + metrics.margin + row * metrics.step;
}

export function pixelToCell(
  metrics: CellMetrics,
  boardSize: number,
  x: number,
  y: number
): Cell | null {
  const { margin, step, offsetX, offsetY } = metrics;
  const col = Math.round((x - offsetX - margin) / step);
  const row = Math.round((y - offsetY - margin) / step);
  if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) return null;
  const dx = Math.abs(x - offsetX - (margin + col * step));
  const dy = Math.abs(y - offsetY - (margin + row * step));
  if (dx > step * 0.5 || dy > step * 0.5) return null;
  return { row, col };
}

export function actionToRowCol(action: number, boardSize: number): Cell {
  return { row: Math.floor(action / boardSize), col: action % boardSize };
}
