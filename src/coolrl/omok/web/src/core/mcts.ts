import type { GameState, Player } from "./game-state";
import { logDebug, logInfo, logWarn } from "../util/logger";

export interface PolicyValue {
  policy: ReadonlyArray<Float32Array>;
  value: ReadonlyArray<number>;
}

export interface Evaluator {
  evaluate(states: GameState[]): Promise<PolicyValue>;
}

export interface CandidateHint {
  action: number;
  visits: number;
  prior: number;
}

export interface ProgressCallback {
  (done: number, total: number, candidates: CandidateHint[]): void;
}

export interface RunOptions {
  reuseRoot?: TreeNode | null;
  onProgress?: ProgressCallback | null;
}

export interface RunResult {
  action: number;
  rootValue: number;
  nextRoot: TreeNode | null;
}

export class TreeNode {
  readonly toPlay: Player;
  prior: number;
  visitCount: number;
  valueSum: number;
  children: Map<number, TreeNode>;
  expanded: boolean;
  stateKey?: string;
  rootValue?: number;

  constructor(toPlay: Player, prior = 0, stateKey?: string) {
    this.toPlay = toPlay;
    this.prior = prior;
    this.visitCount = 0;
    this.valueSum = 0;
    this.children = new Map();
    this.expanded = false;
    this.stateKey = stateKey;
  }

  averageValue(): number {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }
}

interface SchedulerWithYield {
  yield: () => Promise<void>;
}

function yieldToBrowser(): Promise<void> {
  const sched = (globalThis as unknown as { scheduler?: SchedulerWithYield }).scheduler;
  if (sched && typeof sched.yield === "function") {
    return sched.yield();
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function stateKey(state: GameState): string {
  const terminal = state.terminal ? 1 : 0;
  const lastAction = state.lastAction ?? -1;
  return [
    state.boardSize,
    state.toPlay,
    state.moveCount,
    lastAction,
    state.winner,
    terminal,
    state.board.join(","),
  ].join("|");
}

export interface MctsOptions {
  cPuct?: number;
  evaluator: Evaluator;
  yieldEveryMs?: number;
  maxChildren?: number;
}

export class MCTS {
  private readonly cPuct: number;
  private readonly evaluator: Evaluator;
  private readonly yieldEveryMs: number;
  private readonly maxChildren: number;

  constructor(options: MctsOptions) {
    this.cPuct = options.cPuct ?? 1.6;
    this.evaluator = options.evaluator;
    this.yieldEveryMs = options.yieldEveryMs ?? 14;
    this.maxChildren = options.maxChildren ?? Infinity;
  }

  async run(state: GameState, numSims: number, options: RunOptions = {}): Promise<RunResult> {
    const { reuseRoot = null, onProgress = null } = options;
    const startedAt = performance.now();
    const currentKey = stateKey(state);
    const root =
      reuseRoot && reuseRoot.toPlay === state.toPlay && reuseRoot.stateKey === currentKey
        ? reuseRoot
        : new TreeNode(state.toPlay, 0, currentKey);
    logInfo("MCTS", "run.start", {
      reuseRoot: reuseRoot !== null && root === reuseRoot,
      moveCount: state.moveCount,
      toPlay: state.toPlay,
      numSims,
      currentBoardSize: state.boardSize,
    });
    if (root === reuseRoot) {
      logDebug("MCTS", "run.reusedRoot");
    } else {
      logDebug("MCTS", "run.newRoot");
    }

    if (!root.expanded && !state.terminal) {
      const { policy, value } = await this.evaluator.evaluate([state]);
      this.expand(root, state, policy[0]!);
      root.rootValue = value[0]!;
      onProgress?.(1, numSims, this.candidateHints(root, state));
      await yieldToBrowser();
      logDebug("MCTS", "run.rootExpanded", {
        actionCount: root.children.size,
        rootValue: root.rootValue,
      });
    }

    let yieldClock = performance.now();
    const progressStride = Math.max(1, Math.ceil(numSims / 20));

    for (let sim = 0; sim < numSims; sim++) {
      let node = root;
      const path: TreeNode[] = [node];
      const simState = state.clone();

      while (node.expanded && node.children.size > 0 && !simState.terminal) {
        const [action, child] = this.selectChild(node);
        simState.applyAction(action);
        const childKey = stateKey(simState);
        if (child.stateKey !== undefined && child.stateKey !== childKey) {
          logWarn("MCTS", "run.detectedStaleChild", {
            expected: child.stateKey,
            actual: childKey,
          });
          child.children.clear();
          child.expanded = false;
          child.visitCount = 0;
          child.valueSum = 0;
          child.rootValue = undefined;
        }
        child.stateKey = childKey;
        node = child;
        path.push(node);
      }

      let leafValue: number;
      if (simState.terminal) {
        leafValue = simState.outcomeForPlayer(simState.toPlay);
      } else {
        const { policy, value } = await this.evaluator.evaluate([simState]);
        this.expand(node, simState, policy[0]!);
        leafValue = value[0]!;
      }
      this.backup(path, leafValue);

      const now = performance.now();
      if (now - yieldClock >= this.yieldEveryMs) {
        onProgress?.(sim + 1, numSims, this.candidateHints(root, state));
        if ((sim + 1) % progressStride === 0 || sim + 1 === numSims) {
          logDebug("MCTS", "run.progress", {
            done: sim + 1,
            total: numSims,
            nodes: path.length,
          });
        }
        await yieldToBrowser();
        yieldClock = performance.now();
      }
    }
    onProgress?.(numSims, numSims, this.candidateHints(root, state));
    const result = this.chooseAction(root, state);
    logInfo("MCTS", "run.done", {
      action: result.action,
      rootValue: result.rootValue,
      elapsedMs: Number((performance.now() - startedAt).toFixed(1)),
      totalChildren: root.children.size,
    });
    return result;
  }

  private selectChild(node: TreeNode): [number, TreeNode] {
    const sqrtN = Math.sqrt(Math.max(1, node.visitCount));
    let bestAction = -1;
    let bestScore = -Infinity;
    let bestChild: TreeNode | null = null;
    for (const [action, child] of node.children) {
      const q = -child.averageValue();
      const u = (this.cPuct * child.prior * sqrtN) / (1 + child.visitCount);
      const score = q + u;
      if (score > bestScore) {
        bestScore = score;
        bestAction = action;
        bestChild = child;
      }
    }
    if (bestChild === null) throw new Error("selectChild called on node without children");
    return [bestAction, bestChild];
  }

  private expand(node: TreeNode, state: GameState, priors: Float32Array): void {
    const legal = this.limitLegalActions(state.legalIndices(), priors);
    let total = 0;
    for (const a of legal) total += priors[a] ?? 0;
    node.children.clear();
    node.stateKey = stateKey(state);
    const nextPlayer = -state.toPlay as Player;
    if (total <= 0 || !isFinite(total)) {
      const uniform = legal.length > 0 ? 1.0 / legal.length : 0;
      for (const a of legal) node.children.set(a, new TreeNode(nextPlayer, uniform));
    } else {
      for (const a of legal) {
        node.children.set(a, new TreeNode(nextPlayer, (priors[a] ?? 0) / total));
      }
    }
    node.expanded = true;
  }

  private limitLegalActions(legal: number[], priors: Float32Array): number[] {
    if (!Number.isFinite(this.maxChildren) || legal.length <= this.maxChildren) return legal;
    return legal
      .map((action): [number, number] => [
        action,
        Number.isFinite(priors[action]) ? priors[action]! : -Infinity,
      ])
      .sort((a, b) => b[1] - a[1])
      .slice(0, this.maxChildren)
      .map(([action]) => action);
  }

  private backup(path: TreeNode[], value: number): void {
    let v = value;
    for (let i = path.length - 1; i >= 0; i--) {
      const node = path[i]!;
      node.visitCount++;
      node.valueSum += v;
      v = -v;
    }
  }

  private candidateHints(root: TreeNode, state: GameState, limit = 10): CandidateHint[] {
    if (root.children.size === 0) return [];
    return [...root.children.entries()]
      .filter(([action]) => state.board[action] === 0)
      .map(([action, child]) => ({
        action,
        visits: child.visitCount,
        prior: child.prior,
      }))
      .sort((a, b) => b.visits - a.visits || b.prior - a.prior)
      .slice(0, limit);
  }

  private chooseAction(root: TreeNode, state: GameState): RunResult {
    let bestAction = -1;
    let bestCount = -1;
    let total = 0;
    for (const [action, child] of root.children) {
      if (state.board[action] !== 0) continue;
      total += child.visitCount;
      if (child.visitCount > bestCount) {
        bestCount = child.visitCount;
        bestAction = action;
      }
    }
    if (total === 0) {
      const legal = state.legalIndices();
      bestAction = legal[Math.floor(Math.random() * legal.length)] ?? -1;
    }
    const nextRoot = bestAction >= 0 ? root.children.get(bestAction) ?? null : null;
    if (nextRoot) {
      const nextState = state.clone();
      nextState.applyAction(bestAction);
      nextRoot.stateKey = stateKey(nextState);
    }
    return {
      action: bestAction,
      rootValue: root.averageValue(),
      nextRoot: state.terminal ? null : nextRoot,
    };
  }
}
