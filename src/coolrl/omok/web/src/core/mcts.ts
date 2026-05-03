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
  // Action selection temperature applied to root visit counts.
  //   τ = 0  → argmax (default, strongest play)
  //   τ > 0  → sample from N(s,a)^(1/τ); larger τ flattens the distribution
  // Used to deliberately weaken AI moves so the user can win more often.
  temperature?: number;
  // Optional RNG hook for deterministic tests. Defaults to Math.random.
  random?: () => number;
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
    const {
      reuseRoot = null,
      onProgress = null,
      temperature = 0,
      random = Math.random,
    } = options;
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
    const result = this.chooseAction(root, state, temperature, random);
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

  private chooseAction(
    root: TreeNode,
    state: GameState,
    temperature: number,
    random: () => number
  ): RunResult {
    const candidates: Array<{ action: number; visits: number }> = [];
    let total = 0;
    let bestAction = -1;
    let bestCount = -1;
    for (const [action, child] of root.children) {
      if (state.board[action] !== 0) continue;
      total += child.visitCount;
      candidates.push({ action, visits: child.visitCount });
      if (child.visitCount > bestCount) {
        bestCount = child.visitCount;
        bestAction = action;
      }
    }
    if (total === 0) {
      const legal = state.legalIndices();
      bestAction = legal[Math.floor(random() * legal.length)] ?? -1;
    } else if (temperature > 0 && candidates.length > 1) {
      bestAction = sampleByVisitTemperature(candidates, temperature, random);
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

// Sample an action by N(s,a)^(1/τ). High τ flattens; very high τ → uniform
// over visited actions. Falls back to argmax on numerical breakdown.
function sampleByVisitTemperature(
  candidates: ReadonlyArray<{ action: number; visits: number }>,
  temperature: number,
  random: () => number
): number {
  const invT = 1 / temperature;
  let maxLogVisits = -Infinity;
  for (const c of candidates) {
    if (c.visits <= 0) continue;
    const lv = Math.log(c.visits);
    if (lv > maxLogVisits) maxLogVisits = lv;
  }
  if (!Number.isFinite(maxLogVisits)) {
    // No visited candidates — pick the listed argmax.
    let best = candidates[0]!.action;
    let bestV = -1;
    for (const c of candidates) {
      if (c.visits > bestV) { bestV = c.visits; best = c.action; }
    }
    return best;
  }
  const weights: number[] = [];
  let sum = 0;
  for (const c of candidates) {
    const w = c.visits > 0
      ? Math.exp((Math.log(c.visits) - maxLogVisits) * invT)
      : 0;
    weights.push(w);
    sum += w;
  }
  if (!(sum > 0) || !Number.isFinite(sum)) {
    let best = candidates[0]!.action;
    let bestV = -1;
    for (const c of candidates) {
      if (c.visits > bestV) { bestV = c.visits; best = c.action; }
    }
    return best;
  }
  let r = random() * sum;
  for (let i = 0; i < candidates.length; i++) {
    r -= weights[i]!;
    if (r <= 0) return candidates[i]!.action;
  }
  return candidates[candidates.length - 1]!.action;
}
