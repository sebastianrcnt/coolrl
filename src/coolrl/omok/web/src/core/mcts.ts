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

// Parameters that control how the AI deliberately weakens its move selection.
// All filters are applied before sampling so "junk" tail moves never surface.
export interface WeakeningParams {
  temperature: number;    // softmax temperature over filtered candidates
  topK: number;           // keep at most this many candidates by visit count
  minVisitRatio: number;  // drop candidates with N < minVisitRatio * maxN
  qDrop: number;          // drop candidates with Q < bestQ - qDrop
  qWeight: number;        // weight of Q term in softmax score
  priorWeight: number;    // weight of log(P) term in softmax score
}

export interface RunOptions {
  reuseRoot?: TreeNode | null;
  onProgress?: ProgressCallback | null;
  // When set, weakens the AI's move selection so the user wins more often.
  // null / omitted → argmax (strongest play, no weakening).
  weakening?: WeakeningParams | null;
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
      weakening = null,
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
    const result = this.chooseAction(root, state, numSims, weakening, random);
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
    numSims: number,
    weakening: WeakeningParams | null,
    random: () => number
  ): RunResult {
    // Collect visited legal children with Q from current-player perspective.
    // child.averageValue() is from the child node's to-play perspective
    // (opponent of root), so we negate to get root's perspective.
    const candidates: ActionCandidate[] = [];
    let hasAny = false;
    for (const [action, child] of root.children) {
      if (state.board[action] !== 0) continue;
      if (child.visitCount > 0) hasAny = true;
      candidates.push({
        action,
        visits: child.visitCount,
        prior: child.prior,
        q: -child.averageValue(),
      });
    }

    let bestAction: number;
    if (!hasAny) {
      const legal = state.legalIndices();
      bestAction = legal[Math.floor(random() * legal.length)] ?? -1;
    } else if (weakening && weakening.temperature > 0) {
      bestAction = chooseMoveWithWeakening(candidates, weakening, numSims, random);
    } else {
      bestAction = argmaxByVisits(candidates);
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

interface ActionCandidate {
  action: number;
  visits: number;
  prior: number;
  q: number; // from current-player perspective; higher = better for AI
}

function argmaxByVisits(candidates: ActionCandidate[]): number {
  let best = candidates[0]!;
  for (let i = 1; i < candidates.length; i++) {
    if (candidates[i]!.visits > best.visits) best = candidates[i]!;
  }
  return best.action;
}

// Filtered softmax move selection.
// 1. visit ratio cut   — drop junk-tail moves
// 2. top-K cut         — cap candidate pool size
// 3. Q-drop cut        — drop obvious blunders
// 4. forced-move guard — if 1 candidate left, pick it immediately
// 5. dominant-move bypass — if best clearly dominates, skip sampling
// 6. softmax sample    — natural wobble inside the plausible pool
function chooseMoveWithWeakening(
  candidates: ActionCandidate[],
  params: WeakeningParams,
  numSims: number,
  random: () => number
): number {
  if (candidates.length === 0) throw new Error("no candidates");
  if (candidates.length === 1) return candidates[0]!.action;

  const sorted = [...candidates]
    .filter(c => c.visits > 0)
    .sort((a, b) => b.visits - a.visits);

  if (sorted.length === 0) return argmaxByVisits(candidates);

  const best = sorted[0]!;
  const bestQ = Math.max(...sorted.map(c => c.q));

  // Dominant-move bypass: if best is overwhelmingly visited, don't sample.
  // Threshold is sims-proportional so 96-sims and 256-sims behave consistently.
  // Applied only at lower weakness levels (topK <= 5) to protect forced moves
  // without over-riding weaker difficulty levels.
  if (params.topK <= 5 && sorted.length >= 2) {
    const second = sorted[1]!;
    const visitDominance =
      best.visits >= second.visits * 3.0 && best.visits >= 0.20 * numSims;
    const qDominance =
      best.q >= second.q + 0.25 && best.visits >= 0.13 * numSims;
    if (visitDominance || qDominance) return best.action;
  }

  // 1. visit ratio cut
  let pool = sorted.filter(c => c.visits >= best.visits * params.minVisitRatio);
  if (pool.length === 0) pool = [best];

  // 2. top-K cut
  pool = pool.slice(0, params.topK);

  // 3. Q-drop cut (argmax is always kept)
  pool = pool.filter(c => c.action === best.action || c.q >= bestQ - params.qDrop);
  if (pool.length === 0) pool = [best];

  // 4. single candidate
  if (pool.length === 1) return pool[0]!.action;

  // 5. softmax sampling
  const logits = pool.map(c => {
    const visitScore = Math.log(c.visits + 1);
    const qScore = params.qWeight * c.q;
    const priorScore = params.priorWeight * Math.log(Math.max(c.prior, 1e-8));
    return (visitScore + qScore + priorScore) / params.temperature;
  });

  const maxLogit = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (!(sum > 0) || !Number.isFinite(sum)) return best.action;

  let r = random() * sum;
  for (let i = 0; i < exps.length; i++) {
    r -= exps[i]!;
    if (r <= 0) return pool[i]!.action;
  }
  return pool[pool.length - 1]!.action;
}
