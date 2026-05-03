import { describe, it, expect } from "vitest";
import { MCTS } from "./mcts";
import { GameState } from "./game-state";
import type { Evaluator, PolicyValue } from "./mcts";

class UniformEvaluator implements Evaluator {
  calls = 0;

  async evaluate(states: GameState[]): Promise<PolicyValue> {
    this.calls++;
    const policy = states.map((s) => {
      const p = new Float32Array(s.actionSize);
      p.fill(1 / s.actionSize);
      return p;
    });
    return { policy, value: states.map(() => 0) };
  }
}

describe("MCTS.run", () => {
  it("returns a legal action", async () => {
    const game = new GameState(9);
    const mcts = new MCTS({ evaluator: new UniformEvaluator(), cPuct: 1.0 });
    const result = await mcts.run(game, 8);
    expect(game.legalIndices()).toContain(result.action);
  });

  it("returns a nextRoot node with the opponent as toPlay", async () => {
    const game = new GameState(9);
    const mcts = new MCTS({ evaluator: new UniformEvaluator() });
    const result = await mcts.run(game, 8);
    expect(result.nextRoot).not.toBeNull();
    expect(result.nextRoot!.toPlay).toBe(-1);
  });

  it("rootValue is a finite number", async () => {
    const game = new GameState(9);
    const mcts = new MCTS({ evaluator: new UniformEvaluator() });
    const result = await mcts.run(game, 4);
    expect(Number.isFinite(result.rootValue)).toBe(true);
  });

  it("reuseRoot reuses tree when toPlay matches", async () => {
    const game = new GameState(9);
    const mcts = new MCTS({ evaluator: new UniformEvaluator() });
    const first = await mcts.run(game, 8);
    const nextGame = game.clone();
    nextGame.applyAction(first.action);
    const nextGame2 = nextGame.clone();
    nextGame2.applyAction(nextGame.legalIndices()[0]!);

    const second = await mcts.run(nextGame2, 8, { reuseRoot: first.nextRoot });
    expect(nextGame2.legalIndices()).toContain(second.action);
  });

  it("does not reuse a root from a different board with the same player to move", async () => {
    const evaluator = new UniformEvaluator();
    const mcts = new MCTS({ evaluator });
    const firstState = new GameState(9);
    const first = await mcts.run(firstState, 1);
    expect(first.nextRoot).not.toBeNull();

    const otherState = new GameState(9);
    otherState.applyAction(first.action === 0 ? 1 : 0);
    expect(otherState.toPlay).toBe(first.nextRoot!.toPlay);

    evaluator.calls = 0;
    await mcts.run(otherState, 0, { reuseRoot: first.nextRoot });
    expect(evaluator.calls).toBe(1);
  });

  it("temperature=0 (default) chooses the most-visited action deterministically", async () => {
    const game = new GameState(7);
    const mcts = new MCTS({ evaluator: new UniformEvaluator() });
    const a = await mcts.run(game, 32, { random: () => 0.0 });
    const b = await mcts.run(game, 32, { random: () => 0.999 });
    expect(a.action).toBe(b.action);
  });

  it("temperature>0 can return non-argmax actions when sampling", async () => {
    const game = new GameState(7);
    const mcts = new MCTS({ evaluator: new UniformEvaluator() });
    const seen = new Set<number>();
    let i = 0;
    const rng = () => ((i++ * 0.137) % 1);
    for (let k = 0; k < 20; k++) {
      const r = await mcts.run(game, 16, { temperature: 2.0, random: rng });
      expect(game.legalIndices()).toContain(r.action);
      seen.add(r.action);
    }
    expect(seen.size).toBeGreaterThan(1);
  });
});
