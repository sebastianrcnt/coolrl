from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

import inquirer
from loguru import logger
from tqdm import tqdm

# Card labels used in Kuhn poker, and a simple rank map for winner checks.
CARDS = ("J", "Q", "K")
CARD_RANK = {"J": 0, "Q": 1, "K": 2}

# `p` means pass/check or fold depending on context.
# `b` means bet/call depending on context.
ACTIONS = ("p", "b")  # context dependent: pass/check/fold OR bet/call

# Terminal histories for this tiny game using only p/b tokens:
# pp: both checked
# pbp: P1 check, P2 bet, P1 fold
# pbb: P1 check, P2 bet, P1 call
# bp: P1 bet, P2 fold
# bb: P1 bet, P2 call
TERMINAL_HISTORIES = ("pp", "pbp", "pbb", "bp", "bb")


@dataclass
class KuhnTrainer:
    regret_sum: DefaultDict[str, List[float]]
    strategy_sum: DefaultDict[str, List[float]]

    def __init__(self) -> None:
        self.regret_sum = defaultdict(lambda: [0.0, 0.0])
        self.strategy_sum = defaultdict(lambda: [0.0, 0.0])

    def is_terminal(self, history: str) -> bool:
        # A hand is finished when one of the known ending action sequences is reached.
        return history in TERMINAL_HISTORIES

    def terminal_utility_p1(self, history: str, p1_card: str, p2_card: str) -> float:
        # Payoff from Player 1 perspective.
        # Positive means Player 1 wins chips.
        # antes are implicit. Utility is net chips for Player 1.
        if history == "pbp":
            return -1.0
        if history == "bp":
            return 1.0

        # showdown cases: winner gets net +1 for pp, net +2 for called bets.
        if CARD_RANK[p1_card] > CARD_RANK[p2_card]:
            return 1.0 if history == "pp" else 2.0
        return -1.0 if history == "pp" else -2.0

    def current_player(self, history: str) -> int:
        # Decide whose turn it is from the action history.
        # Player 0 starts first, then turns alternate based on valid histories.
        # Player 1 acts first, then depending on history.
        if history == "":
            return 0
        if history == "p":
            return 1
        if history == "b":
            return 1
        if history == "pb":
            return 0
        raise ValueError(f"No current player for terminal history {history!r}")

    def infoset_key(self, player: int, card: str, history: str) -> str:
        # Infoset key = what this player knows when choosing an action:
        # their private card and the action history so far.
        return f"{card}:{history}"

    def get_strategy(self, infoset: str) -> List[float]:
        # Convert positive regret values to a probability distribution over actions.
        # If all regrets are zero, default to 50/50.
        regrets = self.regret_sum[infoset]
        positive = [max(r, 0.0) for r in regrets]
        normalizer = sum(positive)
        if normalizer > 0:
            return [r / normalizer for r in positive]
        return [0.5, 0.5]

    def cfr(
        self, history: str, p1_card: str, p2_card: str, reach_p1: float, reach_p2: float
    ) -> float:
        # Counterfactual Regret Minimization (CFR) core recursion:
        # 1) Evaluate both actions recursively.
        # 2) Compute node value from current mixed strategy.
        # 3) Accumulate regret and average strategy with reach probabilities.
        if self.is_terminal(history):
            return self.terminal_utility_p1(history, p1_card, p2_card)

        player = self.current_player(history)
        my_card = p1_card if player == 0 else p2_card
        infoset = self.infoset_key(player, my_card, history)
        strategy = self.get_strategy(infoset)

        action_util = [0.0, 0.0]
        node_util = 0.0

        for action_index, action in enumerate(ACTIONS):
            next_history = history + action
            if player == 0:
                action_util[action_index] = self.cfr(
                    next_history,
                    p1_card,
                    p2_card,
                    reach_p1 * strategy[action_index],
                    reach_p2,
                )
            else:
                # recursion returns utility for Player 1; negate for Player 2 perspective
                action_util[action_index] = -self.cfr(
                    next_history,
                    p1_card,
                    p2_card,
                    reach_p1,
                    reach_p2 * strategy[action_index],
                )
            node_util += strategy[action_index] * action_util[action_index]

        # accumulate average strategy weighted by own reach
        my_reach = reach_p1 if player == 0 else reach_p2
        for i in range(2):
            self.strategy_sum[infoset][i] += my_reach * strategy[i]

        # regret update weighted by opponent reach
        opp_reach = reach_p2 if player == 0 else reach_p1
        for i in range(2):
            regret = action_util[i] - node_util
            self.regret_sum[infoset][i] += opp_reach * regret

        return node_util if player == 0 else -node_util

    def train(self, iterations: int = 100_000, shuffle_deals: bool = True) -> float:
        # Repeatedly run all possible deals to let regrets and average strategies stabilize.
        cards = list(CARDS)
        util = 0.0
        deals = [(c1, c2) for c1 in cards for c2 in cards if c1 != c2]
        logger.info(
            "Starting CFR training: iterations={}, deals_per_iteration={}, shuffle_deals={}",
            iterations,
            len(deals),
            shuffle_deals,
        )
        for _ in tqdm(range(iterations), desc="Training CFR", unit="iter"):
            if shuffle_deals:
                random.shuffle(deals)
            for p1_card, p2_card in deals:
                util += self.cfr("", p1_card, p2_card, 1.0, 1.0)
        avg_utility = util / (iterations * len(deals))
        logger.success("Finished CFR training: avg_utility_p1={:.6f}", avg_utility)
        return avg_utility

    def average_strategy(self) -> Dict[str, List[float]]:
        # Final policy returned here is the normalized average of visited action probabilities.
        avg: Dict[str, List[float]] = {}
        for infoset, totals in self.strategy_sum.items():
            s = sum(totals)
            if s > 0:
                avg[infoset] = [totals[0] / s, totals[1] / s]
            else:
                avg[infoset] = [0.5, 0.5]
        return avg


def pretty_action(history: str, action: str) -> str:
    # Convert compact internal tokens to human-readable words for display.
    # Translate action token based on context.
    if history in ("", "p"):
        return "check" if action == "p" else "bet"
    if history in ("b", "pb"):
        return "fold" if action == "p" else "call"
    raise ValueError(f"Unsupported history {history!r}")


def sample_action(probs: List[float]) -> str:
    # Draw one action by chance from the strategy probabilities.
    return random.choices(ACTIONS, weights=probs, k=1)[0]


def choose_human_action(history: str) -> str:
    # Interactive menu for the human player's action.
    # The same internal token is returned (`p` or `b`) after UI selection.
    if history in ("", "p"):
        choices = ["check", "bet"]
    else:
        choices = ["fold", "call"]

    answer = inquirer.prompt(
        [inquirer.List("action", message="Your action", choices=choices)]
    )
    if not answer:
        raise KeyboardInterrupt
    action = answer["action"]
    return "p" if action in ("check", "fold") else "b"


def choose_human_turn() -> bool:
    # Ask once per hand whether the human plays first.
    answer = inquirer.prompt(
        [
            inquirer.List(
                "human_is_p1",
                message="Do you want to act first?",
                choices=["yes", "no"],
            )
        ]
    )
    if not answer:
        raise KeyboardInterrupt
    return answer["human_is_p1"] == "yes"


def choose_main_menu() -> str:
    # Top-level flow control after training: either play once or quit.
    answer = inquirer.prompt(
        [
            inquirer.List(
                "choice",
                message="Choose",
                choices=[
                    ("Play vs bot", "p"),
                    ("Quit", "q"),
                ],
            )
        ]
    )
    if not answer:
        raise KeyboardInterrupt
    return answer["choice"]


class TqdmLogSink:
    # Keep loguru logs from breaking the tqdm bar line layout.
    def write(self, message: str) -> None:
        if message.rstrip():
            tqdm.write(message, end="")

    def flush(self) -> None:
        sys.stderr.flush()


def configure_logging() -> None:
    # Replace default logger output with cleaner time/level logs for terminal and progress bar.
    logger.remove()
    logger.add(
        TqdmLogSink(),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )


def display_strategy(avg_strategy: Dict[str, List[float]]) -> None:
    # Show the learned action probabilities per infoset in a fixed order.
    order = [
        "J:",
        "Q:",
        "K:",
        "J:pb",
        "Q:pb",
        "K:pb",
        "J:p",
        "Q:p",
        "K:p",
        "J:b",
        "Q:b",
        "K:b",
    ]
    print("\nAverage strategy (p_prob, b_prob):")
    for key in order:
        probs = avg_strategy.get(key, [0.5, 0.5])
        print(f"  {key:<4} -> p={probs[0]:.3f}, b={probs[1]:.3f}")


def resolve_terminal(
    history: str, human_card: str, bot_card: str, human_is_p1: bool
) -> Tuple[float, str]:
    # Convert game history into who won from the human perspective (you).
    trainer = KuhnTrainer()
    if human_is_p1:
        util_p1 = trainer.terminal_utility_p1(history, human_card, bot_card)
        winner = "You" if util_p1 > 0 else "Bot"
        return util_p1, winner
    util_p1 = trainer.terminal_utility_p1(history, bot_card, human_card)
    util_human = -util_p1
    winner = "You" if util_human > 0 else "Bot"
    return util_human, winner


def play_human_vs_bot(avg_strategy: Dict[str, List[float]]) -> None:
    # Interactive one-hand play using the learned average strategy for the bot.
    print("\n=== Play Kuhn Poker vs CFR bot ===")
    print("Tokens: p/b internally. Use the inquirer menu to choose your action.")
    print(
        "When no bet exists: type 'check' or 'bet'. After a bet: type 'fold' or 'call'."
    )

    human_is_p1 = choose_human_turn()
    cards = list(CARDS)
    random.shuffle(cards)
    p1_card, p2_card = cards[0], cards[1]
    human_card = p1_card if human_is_p1 else p2_card
    bot_card = p2_card if human_is_p1 else p1_card
    print(f"Your card: {human_card}")

    history = ""
    while history not in TERMINAL_HISTORIES:
        player = 0 if history in ("", "pb") else 1
        human_turn = (player == 0 and human_is_p1) or (player == 1 and not human_is_p1)

        if human_turn:
            action = choose_human_action(history)
            print(f"You chose: {pretty_action(history, action)}")
        else:
            infoset = f"{bot_card}:{history}"
            probs = avg_strategy.get(infoset, [0.5, 0.5])
            action = sample_action(probs)
            print(f"Bot chose: {pretty_action(history, action)}")

        history += action

    util_human, winner = resolve_terminal(history, human_card, bot_card, human_is_p1)
    print(f"\nFinal history: {history}")
    if history in ("pp", "pbb", "bb"):
        print(f"Showdown -> You: {human_card}, Bot: {bot_card}")
    else:
        print("No showdown (someone folded).")
    print(f"Winner: {winner}")
    print(f"Your payoff this hand: {util_human:+.0f}")


def expected_value_from_avg_strategy(avg_strategy: Dict[str, List[float]]) -> float:
    # Evaluate the expected value of the final average strategy under random deals.
    trainer = KuhnTrainer()

    def recurse(history: str, p1_card: str, p2_card: str) -> float:
        if trainer.is_terminal(history):
            return trainer.terminal_utility_p1(history, p1_card, p2_card)
        player = trainer.current_player(history)
        card = p1_card if player == 0 else p2_card
        infoset = f"{card}:{history}"
        strategy = avg_strategy.get(infoset, [0.5, 0.5])
        value = 0.0
        for i, action in enumerate(ACTIONS):
            value += strategy[i] * recurse(history + action, p1_card, p2_card)
        return value

    deals = [(c1, c2) for c1 in CARDS for c2 in CARDS if c1 != c2]
    return sum(recurse("", c1, c2) for c1, c2 in deals) / len(deals)


def best_response_value(
    strategy: Dict[str, List[float]], br_player: int, trainer: KuhnTrainer
) -> float:
    # Exact value when one player chooses the best action and the opponent stays fixed.
    if br_player not in (0, 1):
        raise ValueError(f"Unsupported best-response player: {br_player}")

    br_histories = ("", "pb") if br_player == 0 else ("p", "b")
    br_infosets = [
        trainer.infoset_key(br_player, card, history)
        for card in CARDS
        for history in br_histories
    ]

    def recurse(
        history: str, p1_card: str, p2_card: str, br_policy: Dict[str, str]
    ) -> float:
        if trainer.is_terminal(history):
            utility_p1 = trainer.terminal_utility_p1(history, p1_card, p2_card)
            return utility_p1 if br_player == 0 else -utility_p1

        player = trainer.current_player(history)
        card = p1_card if player == 0 else p2_card
        infoset = trainer.infoset_key(player, card, history)

        if player == br_player:
            return recurse(history + br_policy[infoset], p1_card, p2_card, br_policy)

        fixed_strategy = strategy.get(infoset, [0.5, 0.5])
        return sum(
            fixed_strategy[i] * recurse(history + action, p1_card, p2_card, br_policy)
            for i, action in enumerate(ACTIONS)
        )

    deals = [(c1, c2) for c1 in CARDS for c2 in CARDS if c1 != c2]
    best_value = float("-inf")
    for policy_bits in range(2 ** len(br_infosets)):
        br_policy = {
            infoset: ACTIONS[(policy_bits >> i) & 1]
            for i, infoset in enumerate(br_infosets)
        }
        value = sum(
            recurse("", p1_card, p2_card, br_policy) for p1_card, p2_card in deals
        ) / len(deals)
        best_value = max(best_value, value)
    return best_value


def exploitability(strategy: Dict[str, List[float]], trainer: KuhnTrainer) -> float:
    # Average gain available to both players by best responding to the fixed profile.
    br_value_p1 = best_response_value(strategy, 0, trainer)
    br_value_p2 = best_response_value(strategy, 1, trainer)
    return (br_value_p1 + br_value_p2) / 2.0


def main() -> None:
    # Training + optional human-vs-bot gameplay loop.
    configure_logging()
    iterations = int(input("Training iterations [100000]: ") or "100000")
    logger.info("Preparing Kuhn Poker CFR trainer")
    trainer = KuhnTrainer()
    avg_train_value = trainer.train(iterations=iterations)
    avg_strategy = trainer.average_strategy()
    logger.info("Computed average strategy over {} information sets", len(avg_strategy))

    print(
        f"\nEstimated self-play utility during training for P1: {avg_train_value:.6f}"
    )
    exact_ev = expected_value_from_avg_strategy(avg_strategy)
    logger.info("Computed exact EV for average strategy profile: {:.6f}", exact_ev)
    print(f"Expected value of average strategy profile for P1: {exact_ev:.6f}")
    avg_exploitability = exploitability(avg_strategy, trainer)
    logger.info(
        "Computed exploitability for average strategy: {:.6f}", avg_exploitability
    )
    if avg_exploitability > 0.01:
        logger.warning(
            "Exploitability is above 0.01; the strategy may not have converged."
        )
    print(f"Exploitability of average strategy: {avg_exploitability:.6f}")
    print("Known equilibrium target is about -0.055556 for P1.")
    display_strategy(avg_strategy)

    while True:
        choice = choose_main_menu()
        if choice == "q":
            break
        if choice == "p":
            play_human_vs_bot(avg_strategy)


if __name__ == "__main__":
    main()
