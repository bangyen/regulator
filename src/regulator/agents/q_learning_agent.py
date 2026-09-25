"""
Tabular Q-learning pricing agent.

Follows the setup of Calvano, Calzolari, Denicolò and Pastorello (2020),
"Artificial Intelligence, Algorithmic Pricing, and Collusion" (AER): each firm
picks from a discrete price grid, the state is last period's grid prices of
all firms (one-period memory), exploration is epsilon-greedy with
exponentially decaying epsilon, and nothing in the algorithm tells firms to
coordinate. Patient learners (discount factor near 1) tend to settle above
the one-shot Nash price and punish deviations; myopic ones (discount 0)
cannot learn that and stay close to competitive prices.
"""

from typing import Any

import numpy as np

from regulator.agents.firm_agents import BaseAgent

# One-shot Nash and joint-profit-maximizing prices of the default CartelEnv
# (linear demand 100 - p, marginal cost 10, logit shares, two firms).
DEFAULT_NASH_PRICE = 40.0
DEFAULT_MONOPOLY_PRICE = 55.0


def default_price_grid(
    nash_price: float = DEFAULT_NASH_PRICE,
    monopoly_price: float = DEFAULT_MONOPOLY_PRICE,
    n_prices: int = 15,
    margin: float = 0.1,
) -> np.ndarray:
    """Calvano et al.'s grid: evenly spaced, extending `margin` beyond N and M."""
    spread = monopoly_price - nash_price
    return np.linspace(
        nash_price - margin * spread, monopoly_price + margin * spread, n_prices
    )


class QLearningAgent(BaseAgent):
    """Epsilon-greedy tabular Q-learner over a discrete price grid."""

    def __init__(
        self,
        agent_id: int,
        n_firms: int = 2,
        price_grid: np.ndarray | None = None,
        learning_rate: float = 0.15,
        discount: float = 0.95,
        exploration_decay: float = 2e-5,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            agent_id: Unique identifier (also the firm's index in the market)
            n_firms: Number of firms, which fixes the state space size
            price_grid: Prices the agent can choose (default: default_price_grid)
            learning_rate: Q-update step size (alpha)
            discount: Discount factor (delta); 0 makes the agent myopic
            exploration_decay: beta in epsilon_t = exp(-beta * t)
            seed: Random seed for exploration and tie-breaking
        """
        super().__init__(agent_id, seed)
        self.n_firms = n_firms
        self.price_grid = (
            np.asarray(price_grid, dtype=float)
            if price_grid is not None
            else default_price_grid()
        )
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_decay = exploration_decay

        n_prices = len(self.price_grid)
        self.q_table = np.zeros((n_prices**n_firms, n_prices))
        self.t = 0  # learning steps taken; persists across episodes
        self.learning = True
        self.frozen_epsilon = 0.0

        self._state: int | None = None
        self._action: int | None = None
        self._reward: float | None = None

    @property
    def epsilon(self) -> float:
        """Current exploration probability (0 once learning is frozen)."""
        if not self.learning:
            return self.frozen_epsilon
        return float(np.exp(-self.exploration_decay * self.t))

    def _state_index(self, prices: np.ndarray) -> int:
        """Map observed prices to a state: nearest grid index per firm."""
        n_prices = len(self.price_grid)
        indices = np.abs(prices[:, None] - self.price_grid[None, :]).argmin(axis=1)
        return int(np.ravel_multi_index(tuple(indices), (n_prices,) * self.n_firms))

    def choose_price(
        self,
        observation: np.ndarray,
        env: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Learn from the last outcome, then pick the next price."""
        prices = None
        if info is not None and "prices" in info:
            prices = np.asarray(info["prices"], dtype=float)
        elif len(observation) >= self.n_firms:
            prices = np.asarray(observation[: self.n_firms], dtype=float)
        state = (
            self._state_index(prices)
            if prices is not None and len(prices) == self.n_firms
            else 0
        )

        # Q-update for the previous (state, action) now that we know where it led
        if (
            self.learning
            and self._state is not None
            and self._action is not None
            and self._reward is not None
        ):
            target = self._reward + self.discount * self.q_table[state].max()
            old = self.q_table[self._state, self._action]
            self.q_table[self._state, self._action] = old + self.learning_rate * (
                target - old
            )
            self.t += 1

        if self.np_random.random() < self.epsilon:
            action = int(self.np_random.integers(len(self.price_grid)))
        else:
            row = self.q_table[state]
            best = np.flatnonzero(row == row.max())
            action = int(self.np_random.choice(best))

        self._state, self._action, self._reward = state, action, None
        return float(self.price_grid[action])

    def observe_outcome(self, profit: float) -> None:
        """Store the profit from the price just chosen (used at the next step)."""
        self._reward = profit

    def freeze(self, epsilon: float = 0.0) -> None:
        """
        Stop learning and play the learned policy.

        Args:
            epsilon: Probability of a random price each step (0 = greedy)
        """
        self.learning = False
        self.frozen_epsilon = epsilon

    def greedy_price(self, prices: np.ndarray) -> float:
        """The learned policy's price in the state given by `prices`."""
        return float(self.price_grid[self.q_table[self._state_index(prices)].argmax()])

    def reset(self) -> None:
        """New episode: forget the last transition but keep what was learned."""
        super().reset()
        self._state = self._action = self._reward = None
