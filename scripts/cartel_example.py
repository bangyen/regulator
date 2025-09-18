#!/usr/bin/env python3
"""
Example usage of the CartelEnv environment.

This script demonstrates how to use the CartelEnv for simulating
oligopolistic price competition.
"""

from src.cartel.cartel_env import CartelEnv


def main() -> None:
    """Run a simple example of the CartelEnv."""
    print("CartelEnv Example")
    print("================")

    # Create environment
    env = CartelEnv(n_firms=3, max_steps=5, seed=42)

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    print()

    # Run a few steps
    for step in range(5):
        # Random action (prices between 1 and 100)
        action = env.action_space.sample()

        obs, rewards, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}:")
        print(f"  Action (prices): {action}")
        print(f"  Rewards (profits): {rewards}")
        print(f"  Market price: {info['market_price']:.2f}")
        print(f"  Total demand: {info['total_demand']:.2f}")
        print(f"  Demand shock: {info['demand_shock']:.2f}")
        print(f"  Total profits so far: {info['total_profits']}")
        print()

        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break


if __name__ == "__main__":
    main()
