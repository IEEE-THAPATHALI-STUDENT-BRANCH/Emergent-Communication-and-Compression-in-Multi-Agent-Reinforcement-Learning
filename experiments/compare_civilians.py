"""Train and compare different numbers of civilians, with and without communication.

This script sweeps over a list of `n_civilians` values, trains both the baseline
and communication-enabled agents for each setting, and plots success rates on
the same figure for easier comparison.
"""

from __future__ import annotations

import argparse
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.police_bomb_env import GridWorldEnv, Action
from agents.q_agent import QAgent


def run_single(
    n_civilians: int,
    episodes: int,
    seed: int,
    communication_enabled: bool,
) -> List[float]:
    env = GridWorldEnv(communication_enabled=communication_enabled, n_civilians=n_civilians)
    agents = {
        i: QAgent(
            agent_id=i,
            actions=tuple(Action),
            seed=seed + i,
            comm_vocab_size=env.communication_vocab_size if communication_enabled else None,
        )
        for i in range(env.n_agents)
    }

    success_rate: List[float] = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False

        while not done:
            actions = {}
            messages = {} if communication_enabled else None
            action_indices = {}

            for aid, agent in agents.items():
                move_idx, msg, joint_idx = agent.act(obs[aid], explore=ep < episodes // 2)
                actions[aid] = Action(move_idx)
                action_indices[aid] = joint_idx
                if communication_enabled and messages is not None:
                    messages[aid] = msg if msg is not None else 0

            next_obs, rewards, done, info = env.step(actions, messages)
            for aid, agent in agents.items():
                agent.learn(obs[aid], action_indices[aid], rewards[aid], next_obs[aid], done)
            obs = next_obs

        success_rate.append(1.0 if info.get("bomb_defused", False) else 0.0)
        for agent in agents.values():
            agent.record_episode()

    return success_rate


def main():
    parser = argparse.ArgumentParser(description="Compare different numbers of civilians.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--civilians",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="List of n_civilians values to compare, e.g. --civilians 1 2 3",
    )
    args = parser.parse_args()

    civilians_list: List[int] = args.civilians
    episodes = args.episodes

    plt.figure(figsize=(8, 5))

    for n_civ in civilians_list:
        # Baseline (no communication)
        base_success = run_single(
            n_civilians=n_civ,
            episodes=episodes,
            seed=args.seed,
            communication_enabled=False,
        )
        base_success_ma = np.convolve(
            np.array(base_success, dtype=float),
            np.ones(max(1, episodes // 20)) / max(1, episodes // 20),
            mode="valid",
        )
        x_base = np.arange(len(base_success_ma)) + 1
        plt.plot(x_base, base_success_ma, linestyle="--", label=f"no-comm, civ={n_civ}")

        # With communication
        comm_success = run_single(
            n_civilians=n_civ,
            episodes=episodes,
            seed=args.seed + 1000,
            communication_enabled=True,
        )
        comm_success_ma = np.convolve(
            np.array(comm_success, dtype=float),
            np.ones(max(1, episodes // 20)) / max(1, episodes // 20),
            mode="valid",
        )
        x_comm = np.arange(len(comm_success_ma)) + 1
        plt.plot(x_comm, comm_success_ma, label=f"comm, civ={n_civ}")

    plt.xlabel("episode")
    plt.ylabel("success (moving average)")
    plt.title("Success vs. episodes for different numbers of civilians")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/compare_civilians_success.png")


if __name__ == "__main__":
    main()

