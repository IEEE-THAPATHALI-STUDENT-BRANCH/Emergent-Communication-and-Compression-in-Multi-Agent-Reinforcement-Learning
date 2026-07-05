"""Compare baseline vs communication learning curves on the same normalized scale.

This script runs both training routines for a given number of episodes and
plots:
- average reward (baseline vs comm) on a single figure
- success rate (baseline vs comm) on a single figure

so you can visually compare them with the same x/y axes.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.train_baseline import train as train_baseline
from experiments.train_comm import train as train_comm


def moving_average(x, window: int):
    if window <= 1:
        return x, np.arange(1, len(x) + 1)
    kernel = np.ones(window) / window
    ma = np.convolve(x, kernel, mode="valid")
    xs = np.arange(window, len(x) + 1)
    return ma, xs


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs communication learning curves.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window_frac", type=float, default=0.05, help="Moving-average window as fraction of episodes.")
    args = parser.parse_args()

    episodes = args.episodes
    window = max(1, int(episodes * args.window_frac))

    # Run both trainings without their own plotting
    baseline = train_baseline(episodes=episodes, seed=args.seed, plot=False)
    comm = train_comm(episodes=episodes, seed=args.seed, plot=False)

    base_rewards = np.array(baseline["episode_rewards"], dtype=float)
    comm_rewards = np.array(comm["episode_rewards"], dtype=float)
    base_success = np.array(baseline["success_rate"], dtype=float)
    comm_success = np.array(comm["success_rate"], dtype=float)

    # Shared normalization for y-axis
    all_rewards = np.concatenate([base_rewards, comm_rewards])
    all_success = np.concatenate([base_success, comm_success])

    # Reward plot
    plt.figure(figsize=(8, 5))
    br_ma, br_x = moving_average(base_rewards, window)
    cr_ma, cr_x = moving_average(comm_rewards, window)
    plt.plot(br_x, br_ma, label=f"baseline (MA {window})", linestyle="--")
    plt.plot(cr_x, cr_ma, label=f"comm (MA {window})")
    plt.xlabel("episode")
    plt.ylabel("average reward")
    plt.title("Baseline vs Communication: average reward")
    plt.ylim(all_rewards.min(), all_rewards.max())
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/combined_learning_curve.png")

    # Success plot
    plt.clf()
    bs_ma, bs_x = moving_average(base_success, window)
    cs_ma, cs_x = moving_average(comm_success, window)
    plt.plot(bs_x, bs_ma, label=f"baseline (MA {window})", linestyle="--")
    plt.plot(cs_x, cs_ma, label=f"comm (MA {window})")
    plt.xlabel("episode")
    plt.ylabel("success")
    plt.title("Baseline vs Communication: success rate")
    # Success is already between 0 and 1, but use combined bounds for consistency
    plt.ylim(all_success.min(), all_success.max())
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/combined_success_rate.png")


if __name__ == "__main__":
    main()

