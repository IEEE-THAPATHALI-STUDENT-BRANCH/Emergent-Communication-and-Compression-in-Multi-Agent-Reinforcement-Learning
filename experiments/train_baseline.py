"""Train independent Q-learning agents without communication.

This version uses the police–civilian bomb environment, where:
- One bomb spawns per episode.
- Only the police agent (id=0) can defuse the bomb.
- Civilians can see the bomb locally but cannot defuse it.

The baseline here cannot send messages; performance should be
compared with the communication-enabled counterpart in
`train_comm.py`.
"""

import argparse
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from env.police_bomb_env import GridWorldEnv, Action
from agents.q_agent import QAgent


def run_episode(env: GridWorldEnv, agents: dict, explore: bool):
    obs = env.reset()
    total_reward = defaultdict(float)
    done = False

    while not done:
        actions = {}
        action_indices = {}
        for aid, agent in agents.items():
            move_idx, _, joint_idx = agent.act(obs[aid], explore=explore)
            actions[aid] = Action(move_idx)
            action_indices[aid] = joint_idx

        next_obs, rewards, done, info = env.step(actions)
        for aid, agent in agents.items():
            agent.learn(obs[aid], action_indices[aid], rewards[aid], next_obs[aid], done)
            total_reward[aid] += rewards[aid]
        obs = next_obs

    return total_reward, info


def train(
    episodes: int = 2000,
    seed: int = 0,
    plot: bool = False,
    output_dir: str = "results/baseline",
):
    os.makedirs(output_dir, exist_ok=True)
    env = GridWorldEnv(communication_enabled=False)
    agents = {
        i: QAgent(agent_id=i, actions=tuple(Action), seed=seed + i)
        for i in range(env.n_agents)
    }

    episode_rewards = []
    success_rate = []
    for ep in range(1, episodes + 1):
        explore = ep < episodes // 2
        total_reward, info = run_episode(env, agents, explore)
        episode_rewards.append(sum(total_reward.values()) / env.n_agents)
        # success = bomb defused at least once in the episode
        success_rate.append(1.0 if info.get("bomb_defused", False) else 0.0)

        for agent in agents.values():
            agent.record_episode()

    # save model (Q-tables + env config)
    model = {
        "env_kwargs": {
            "width": env.width,
            "height": env.height,
            "n_agents": env.n_agents,
            "n_resources": env.n_resources,
            "vision_radius": env.vision_radius,
            "max_steps": env.max_steps,
            "communication_enabled": env.communication_enabled,
            "communication_vocab_size": env.communication_vocab_size,
            "communication_cost": env.communication_cost,
            "step_penalty": env.step_penalty,
            "collision_penalty": env.collision_penalty,
            "resource_reward": env.resource_reward,
        },
        "q_tables": {str(aid): agent.q for aid, agent in agents.items()},
    }
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    if plot:
        episodes_axis = np.arange(1, episodes + 1)
        window = max(1, episodes // 20)

        # Smoothed average reward
        rewards_arr = np.array(episode_rewards)
        if window > 1:
            kernel = np.ones(window) / window
            rewards_ma = np.convolve(rewards_arr, kernel, mode="valid")
            x_rewards = np.arange(window, episodes + 1)
            plt.plot(x_rewards, rewards_ma, label=f"avg reward (MA {window})")
        else:
            plt.plot(episodes_axis, rewards_arr, label="avg reward")
        plt.title("Baseline learning curve")
        plt.xlabel("episode")
        plt.ylabel("average reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learning_curve.png"))

        plt.clf()

        # Smoothed success rate
        success_arr = np.array(success_rate, dtype=float)
        if window > 1:
            success_ma = np.convolve(success_arr, kernel, mode="valid")
            x_success = np.arange(window, episodes + 1)
            plt.plot(x_success, success_ma, label=f"success (MA {window})")
        else:
            plt.plot(episodes_axis, success_arr, label="success")
        plt.xlabel("episode")
        plt.ylabel("success")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "success_rate.png"))

    # Simple textual summary for quick comparison
    last_window = min(100, len(success_rate))
    if last_window > 0:
        avg_succ = float(np.mean(success_rate[-last_window:]))
        avg_reward_tail = float(np.mean(episode_rewards[-last_window:]))
        print(
            f"[baseline] episodes={episodes}, "
            f"last_{last_window}_avg_success={avg_succ:.3f}, "
            f"last_{last_window}_avg_reward={avg_reward_tail:.3f}"
        )

    return {
        "episode_rewards": episode_rewards,
        "success_rate": success_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline Q-learning agents.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", type=str, default="results/baseline")
    args = parser.parse_args()

    train(episodes=args.episodes, seed=args.seed, plot=args.plot, output_dir=args.output)


if __name__ == "__main__":
    main()
