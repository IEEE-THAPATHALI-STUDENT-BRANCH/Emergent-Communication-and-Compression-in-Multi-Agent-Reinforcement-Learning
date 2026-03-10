"""Train independent Q-learning agents with communication."""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from env.grid_world import GridWorldEnv, Action
from agents.q_agent import QAgent


def run_episode(env: GridWorldEnv, agents: dict, explore: bool):
    obs = env.reset()
    total_reward = defaultdict(float)
    done = False

    while not done:
        actions = {}
        messages = {}
        action_indices = {}
        for aid, agent in agents.items():
            move_idx, msg, joint_idx = agent.act(obs[aid], explore=explore)
            actions[aid] = Action(move_idx)
            messages[aid] = msg if msg is not None else 0
            action_indices[aid] = joint_idx

        next_obs, rewards, done, info = env.step(actions, messages)
        for aid, agent in agents.items():
            agent.learn(obs[aid], action_indices[aid], rewards[aid], next_obs[aid], done)
            total_reward[aid] += rewards[aid]
        obs = next_obs

    return total_reward, info


def train(
    episodes: int = 500,
    seed: int = 0,
    plot: bool = False,
    output_dir: str = "results/comm",
):
    os.makedirs(output_dir, exist_ok=True)
    env = GridWorldEnv(communication_enabled=True)
    agents = {
        i: QAgent(
            agent_id=i,
            actions=tuple(Action),
            seed=seed + i,
            comm_vocab_size=env.communication_vocab_size,
        )
        for i in range(env.n_agents)
    }

    episode_rewards = []
    success_rate = []
    for ep in range(1, episodes + 1):
        explore = ep < episodes // 2
        total_reward, info = run_episode(env, agents, explore)
        episode_rewards.append(sum(total_reward.values()) / env.n_agents)
        success_rate.append(1.0 if len(info["collected"]) > 0 else 0.0)

    if plot:
        plt.plot(episode_rewards, label="avg reward")
        plt.title("Communication learning curve")
        plt.xlabel("episode")
        plt.ylabel("average reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learning_curve.png"))

        plt.clf()
        plt.plot(success_rate, label="success")
        plt.xlabel("episode")
        plt.ylabel("success")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "success_rate.png"))

    return {
        "episode_rewards": episode_rewards,
        "success_rate": success_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Q-learning agents with communication.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", type=str, default="results/comm")
    args = parser.parse_args()

    train(episodes=args.episodes, seed=args.seed, plot=args.plot, output_dir=args.output)


if __name__ == "__main__":
    main()
