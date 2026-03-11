"""
Train baseline (no meaningful communication) and communication-enabled agents
and plot their learning curves on a shared figure.

Baseline: Listener receives a constant dummy message (no information) and
optimizes its policy over objects alone.

Communication: Full Speaker+Listener setup as in `train.py`.

Usage:
    python -m referential_game.train_compare --config referential_game/config.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .agents import ListenerAgent, ListenerConfig, SpeakerAgent, SpeakerConfig
from .communication import Vocabulary
from .environment import ChannelConfig, EmergentCommunicationEnv, WorldConfig
from .train import load_config


def build_env_and_agents(cfg: Dict, device: torch.device) -> Tuple[EmergentCommunicationEnv, SpeakerAgent, ListenerAgent]:
    """Same helper as in train.py, duplicated here to avoid circular imports."""
    world_cfg = WorldConfig(
        num_objects=int(cfg["world"]["num_objects"]),
        colors=tuple(cfg["world"]["colors"]),
        shapes=tuple(cfg["world"]["shapes"]),
        sizes=tuple(cfg["world"]["sizes"]),
    )
    channel_cfg = ChannelConfig(
        vocabulary_size=int(cfg["channel"]["vocabulary_size"]),
        message_length=int(cfg["channel"]["message_length"]),
    )

    env = EmergentCommunicationEnv(world_cfg, channel_cfg, device=device)
    vocab = Vocabulary(size=channel_cfg.vocabulary_size)

    speaker_cfg = SpeakerConfig(hidden_dim=int(cfg["agents"]["speaker_hidden_dim"]))
    listener_cfg = ListenerConfig(hidden_dim=int(cfg["agents"]["listener_hidden_dim"]))

    speaker = SpeakerAgent(
        vocab=vocab,
        message_length=channel_cfg.message_length,
        world_color_dim=world_cfg.color_dim,
        world_shape_dim=world_cfg.shape_dim,
        world_size_dim=world_cfg.size_dim,
        config=speaker_cfg,
    ).to(device)

    listener = ListenerAgent(
        vocab=vocab,
        message_length=channel_cfg.message_length,
        world_color_dim=world_cfg.color_dim,
        world_shape_dim=world_cfg.shape_dim,
        world_size_dim=world_cfg.size_dim,
        config=listener_cfg,
    ).to(device)

    return env, speaker, listener


def run_training(
    cfg: Dict,
    use_communication: bool,
    device: torch.device,
) -> List[float]:
    """Run training and return list of average rewards per batch."""
    env, speaker, listener = build_env_and_agents(cfg, device)

    episodes = int(cfg["training"]["episodes"])
    batch_size = int(cfg["training"]["batch_size"])
    entropy_coeff = float(cfg["training"]["entropy_coeff"])

    speaker_optim = optim.Adam(speaker.parameters(), lr=float(cfg["training"]["speaker_lr"]))
    listener_optim = optim.Adam(listener.parameters(), lr=float(cfg["training"]["listener_lr"]))

    all_rewards: List[float] = []
    L = env.channel.message_length

    for _ in range(1, episodes + 1):
        log_probs_s = []
        log_probs_l = []
        entropies_s = []
        entropies_l = []
        rewards = []

        for _ in range(batch_size):
            obs = env.reset()
            target = obs["speaker_obs"]
            objects = obs["listener_obs"]

            if use_communication:
                # Speaker acts and sends a learned message
                msg_logits, _ = speaker.forward(target.unsqueeze(0))  # (1, L, V)
                msg_logits = msg_logits.squeeze(0)
                msg_dist = torch.distributions.Categorical(logits=msg_logits)
                message = msg_dist.sample()
                log_prob_msg = msg_dist.log_prob(message).sum()
                entropy_msg = msg_dist.entropy().sum()
            else:
                # Baseline: fixed dummy message (no information)
                message = torch.zeros(L, dtype=torch.long, device=device)
                log_prob_msg = torch.zeros((), device=device)
                entropy_msg = torch.zeros((), device=device)

            scores = listener.forward(objects, message)
            act_dist = torch.distributions.Categorical(logits=scores)
            action = act_dist.sample()
            log_prob_act = act_dist.log_prob(action)
            entropy_act = act_dist.entropy()

            _, reward, _, _ = env.step(message, int(action.item()))

            log_probs_s.append(log_prob_msg)
            log_probs_l.append(log_prob_act)
            entropies_s.append(entropy_msg)
            entropies_l.append(entropy_act)
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))

        log_probs_s_t = torch.stack(log_probs_s)
        log_probs_l_t = torch.stack(log_probs_l)
        rewards_t = torch.stack(rewards)
        entropies_s_t = torch.stack(entropies_s)
        entropies_l_t = torch.stack(entropies_l)

        baseline = rewards_t.mean()
        advantages = rewards_t - baseline

        # Speaker loss only meaningful when communication is enabled
        speaker_loss = -(log_probs_s_t * advantages.detach()).mean()
        listener_loss = -(log_probs_l_t * advantages.detach()).mean()

        if use_communication:
            speaker_loss = speaker_loss - entropy_coeff * entropies_s_t.mean()
        else:
            # No learning signal for the dummy speaker
            speaker_loss = torch.zeros_like(listener_loss)

        listener_loss = listener_loss - entropy_coeff * entropies_l_t.mean()

        speaker_optim.zero_grad()
        listener_optim.zero_grad()
        (speaker_loss + listener_loss).backward()
        nn.utils.clip_grad_norm_(listener.parameters(), max_norm=5.0)
        if use_communication:
            nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=5.0)
        listener_optim.step()
        if use_communication:
            speaker_optim.step()

        avg_reward = float(rewards_t.mean().item())
        all_rewards.append(avg_reward)

    return all_rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs communication learning curves.")
    parser.add_argument(
        "--config",
        type=str,
        default="referential_game/config.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["training"]["seed"]))

    print("Training baseline (no communication)...")
    rewards_base = run_training(cfg, use_communication=False, device=device)
    print("Training with communication...")
    rewards_comm = run_training(cfg, use_communication=True, device=device)

    episodes = len(rewards_base)
    assert len(rewards_comm) == episodes

    xs = np.arange(1, episodes + 1)
    window = max(1, episodes // 50)
    kernel = np.ones(window) / window

    def smooth(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if window == 1:
            return xs, x
        ma = np.convolve(x, kernel, mode="valid")
        return np.arange(window, episodes + 1), ma

    rb = np.array(rewards_base, dtype=np.float32)
    rc = np.array(rewards_comm, dtype=np.float32)
    xb, rb_s = smooth(rb)
    xc, rc_s = smooth(rc)

    results_dir = Path("referential_game/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(xb, rb_s, label="baseline (no comm)")
    plt.plot(xc, rc_s, label="with communication")
    plt.xlabel("training batch (episode index)")
    plt.ylabel("average reward / success rate")
    plt.title("Baseline vs communication learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "learning_curve_baseline_vs_comm.png")
    plt.close()

    print(f"Saved comparison plot to {results_dir / 'learning_curve_baseline_vs_comm.png'}")


if __name__ == "__main__":
    main()

