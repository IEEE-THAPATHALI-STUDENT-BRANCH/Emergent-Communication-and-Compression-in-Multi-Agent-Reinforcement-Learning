"""
Sweep over communication channel configurations (vocabulary size, message length)
and measure performance.

For each (vocab_size, message_length) pair, this script trains a communication-
enabled Speaker+Listener using the same policy-gradient objective as
`referential_game.train`, then records the average reward. Results are visualized
as a heatmap showing how performance varies with channel capacity.

Usage example:

    python -m referential_game.train_sweep \\
        --config referential_game/config.json \\
        --vocab_sizes 4 8 16 \\
        --message_lengths 1 2 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .agents import ListenerAgent, ListenerConfig, SpeakerAgent, SpeakerConfig
from .communication import Vocabulary
from .environment import ChannelConfig, EmergentCommunicationEnv, WorldConfig
from .train import load_config


def build_env_and_agents_for_channel(
    cfg: Dict,
    device: torch.device,
    vocab_size: int,
    message_length: int,
) -> Tuple[EmergentCommunicationEnv, SpeakerAgent, ListenerAgent]:
    """Construct env + agents but overriding channel configuration."""
    world_cfg = WorldConfig(
        num_objects=int(cfg["world"]["num_objects"]),
        colors=tuple(cfg["world"]["colors"]),
        shapes=tuple(cfg["world"]["shapes"]),
        sizes=tuple(cfg["world"]["sizes"]),
    )
    channel_cfg = ChannelConfig(
        vocabulary_size=int(vocab_size),
        message_length=int(message_length),
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


def run_training_single(
    cfg: Dict,
    device: torch.device,
    vocab_size: int,
    message_length: int,
) -> List[float]:
    """Train Speaker+Listener for a single channel configuration."""
    env, speaker, listener = build_env_and_agents_for_channel(cfg, device, vocab_size, message_length)

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

            # Speaker acts
            msg_logits, _ = speaker.forward(target.unsqueeze(0))  # (1, L, V)
            msg_logits = msg_logits.squeeze(0)
            msg_dist = torch.distributions.Categorical(logits=msg_logits)
            message = msg_dist.sample()
            log_prob_msg = msg_dist.log_prob(message).sum()
            entropy_msg = msg_dist.entropy().sum()

            # Listener acts
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

        speaker_loss = -(log_probs_s_t * advantages.detach()).mean() - entropy_coeff * entropies_s_t.mean()
        listener_loss = -(log_probs_l_t * advantages.detach()).mean() - entropy_coeff * entropies_l_t.mean()

        speaker_optim.zero_grad()
        listener_optim.zero_grad()
        (speaker_loss + listener_loss).backward()
        torch.nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(listener.parameters(), max_norm=5.0)
        speaker_optim.step()
        listener_optim.step()

        avg_reward = float(rewards_t.mean().item())
        all_rewards.append(avg_reward)

    return all_rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep vocab size and message length.")
    parser.add_argument(
        "--config",
        type=str,
        default="referential_game/config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--vocab_sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="List of vocabulary sizes to test.",
    )
    parser.add_argument(
        "--message_lengths",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="List of message lengths to test.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["training"]["seed"]))

    vocab_sizes = sorted(set(args.vocab_sizes))
    msg_lengths = sorted(set(args.message_lengths))

    results: Dict[Tuple[int, int], float] = {}

    for V in vocab_sizes:
        for L in msg_lengths:
            print(f"Training with vocab_size={V}, message_length={L}...")
            rewards = run_training_single(cfg, device, V, L)
            # Use mean reward over the last 20% of batches as performance
            tail = max(1, len(rewards) // 5)
            score = float(np.mean(rewards[-tail:]))
            results[(V, L)] = score
            print(f"  -> mean tail reward: {score:.3f}")

    # Build heatmap matrix: rows = message_lengths, cols = vocab_sizes
    mat = np.zeros((len(msg_lengths), len(vocab_sizes)), dtype=np.float32)
    for i, L in enumerate(msg_lengths):
        for j, V in enumerate(vocab_sizes):
            mat[i, j] = results[(V, L)]

    results_dir = Path("referential_game/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    im = plt.imshow(mat, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="mean success rate (tail)")
    plt.xticks(np.arange(len(vocab_sizes)), vocab_sizes)
    plt.yticks(np.arange(len(msg_lengths)), msg_lengths)
    plt.xlabel("Vocabulary size")
    plt.ylabel("Message length")
    plt.title("Performance vs. communication channel capacity")
    plt.tight_layout()
    out_path = results_dir / "performance_vs_channel_heatmap.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved sweep heatmap to {out_path}")


if __name__ == "__main__":
    main()

