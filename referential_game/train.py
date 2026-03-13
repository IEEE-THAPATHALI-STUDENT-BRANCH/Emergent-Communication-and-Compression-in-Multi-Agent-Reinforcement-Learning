"""
Policy-gradient training loop for the referential game (with communication).

This script wires together:
  * EmergentCommunicationEnv
  * SpeakerAgent
  * ListenerAgent

and trains them jointly using REINFORCE.

For baseline vs communication comparisons and plotting, see
`referential_game/train_compare.py`.

Usage:
    python -m referential_game.train --config referential_game/config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .agents import ListenerAgent, ListenerConfig, SpeakerAgent, SpeakerConfig
from .communication import Vocabulary
from .environment import ChannelConfig, EmergentCommunicationEnv, WorldConfig


def load_config(path: str | Path) -> Dict:
    """Load a JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_env_and_agents(cfg: Dict, device: torch.device) -> Tuple[EmergentCommunicationEnv, SpeakerAgent, ListenerAgent]:
    """Construct environment, Speaker, and Listener from config."""
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


def train(cfg: Dict) -> None:
    """Run the policy-gradient training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["training"]["seed"]))

    env, speaker, listener = build_env_and_agents(cfg, device)

    episodes = int(cfg["training"]["episodes"])
    batch_size = int(cfg["training"]["batch_size"])
    entropy_coeff = float(cfg["training"]["entropy_coeff"])
    log_interval = int(cfg["training"]["log_interval"])

    speaker_optim = optim.Adam(speaker.parameters(), lr=float(cfg["training"]["speaker_lr"]))
    listener_optim = optim.Adam(listener.parameters(), lr=float(cfg["training"]["listener_lr"]))

    all_rewards = []

    for episode in range(1, episodes + 1):
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
            msg_logits = msg_logits.squeeze(0)  # (L, V)
            msg_dist = torch.distributions.Categorical(logits=msg_logits)
            message = msg_dist.sample()  # (L,)
            log_prob_msg = msg_dist.log_prob(message).sum()
            entropy_msg = msg_dist.entropy().sum()

            # Listener acts
            scores = listener.forward(objects, message)  # (N,)
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

        # Stack
        log_probs_s_t = torch.stack(log_probs_s)
        log_probs_l_t = torch.stack(log_probs_l)
        rewards_t = torch.stack(rewards)
        entropies_s_t = torch.stack(entropies_s)
        entropies_l_t = torch.stack(entropies_l)

        # Baseline: mean reward in batch
        baseline = rewards_t.mean()
        advantages = rewards_t - baseline

        speaker_loss = -(log_probs_s_t * advantages.detach()).mean() - entropy_coeff * entropies_s_t.mean()
        listener_loss = -(log_probs_l_t * advantages.detach()).mean() - entropy_coeff * entropies_l_t.mean()

        speaker_optim.zero_grad()
        listener_optim.zero_grad()
        (speaker_loss + listener_loss).backward()
        nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=5.0)
        nn.utils.clip_grad_norm_(listener.parameters(), max_norm=5.0)
        speaker_optim.step()
        listener_optim.step()

        avg_reward = float(rewards_t.mean().item())
        all_rewards.append(avg_reward)

        if episode % log_interval == 0:
            recent = all_rewards[-log_interval:]
            print(
                f"Episode {episode}/{episodes} | "
                f"avg_reward (last {log_interval} batches): {sum(recent) / len(recent):.3f}"
            )

    # Save trained models for later evaluation / visualization.
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "referential_game/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(speaker.state_dict(), ckpt_dir / "speaker.pt")
    torch.save(listener.state_dict(), ckpt_dir / "listener.pt")
    print(f"Saved checkpoints to {ckpt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Speaker and Listener in a referential game.")
    parser.add_argument(
        "--config",
        type=str,
        default="referential_game/config.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

