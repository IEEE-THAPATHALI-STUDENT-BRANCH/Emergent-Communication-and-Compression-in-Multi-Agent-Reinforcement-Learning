"""
Progressive communication pressure experiment for the referential game.

Curriculum over 4 stages:
    Stage 1: unconstrained channel
    Stage 2: shorter effective message length
    Stage 3: add communication cost
    Stage 4: add channel noise (+ cost)

We keep a fixed maximum vocabulary size and message length in the model
and implement *effective* constraints by masking logits and charging
communication cost only for "used" positions.

Outputs (saved to `referential_game/results/`):
  - progress_success_curve.png          (success vs training batch, with stage markers)
  - progress_avg_message_length.png     (effective message length vs training batch)
  - progress_token_entropy.png          (token entropy vs training batch)
  - progress_comm_cost_vs_success.png   (per-stage comm cost vs success)

Usage:
    python -m referential_game.experiment_progressive --config referential_game/config.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass
class Stage:
    """One curriculum stage with effective communication constraints."""

    name: str
    start_batch: int
    end_batch: int
    effective_vocab: int
    effective_length: int
    comm_cost: float
    noise_prob: float


def build_env_and_agents(cfg: Dict, device: torch.device) -> Tuple[EmergentCommunicationEnv, SpeakerAgent, ListenerAgent]:
    """Build env + agents using *maximum* channel capacity from config.

    The curriculum will impose stricter constraints on top of this during training.
    """
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


def make_curriculum(cfg: Dict, max_batches: int) -> List[Stage]:
    """Create a 4-stage curriculum schedule over [0, max_batches)."""
    # Split training batches into 4 equal stages
    quarter = max_batches // 4

    # Use config channel as Stage 1 maximum
    max_vocab = int(cfg["channel"]["vocabulary_size"])
    max_len = int(cfg["channel"]["message_length"])

    stages = [
        Stage(
            name="Stage 1 (unconstrained)",
            start_batch=0,
            end_batch=quarter,
            effective_vocab=max_vocab,
            effective_length=max_len,
            comm_cost=0.0,
            noise_prob=0.0,
        ),
        Stage(
            name="Stage 2 (shorter messages)",
            start_batch=quarter,
            end_batch=2 * quarter,
            effective_vocab=max_vocab // 2,
            effective_length=max(1, max_len // 2),
            comm_cost=0.0,
            noise_prob=0.0,
        ),
        Stage(
            name="Stage 3 (add cost)",
            start_batch=2 * quarter,
            end_batch=3 * quarter,
            effective_vocab=max_vocab // 2,
            effective_length=max(1, max_len // 2),
            comm_cost=0.05,
            noise_prob=0.0,
        ),
        Stage(
            name="Stage 4 (cost + noise)",
            start_batch=3 * quarter,
            end_batch=max_batches,
            effective_vocab=max_vocab // 2,
            effective_length=max(1, max_len // 2),
            comm_cost=0.05,
            noise_prob=0.1,
        ),
    ]

    # Ensure last stage covers the rest
    stages[-1].end_batch = max_batches
    return stages


def find_stage(stages: List[Stage], batch_idx: int) -> Stage:
    for s in stages:
        if s.start_batch <= batch_idx < s.end_batch:
            return s
    return stages[-1]


def experiment_progressive(cfg: Dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["training"]["seed"]))

    env, speaker, listener = build_env_and_agents(cfg, device)

    episodes = int(cfg["training"]["episodes"])
    batch_size = int(cfg["training"]["batch_size"])
    entropy_coeff = float(cfg["training"]["entropy_coeff"])

    speaker_optim = optim.Adam(speaker.parameters(), lr=float(cfg["training"]["speaker_lr"]))
    listener_optim = optim.Adam(listener.parameters(), lr=float(cfg["training"]["listener_lr"]))

    # We interpret "episodes" here as number of batches to keep it consistent with earlier scripts.
    stages = make_curriculum(cfg, episodes)
    print("Curriculum:")
    for s in stages:
        print(
            f"  {s.name}: batches {s.start_batch}..{s.end_batch-1}, "
            f"vocab={s.effective_vocab}, len={s.effective_length}, "
            f"cost={s.comm_cost}, noise={s.noise_prob}"
        )

    max_vocab = env.channel.vocabulary_size
    max_len = env.channel.message_length

    # Tracking over batches
    success_curve: List[float] = []
    eff_len_curve: List[float] = []
    entropy_curve: List[float] = []
    cost_curve: List[float] = []

    # Per-stage stats for comm cost vs success
    stage_success_sums = {s.name: 0.0 for s in stages}
    stage_success_counts = {s.name: 0 for s in stages}

    for batch_idx in range(episodes):
        stage = find_stage(stages, batch_idx)

        log_probs_s = []
        log_probs_l = []
        entropies_s = []
        entropies_l = []
        rewards = []
        base_rewards = []
        used_lengths = []
        tokens_this_batch = []

        for _ in range(batch_size):
            obs = env.reset()
            target = obs["speaker_obs"]
            objects = obs["listener_obs"]

            # Speaker forward over full channel
            msg_logits, _ = speaker.forward(target.unsqueeze(0))  # (1, L, V)
            msg_logits = msg_logits.squeeze(0)  # (L, V)

            # Mask logits to only allow effective_vocab tokens for positions being used
            logits_masked = msg_logits.clone()
            if stage.effective_vocab < max_vocab:
                logits_masked[:, stage.effective_vocab:] = -1e9

            # Sample tokens for used positions
            tokens = torch.zeros(max_len, dtype=torch.long, device=device)
            log_prob_msg = torch.zeros((), device=device)
            entropy_msg = torch.zeros((), device=device)

            for pos in range(max_len):
                if pos < stage.effective_length:
                    dist = torch.distributions.Categorical(logits=logits_masked[pos])
                    t = dist.sample()
                    tokens[pos] = t
                    log_prob_msg = log_prob_msg + dist.log_prob(t)
                    entropy_msg = entropy_msg + dist.entropy()
                else:
                    # positions beyond effective_length are forced to 0, no cost / no gradient
                    tokens[pos] = 0

            # Apply channel noise before Listener sees the message
            noisy_tokens = tokens.clone()
            if stage.noise_prob > 0.0:
                for pos in range(stage.effective_length):
                    if torch.rand(()) < stage.noise_prob:
                        noisy_tokens[pos] = torch.randint(
                            low=0,
                            high=stage.effective_vocab,
                            size=(),
                            device=device,
                        )

            # Listener acts based on noisy message
            scores = listener.forward(objects, noisy_tokens)
            act_dist = torch.distributions.Categorical(logits=scores)
            action = act_dist.sample()
            log_prob_act = act_dist.log_prob(action)
            entropy_act = act_dist.entropy()

            _, base_reward, _, _ = env.step(noisy_tokens, int(action.item()))

            # Communication cost: per non-zero token in effective positions
            effective_tokens = tokens[: stage.effective_length]
            used_len = int((effective_tokens != 0).sum().item())
            comm_penalty = stage.comm_cost * used_len
            reward = float(base_reward - comm_penalty)

            log_probs_s.append(log_prob_msg)
            log_probs_l.append(log_prob_act)
            entropies_s.append(entropy_msg)
            entropies_l.append(entropy_act)
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
            base_rewards.append(float(base_reward))
            used_lengths.append(used_len)
            tokens_this_batch.append(effective_tokens.detach().cpu().numpy())

        # Stack batch tensors
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
        nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=5.0)
        nn.utils.clip_grad_norm_(listener.parameters(), max_norm=5.0)
        speaker_optim.step()
        listener_optim.step()

        # Metrics for this batch
        avg_base_success = float(np.mean(base_rewards))
        avg_used_len = float(np.mean(used_lengths))

        # Token entropy (over effective positions only)
        flat_tokens = np.concatenate(tokens_this_batch, axis=0)
        token_counts = np.bincount(flat_tokens, minlength=max_vocab).astype(np.float64)
        p = token_counts / token_counts.sum() if token_counts.sum() > 0 else token_counts
        p_nonzero = p[p > 0]
        token_entropy = float(-(p_nonzero * np.log2(p_nonzero)).sum()) if p_nonzero.size > 0 else 0.0

        success_curve.append(avg_base_success)
        eff_len_curve.append(avg_used_len)
        entropy_curve.append(token_entropy)
        cost_curve.append(stage.comm_cost)

        stage_success_sums[stage.name] += avg_base_success
        stage_success_counts[stage.name] += 1

        if (batch_idx + 1) % max(1, episodes // 20) == 0:
            print(
                f"Batch {batch_idx+1}/{episodes} | {stage.name} | "
                f"success={avg_base_success:.3f}, eff_len={avg_used_len:.2f}, "
                f"entropy={token_entropy:.2f}, cost={stage.comm_cost:.3f}"
            )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    results_dir = Path("referential_game/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    xs = np.arange(1, episodes + 1)

    # Success curve with stage boundaries
    plt.figure(figsize=(8, 4))
    plt.plot(xs, success_curve, label="success rate (per batch)")
    for s in stages:
        plt.axvline(s.start_batch + 1, color="gray", linestyle="--", alpha=0.3)
    plt.xlabel("Training batch")
    plt.ylabel("Success rate")
    plt.title("Progressive communication pressure: success over time")
    plt.tight_layout()
    plt.savefig(results_dir / "progress_success_curve.png")
    plt.close()

    # Average effective message length
    plt.figure(figsize=(8, 4))
    plt.plot(xs, eff_len_curve)
    for s in stages:
        plt.axvline(s.start_batch + 1, color="gray", linestyle="--", alpha=0.3)
    plt.xlabel("Training batch")
    plt.ylabel("Average effective message length")
    plt.title("Effective message length over time")
    plt.tight_layout()
    plt.savefig(results_dir / "progress_avg_message_length.png")
    plt.close()

    # Token entropy
    plt.figure(figsize=(8, 4))
    plt.plot(xs, entropy_curve)
    for s in stages:
        plt.axvline(s.start_batch + 1, color="gray", linestyle="--", alpha=0.3)
    plt.xlabel("Training batch")
    plt.ylabel("Token entropy (bits)")
    plt.title("Token entropy over time")
    plt.tight_layout()
    plt.savefig(results_dir / "progress_token_entropy.png")
    plt.close()

    # Communication cost vs success (per stage)
    stage_means = []
    stage_costs = []
    for s in stages:
        if stage_success_counts[s.name] > 0:
            mean_succ = stage_success_sums[s.name] / stage_success_counts[s.name]
            stage_means.append(mean_succ)
            stage_costs.append(s.comm_cost)

    plt.figure(figsize=(5, 4))
    plt.scatter(stage_costs, stage_means)
    for cost, succ, s in zip(stage_costs, stage_means, stages):
        plt.text(cost, succ, s.name.split()[1], fontsize=8, ha="center", va="bottom")
    plt.xlabel("Communication cost λ")
    plt.ylabel("Mean success rate (per stage)")
    plt.title("Communication cost vs success")
    plt.tight_layout()
    plt.savefig(results_dir / "progress_comm_cost_vs_success.png")
    plt.close()

    print(f"Saved progressive experiment plots to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive communication pressure experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="referential_game/config.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    experiment_progressive(cfg)


if __name__ == "__main__":
    main()

