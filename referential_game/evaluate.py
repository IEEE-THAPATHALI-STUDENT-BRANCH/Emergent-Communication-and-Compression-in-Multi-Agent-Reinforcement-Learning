"""
Evaluation and visualization script for trained referential-game agents.

This script loads trained Speaker and Listener checkpoints, runs a number of
episodes without learning, and computes:

- overall success rate
- token usage frequencies and entropy
- simple mutual information between attributes and tokens

It also saves simple plots of token frequencies and token-attribute
relationships to PNG files.

Usage:
    python -m referential_game.evaluate --config referential_game/config.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .agents import ListenerAgent, ListenerConfig, SpeakerAgent, SpeakerConfig
from .communication import Vocabulary
from .environment import ChannelConfig, EmergentCommunicationEnv, WorldConfig
from .train import load_config, build_env_and_agents


def load_checkpoints(
    cfg: Dict,
    speaker: SpeakerAgent,
    listener: ListenerAgent,
    device: torch.device,
) -> None:
    """Load trained weights into Speaker and Listener."""
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "referential_game/checkpoints"))
    speaker_path = ckpt_dir / "speaker.pt"
    listener_path = ckpt_dir / "listener.pt"

    if not speaker_path.exists() or not listener_path.exists():
        raise FileNotFoundError(
            f"Checkpoint files not found in {ckpt_dir}. "
            f"Run training first (referential_game.train)."
        )

    speaker.load_state_dict(torch.load(speaker_path, map_location=device))
    listener.load_state_dict(torch.load(listener_path, map_location=device))
    speaker.eval()
    listener.eval()


def empirical_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy (bits) from integer counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def evaluate(cfg: Dict) -> None:
    """Run evaluation episodes and visualize communication statistics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, speaker, listener = build_env_and_agents(cfg, device)
    load_checkpoints(cfg, speaker, listener, device)

    num_episodes = int(cfg.get("evaluation", {}).get("episodes", 10000))
    L = env.channel.message_length
    V = env.channel.vocabulary_size

    rewards = []
    # Collect messages and attributes
    all_messages = []
    colors = []
    shapes = []
    sizes = []
    successes = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            target = obs["speaker_obs"]
            objects = obs["listener_obs"]

            # Speaker and Listener act greedily (argmax) instead of sampling
            msg_logits, _ = speaker.forward(target.unsqueeze(0))  # (1, L, V)
            msg_logits = msg_logits.squeeze(0)  # (L, V)
            message = msg_logits.argmax(dim=-1)  # (L,)

            scores = listener.forward(objects, message)  # (N,)
            action = int(scores.argmax().item())

            _, reward, _, _ = env.step(message, action)

            rewards.append(reward)
            all_messages.append(message.cpu().numpy())

            colors.append(int(target[0].item()))
            shapes.append(int(target[1].item()))
            sizes.append(int(target[2].item()))
            successes.append(int(reward > 0.5))

    rewards_arr = np.array(rewards, dtype=np.float32)
    success_rate = float(rewards_arr.mean())
    print(f"Evaluation episodes: {num_episodes}")
    print(f"Success rate: {success_rate:.3f}")

    messages_arr = np.stack(all_messages, axis=0)  # (N, L)
    colors_arr = np.array(colors, dtype=np.int64)
    shapes_arr = np.array(shapes, dtype=np.int64)
    sizes_arr = np.array(sizes, dtype=np.int64)
    successes_arr = np.array(successes, dtype=np.int64)

    # ------------------------------------------------------------------
    # Token usage statistics
    # ------------------------------------------------------------------
    # Overall token histogram
    flat_tokens = messages_arr.reshape(-1)
    overall_counts = np.bincount(flat_tokens, minlength=V)
    overall_entropy = empirical_entropy(overall_counts)
    print(f"Overall token entropy: {overall_entropy:.3f} bits")

    # Per-position histograms and entropies
    pos_entropies = []
    for pos in range(L):
        counts = np.bincount(messages_arr[:, pos], minlength=V)
        H = empirical_entropy(counts)
        pos_entropies.append(H)
        print(f"Position {pos}: entropy={H:.3f} bits")

    # ------------------------------------------------------------------
    # Simple mutual information: token at position 0 vs attributes
    # ------------------------------------------------------------------
    def mutual_information_token_attr(tokens: np.ndarray, attrs: np.ndarray, num_attr_vals: int) -> float:
        """Estimate I(token; attr) for discrete variables."""
        joint_counts = np.zeros((V, num_attr_vals), dtype=np.float64)
        for t, a in zip(tokens, attrs):
            joint_counts[t, a] += 1.0

        total = joint_counts.sum()
        if total == 0:
            return 0.0

        p_joint = joint_counts / total
        p_t = p_joint.sum(axis=1, keepdims=True)
        p_a = p_joint.sum(axis=0, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p_joint / (p_t * p_a)
            ratio[p_joint == 0] = 1.0  # 0 * log(0) convention
            mi = (p_joint * np.log2(ratio)).sum()
        return float(mi)

    pos0_tokens = messages_arr[:, 0]
    world_cfg = env.world

    mi_color = mutual_information_token_attr(pos0_tokens, colors_arr, world_cfg.color_dim)
    mi_shape = mutual_information_token_attr(pos0_tokens, shapes_arr, world_cfg.shape_dim)
    mi_size = mutual_information_token_attr(pos0_tokens, sizes_arr, world_cfg.size_dim)

    print(f"MI(token_pos0 ; color): {mi_color:.3f} bits")
    print(f"MI(token_pos0 ; shape): {mi_shape:.3f} bits")
    print(f"MI(token_pos0 ; size): {mi_size:.3f} bits")

    # ------------------------------------------------------------------
    # Visualization: token frequencies and token-color heatmap (pos 0)
    # ------------------------------------------------------------------
    results_dir = Path("referential_game/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Overall token frequency bar plot
    plt.figure(figsize=(6, 4))
    xs = np.arange(V)
    plt.bar(xs, overall_counts)
    plt.xlabel("Token")
    plt.ylabel("Count")
    plt.title("Overall token usage")
    plt.tight_layout()
    plt.savefig(results_dir / "token_frequencies.png")
    plt.close()

    # Helper to build and save token–attribute heatmaps for position 0
    def save_token_attr_heatmap(
        tokens: np.ndarray,
        attrs: np.ndarray,
        num_attr_vals: int,
        name: str,
        pos: int,
    ) -> None:
        joint = np.zeros((V, num_attr_vals), dtype=np.float64)
        for t, a in zip(tokens, attrs):
            joint[t, a] += 1.0

        row_sums = joint.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            heat = np.divide(joint, row_sums, where=row_sums > 0)

        plt.figure(figsize=(6, 4))
        im = plt.imshow(heat, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(im, label=f"p({name} | token)")
        plt.xlabel(f"{name} index")
        plt.ylabel(f"Token (position {pos})")
        plt.title(f"Token–{name} relation at position {pos}")
        plt.tight_layout()
        plt.savefig(results_dir / f"token_{name}_heatmap_pos{pos}.png")
        plt.close()

    # Token vs color / shape / size heatmaps at each message position
    for pos in range(L):
        pos_tokens = messages_arr[:, pos]
        save_token_attr_heatmap(pos_tokens, colors_arr, world_cfg.color_dim, "color", pos)
        save_token_attr_heatmap(pos_tokens, shapes_arr, world_cfg.shape_dim, "shape", pos)
        save_token_attr_heatmap(pos_tokens, sizes_arr, world_cfg.size_dim, "size", pos)

    # ------------------------------------------------------------------
    # Token-wise success rate: how effective each token is.
    # ------------------------------------------------------------------
    token_success_counts = np.zeros(V, dtype=np.float64)
    token_total_counts = np.zeros(V, dtype=np.float64)
    for t, s in zip(flat_tokens, successes_arr.repeat(L)):
        token_total_counts[t] += 1.0
        token_success_counts[t] += float(s)

    with np.errstate(divide="ignore", invalid="ignore"):
        token_success_rate = np.divide(
            token_success_counts,
            token_total_counts,
            where=token_total_counts > 0,
        )

    plt.figure(figsize=(6, 4))
    xs = np.arange(V)
    plt.bar(xs, token_success_rate)
    plt.xlabel("Token")
    plt.ylabel("Success rate when token used")
    plt.title("Per-token success rate")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(results_dir / "token_success_rates.png")
    plt.close()

    print(f"Saved plots to {results_dir}")

    print(f"Saved plots to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained referential-game agents.")
    parser.add_argument(
        "--config",
        type=str,
        default="referential_game/config.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()

