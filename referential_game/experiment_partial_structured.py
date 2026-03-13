"""
Partial-observability + structured-distractor referential game experiment.

Features:
- Partial observability: Speaker sees (color, shape, constant_size); Listener sees (constant_color, shape, size).
- Structured distractors: distractors share at least one attribute with the target.
- Ambiguity penalty: reward scaled by 1 / (# plausible objects given the message).
- Generalization test: train on subset of (color, shape) pairs, evaluate on held-out pairs.

Metrics / Plots (saved to `referential_game/results/`):
- po_success_curve.png                 (in-distribution success vs batch)
- po_ambiguity_curve.png               (average # plausible objects vs batch)
- po_token_entropy_curve.png           (token entropy vs batch)
- po_generalization_bar.png            (success in vs out of distribution)
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
from .environment import WorldConfig
from .train import load_config


@dataclass(frozen=True)
class Triple:
    color: int
    shape: int
    size: int


def all_triples(world: WorldConfig) -> List[Triple]:
    triples = []
    for c in range(world.color_dim):
        for s in range(world.shape_dim):
            for z in range(world.size_dim):
                triples.append(Triple(c, s, z))
    return triples


def build_train_test_splits(world: WorldConfig, holdout_frac: float = 0.25, seed: int = 0) -> Tuple[List[Triple], List[Triple]]:
    """Randomly split (color, shape, size) triples into train / held-out.

    Args:
        world: WorldConfig defining attribute cardinalities.
        holdout_frac: Fraction of triples to reserve for held-out evaluation
            (e.g., 0.2–0.3).
        seed: Random seed for reproducibility.
    """
    triples = all_triples(world)
    n_total = len(triples)
    n_holdout = max(1, int(round(n_total * holdout_frac)))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    hold_idx = set(perm[:n_holdout])

    train, heldout = [], []
    for i, t in enumerate(triples):
        if i in hold_idx:
            heldout.append(t)
        else:
            train.append(t)
    return train, heldout


def sample_structured_world(
    world: WorldConfig,
    triples_pool: List[Triple],
    num_objects: int,
) -> Tuple[np.ndarray, int]:
    """Sample target and structured distractors from a pool of allowed triples.

    - Target chosen uniformly from triples_pool.
    - Distractors share at least one attribute (color, shape, or size) with target.
    """
    # Target chosen uniformly from the pool
    target = triples_pool[np.random.randint(0, len(triples_pool))]

    objects = [target]
    # Precompute pools by attribute match
    same_color = [t for t in triples_pool if t.color == target.color and t != target]
    same_shape = [t for t in triples_pool if t.shape == target.shape and t != target]
    same_size = [t for t in triples_pool if t.size == target.size and t != target]

    all_tr = set(triples_pool)

    # Fill up to the requested num_objects.
    # If the pool is small, we may sample the same distractor multiple times;
    # this is fine, and keeps evaluation difficulty comparable between splits.
    while len(objects) < num_objects:
        # Randomly choose which attribute to match
        choice = np.random.choice(["color", "shape", "size"])
        if choice == "color" and same_color:
            cand = np.random.choice(same_color)
        elif choice == "shape" and same_shape:
            cand = np.random.choice(same_shape)
        elif choice == "size" and same_size:
            cand = np.random.choice(same_size)
        else:
            # Fallback: any different triple
            cand = np.random.choice(list(all_tr - {target}))

        if cand not in objects:
            objects.append(cand)

    # Shuffle objects so target is not always index 0
    idxs = np.arange(num_objects)
    np.random.shuffle(idxs)
    arr = np.zeros((num_objects, 3), dtype=np.int64)
    target_index = None
    for new_i, old_i in enumerate(idxs):
        t = objects[old_i]
        arr[new_i] = [t.color, t.shape, t.size]
        if old_i == 0:
            target_index = new_i
    assert target_index is not None
    return arr, target_index


def partial_observations(
    objects_np: np.ndarray,
    target_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct partial observations for Speaker and Listener.

    Speaker: sees (color, shape, constant_size=0).
    Listener: sees (constant_color=0, shape, size) for each object.
    """
    objs = torch.from_numpy(objects_np.astype(np.int64))
    target_full = objs[target_index]

    # Speaker: mask size
    speaker_obs = target_full.clone()
    speaker_obs[2] = 0  # size hidden / constant

    # Listener: mask color
    listener_obs = objs.clone()
    listener_obs[:, 0] = 0
    return speaker_obs, listener_obs


def mutual_information_token_attr(tokens: np.ndarray, attrs: np.ndarray, num_attr_vals: int, V: int) -> float:
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
        ratio[p_joint == 0] = 1.0
        mi = (p_joint * np.log2(ratio)).sum()
    return float(mi)


def run_training_partial_structured(cfg: Dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["training"]["seed"]))

    world = WorldConfig(
        num_objects=int(cfg["world"]["num_objects"]),
        colors=tuple(cfg["world"]["colors"]),
        shapes=tuple(cfg["world"]["shapes"]),
        sizes=tuple(cfg["world"]["sizes"]),
    )
    train_triples, held_triples = build_train_test_splits(
        world,
        holdout_frac=0.30,
        seed=int(cfg["training"]["seed"]),
    )

    vocab_size = int(cfg["channel"]["vocabulary_size"])
    msg_len = int(cfg["channel"]["message_length"])
    vocab = Vocabulary(size=vocab_size)

    speaker_cfg = SpeakerConfig(hidden_dim=int(cfg["agents"]["speaker_hidden_dim"]))
    listener_cfg = ListenerConfig(hidden_dim=int(cfg["agents"]["listener_hidden_dim"]))

    speaker = SpeakerAgent(
        vocab=vocab,
        message_length=msg_len,
        world_color_dim=world.color_dim,
        world_shape_dim=world.shape_dim,
        world_size_dim=world.size_dim,
        config=speaker_cfg,
    ).to(device)
    listener = ListenerAgent(
        vocab=vocab,
        message_length=msg_len,
        world_color_dim=world.color_dim,
        world_shape_dim=world.shape_dim,
        world_size_dim=world.size_dim,
        config=listener_cfg,
    ).to(device)

    episodes = int(cfg["training"]["episodes"])
    batch_size = int(cfg["training"]["batch_size"])
    entropy_coeff = float(cfg["training"]["entropy_coeff"])
    comm_cost = 0.02  # small cost per non-zero token
    ambiguity_threshold = 0.5  # plausible if prob >= 0.5 * max_prob

    optim_s = optim.Adam(speaker.parameters(), lr=float(cfg["training"]["speaker_lr"]))
    optim_l = optim.Adam(listener.parameters(), lr=float(cfg["training"]["listener_lr"]))

    success_curve: List[float] = []
    ambiguity_curve: List[float] = []
    entropy_curve: List[float] = []

    for batch_idx in range(episodes):
        log_probs_s = []
        log_probs_l = []
        entropies_s = []
        entropies_l = []
        rewards = []
        succs = []
        ambiguities = []
        tokens_batch = []

        for _ in range(batch_size):
            # Sample structured world from TRAIN triples
            objs_np, target_idx = sample_structured_world(world, train_triples, world.num_objects)
            speaker_obs, listener_obs = partial_observations(objs_np, target_idx)
            speaker_obs = speaker_obs.to(device)
            listener_obs = listener_obs.to(device)

            # Speaker acts
            msg_logits, _ = speaker.forward(speaker_obs.unsqueeze(0))  # (1, L, V)
            msg_logits = msg_logits.squeeze(0)  # (L, V)
            msg_dist = torch.distributions.Categorical(logits=msg_logits)
            message = msg_dist.sample()  # (L,)
            log_prob_msg = msg_dist.log_prob(message).sum()
            entropy_msg = msg_dist.entropy().sum()

            # Listener acts
            scores = listener.forward(listener_obs, message)  # (N,)
            act_dist = torch.distributions.Categorical(logits=scores)
            action = act_dist.sample()
            log_prob_act = act_dist.log_prob(action)
            entropy_act = act_dist.entropy()

            # Base success reward
            success = int(action.item() == target_idx)

            # Ambiguity: how many objects are plausible given the message?
            probs = act_dist.probs.detach().cpu().numpy()
            max_p = probs.max()
            plausible = (probs >= ambiguity_threshold * max_p).sum()
            n_plausible = int(plausible)

            # Reward: scale by ambiguity and communication cost
            base_reward = float(success) / max(1, n_plausible)
            used_len = int((message != 0).sum().item())
            reward = base_reward - comm_cost * used_len

            log_probs_s.append(log_prob_msg)
            log_probs_l.append(log_prob_act)
            entropies_s.append(entropy_msg)
            entropies_l.append(entropy_act)
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
            succs.append(float(success))
            ambiguities.append(float(n_plausible))
            tokens_batch.append(message.detach().cpu().numpy())

        log_probs_s_t = torch.stack(log_probs_s)
        log_probs_l_t = torch.stack(log_probs_l)
        rewards_t = torch.stack(rewards)
        entropies_s_t = torch.stack(entropies_s)
        entropies_l_t = torch.stack(entropies_l)

        baseline = rewards_t.mean()
        adv = rewards_t - baseline

        loss_s = -(log_probs_s_t * adv.detach()).mean() - entropy_coeff * entropies_s_t.mean()
        loss_l = -(log_probs_l_t * adv.detach()).mean() - entropy_coeff * entropies_l_t.mean()

        optim_s.zero_grad()
        optim_l.zero_grad()
        (loss_s + loss_l).backward()
        nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=5.0)
        nn.utils.clip_grad_norm_(listener.parameters(), max_norm=5.0)
        optim_s.step()
        optim_l.step()

        # Batch-level metrics
        success_curve.append(float(np.mean(succs)))
        ambiguity_curve.append(float(np.mean(ambiguities)))

        flat_tokens = np.concatenate(tokens_batch, axis=0)
        counts = np.bincount(flat_tokens, minlength=vocab_size).astype(np.float64)
        p = counts / counts.sum() if counts.sum() > 0 else counts
        p_nonzero = p[p > 0]
        H = float(-(p_nonzero * np.log2(p_nonzero)).sum()) if p_nonzero.size > 0 else 0.0
        entropy_curve.append(H)

        if (batch_idx + 1) % max(1, episodes // 20) == 0:
            print(
                f"Batch {batch_idx+1}/{episodes} | "
                f"success={success_curve[-1]:.3f}, "
                f"ambig={ambiguity_curve[-1]:.2f}, "
                f"entropy={entropy_curve[-1]:.2f}"
            )

    # ------------------------------------------------------------------
    # Generalization evaluation: held-out triples
    # ------------------------------------------------------------------
    def evaluate_split(triples_pool: List[Triple], num_eps: int) -> Tuple[float, float, float]:
        succs_eval = []
        tokens_eval = []
        colors = []
        shapes = []
        sizes = []
        with torch.no_grad():
            for i in range(num_eps):
                objs_np, target_idx = sample_structured_world(world, triples_pool, world.num_objects)
                speaker_obs, listener_obs = partial_observations(objs_np, target_idx)
                speaker_obs = speaker_obs.to(device)
                listener_obs = listener_obs.to(device)

                msg_logits, _ = speaker.forward(speaker_obs.unsqueeze(0))
                msg_logits = msg_logits.squeeze(0)
                message = msg_logits.argmax(dim=-1)

                scores = listener.forward(listener_obs, message)
                action = int(scores.argmax().item())
                success = int(action == target_idx)
                succs_eval.append(float(success))

                tokens_eval.append(message.detach().cpu().numpy())

                full_objs = torch.from_numpy(objs_np)
                target_full = full_objs[target_idx]
                colors.append(int(target_full[0].item()))
                shapes.append(int(target_full[1].item()))
                sizes.append(int(target_full[2].item()))

                if (i + 1) % max(1, num_eps // 5) == 0:
                    print(f"  eval progress: {i+1}/{num_eps}")

        succ_rate = float(np.mean(succs_eval))
        tokens_arr = np.stack(tokens_eval, axis=0)
        flat = tokens_arr.reshape(-1)
        counts = np.bincount(flat, minlength=vocab_size).astype(np.float64)
        p = counts / counts.sum() if counts.sum() > 0 else counts
        p_nonzero = p[p > 0]
        H = float(-(p_nonzero * np.log2(p_nonzero)).sum()) if p_nonzero.size > 0 else 0.0

        colors_arr = np.array(colors, dtype=np.int64)
        shapes_arr = np.array(shapes, dtype=np.int64)
        sizes_arr = np.array(sizes, dtype=np.int64)
        pos0 = tokens_arr[:, 0]
        mi_color = mutual_information_token_attr(pos0, colors_arr, world.color_dim, vocab_size)
        mi_shape = mutual_information_token_attr(pos0, shapes_arr, world.shape_dim, vocab_size)
        mi_size = mutual_information_token_attr(pos0, sizes_arr, world.size_dim, vocab_size)

        print(
            f"  eval: succ={succ_rate:.3f}, H(tokens)={H:.2f}, "
            f"MI(pos0;color)={mi_color:.2f}, MI(shape)={mi_shape:.2f}, MI(size)={mi_size:.2f}"
        )
        return succ_rate, H, max(mi_color, mi_shape, mi_size)

    eval_eps = int(cfg.get("evaluation", {}).get("episodes", 5000))

    print("Evaluating IN-distribution (train triples)...")
    succ_in, H_in, mi_in = evaluate_split(train_triples, num_eps=eval_eps)
    print("Evaluating OUT-of-distribution (held-out triples)...")
    succ_out, H_out, mi_out = evaluate_split(held_triples, num_eps=eval_eps)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    results_dir = Path("referential_game/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(1, episodes + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(xs, success_curve)
    plt.xlabel("Training batch")
    plt.ylabel("Success rate")
    plt.title("Partial-observability structured game: success over time")
    plt.tight_layout()
    plt.savefig(results_dir / "po_success_curve.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ambiguity_curve)
    plt.xlabel("Training batch")
    plt.ylabel("Avg # plausible objects")
    plt.title("Ambiguity (plausible objects) over time")
    plt.tight_layout()
    plt.savefig(results_dir / "po_ambiguity_curve.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(xs, entropy_curve)
    plt.xlabel("Training batch")
    plt.ylabel("Token entropy (bits)")
    plt.title("Token entropy over time")
    plt.tight_layout()
    plt.savefig(results_dir / "po_token_entropy_curve.png")
    plt.close()

    # Generalization bar plot
    plt.figure(figsize=(6, 4))
    labels = ["In-distribution", "Held-out"]
    succs = [succ_in, succ_out]
    plt.bar(labels, succs)
    plt.ylabel("Success rate")
    plt.title("Generalization: in vs held-out triples")
    plt.tight_layout()
    plt.savefig(results_dir / "po_generalization_bar.png")
    plt.close()

    print(f"Saved partial-structured experiment plots to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Partial-observability structured referential game experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="referential_game/config.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_training_partial_structured(cfg)


if __name__ == "__main__":
    main()

