"""Training entry point for Experiment 5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any, Dict, List

from .agents import ModelConfig, build_population, empty_message, message_content_length, require_torch, sample_message
from .config import condition_from_config, load_config, save_config_snapshot
from .messages import apply_content_noise
from .world import CandidateWorld, WorldSpec, split_objects, world_for_split


def _torch_imports():
    require_torch()
    import torch
    import torch.nn.functional as F

    return torch, F


def _tensor_object(obj, device):
    torch, _ = _torch_imports()
    return torch.tensor(obj, dtype=torch.long, device=device)


def _tensor_world(world: CandidateWorld, device):
    torch, _ = _torch_imports()
    return torch.tensor(world.candidates, dtype=torch.long, device=device)


def _noise_message(message, vocabulary_size: int, p_noise: float, rng: random.Random, device):
    torch, _ = _torch_imports()
    noisy, stats = apply_content_noise(message.detach().cpu().tolist(), vocabulary_size, p_noise, rng)
    return torch.tensor(noisy, dtype=torch.long, device=device), stats


def _sample_pair(population_size: int, rng: random.Random) -> tuple[int, int]:
    a, b = rng.sample(range(population_size), 2)
    return a, b


def run_episode(
    population,
    world: CandidateWorld,
    condition: Dict[str, Any],
    rng: random.Random,
    device,
    greedy: bool = False,
):
    torch, F = _torch_imports()
    kind = condition["kind"]
    vocab_size = int(condition["vocabulary_size"])
    l_initial = int(condition.get("l_initial", 0))
    l_feedback = int(condition.get("l_feedback", 0))
    l_response = int(condition.get("l_response", 0))
    p_noise = float(condition.get("p_noise", 0.0))
    lambda_cost = float(condition.get("lambda_cost", 0.0))

    informant_id, receiver_id = _sample_pair(len(population), rng)
    informant = population[informant_id]
    receiver = population[receiver_id]

    target = _tensor_object(world.target, device)
    candidates = _tensor_world(world, device)
    target_index = torch.tensor(world.target_index, dtype=torch.long, device=device)

    log_probs = []
    entropies = []
    values = []
    noise_stats = []

    if kind == "no_comm":
        m_initial = empty_message(0, device)
        q_feedback = empty_message(0, device)
        m_response = empty_message(0, device)
    else:
        logits, value = informant.initial_logits(target, l_initial)
        msg = sample_message(logits, vocab_size, greedy=greedy)
        m_initial = msg["message"]
        log_probs.append(msg["log_prob"])
        entropies.append(msg["entropy"])
        values.append(value)

    m_initial_rx, stats = _noise_message(m_initial, vocab_size, p_noise, rng, device)
    noise_stats.append(stats)

    if kind == "duplex":
        logits, value = receiver.feedback_logits(candidates, m_initial_rx, l_feedback)
        msg = sample_message(logits, vocab_size, greedy=greedy)
        q_feedback = msg["message"]
        log_probs.append(msg["log_prob"])
        entropies.append(msg["entropy"])
        values.append(value)
    else:
        q_feedback = empty_message(l_feedback, device)

    q_feedback_rx, stats = _noise_message(q_feedback, vocab_size, p_noise, rng, device)
    noise_stats.append(stats)

    if kind == "duplex":
        logits, value = informant.response_logits(target, m_initial, q_feedback_rx, l_response)
        msg = sample_message(logits, vocab_size, greedy=greedy)
        m_response = msg["message"]
        log_probs.append(msg["log_prob"])
        entropies.append(msg["entropy"])
        values.append(value)
    else:
        m_response = empty_message(l_response, device)

    m_response_rx, stats = _noise_message(m_response, vocab_size, p_noise, rng, device)
    noise_stats.append(stats)

    selection_logits, selection_value = receiver.selection_logits(candidates, m_initial_rx, q_feedback, m_response_rx)
    action_dist = torch.distributions.Categorical(logits=selection_logits)
    action = torch.argmax(selection_logits) if greedy else action_dist.sample()
    log_probs.append(action_dist.log_prob(action))
    entropies.append(action_dist.entropy())
    values.append(selection_value)

    success = float(int(action.item()) == world.target_index)
    lengths = {
        "initial_length": message_content_length(m_initial),
        "feedback_length": message_content_length(q_feedback),
        "response_length": message_content_length(m_response),
    }
    total_length = sum(lengths.values())
    reward = success - lambda_cost * total_length
    ret = torch.tensor(reward, dtype=torch.float32, device=device)

    policy_loss = torch.tensor(0.0, device=device)
    critic_loss = torch.tensor(0.0, device=device)
    entropy_sum = torch.stack(entropies).sum() if entropies else torch.tensor(0.0, device=device)
    for log_prob, value in zip(log_probs, values):
        advantage = (ret - value).detach()
        policy_loss = policy_loss - log_prob * advantage
        critic_loss = critic_loss + F.mse_loss(value, ret)

    return {
        "loss_parts": (policy_loss, critic_loss, entropy_sum),
        "success": success,
        "return": reward,
        "total_length": total_length,
        "lengths": lengths,
        "informant_id": informant_id,
        "receiver_id": receiver_id,
        "messages": {
            "initial": m_initial.detach().cpu().tolist(),
            "feedback": q_feedback.detach().cpu().tolist(),
            "response": m_response.detach().cpu().tolist(),
        },
        "noise_content_tokens": sum(s.content_tokens for s in noise_stats),
        "noise_corrupted_tokens": sum(s.corrupted_tokens for s in noise_stats),
    }


def train_condition(cfg: Dict[str, Any], condition_name: str, fold: int, seed: int) -> Path:
    torch, _ = _torch_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = random.Random(seed)

    world_cfg = cfg["world"]
    spec = WorldSpec(
        num_colors=int(world_cfg["num_colors"]),
        num_shapes=int(world_cfg["num_shapes"]),
        num_sizes=int(world_cfg["num_sizes"]),
        num_candidates=int(world_cfg["num_candidates"]),
    )
    base_condition = cfg["conditions"][condition_name]
    max_vocab = max(
        [int(base_condition.get("vocabulary_size", 0))]
        + [int(stage.get("vocabulary_size", 0)) for stage in base_condition.get("stages", [])]
    )
    model_cfg = ModelConfig(
        num_colors=spec.num_colors,
        num_shapes=spec.num_shapes,
        num_sizes=spec.num_sizes,
        num_candidates=spec.num_candidates,
        vocabulary_size=max_vocab,
        attribute_embedding_dim=int(cfg["model"]["attribute_embedding_dim"]),
        token_embedding_dim=int(cfg["model"]["token_embedding_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        gru_layers=int(cfg["model"]["gru_layers"]),
    )
    population = build_population(model_cfg, int(cfg["model"]["population_size"]), device)
    params = [p for agent in population for p in agent.parameters()]
    optimizer = torch.optim.Adam(params, lr=float(cfg["training"]["learning_rate"]))

    out_dir = Path(cfg["output_dir"]) / condition_name / f"fold_{fold}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, out_dir)
    metrics_path = out_dir / "training_metrics.csv"

    if base_condition.get("kind") == "duplex_curriculum":
        stages = [dict(stage, kind="duplex") for stage in base_condition["stages"]]
        total_batches = int(cfg["training"]["batches"])
        stage_batches = max(1, total_batches // len(stages))
    else:
        stages = [condition_from_config(cfg, condition_name)]
        stage_batches = int(cfg["training"]["batches"])

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "fold",
                "condition",
                "stage",
                "batch",
                "success",
                "return",
                "total_length",
                "initial_length",
                "feedback_length",
                "response_length",
            ],
        )
        writer.writeheader()

        global_batch = 0
        for stage_idx, condition in enumerate(stages, start=1):
            for batch in range(1, stage_batches + 1):
                global_batch += 1
                optimizer.zero_grad()
                batch_records = []
                policy_losses = []
                critic_losses = []
                entropies = []
                for _ in range(int(cfg["training"]["batch_size"])):
                    world = world_for_split(spec, fold, "id", rng)
                    record = run_episode(population, world, condition, rng, device, greedy=False)
                    policy_loss, critic_loss, entropy = record["loss_parts"]
                    policy_losses.append(policy_loss)
                    critic_losses.append(critic_loss)
                    entropies.append(entropy)
                    batch_records.append(record)

                loss = (
                    torch.stack(policy_losses).mean()
                    + float(cfg["training"]["critic_loss_weight"]) * torch.stack(critic_losses).mean()
                    - float(cfg["training"]["entropy_coefficient"]) * torch.stack(entropies).mean()
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(cfg["training"]["gradient_clip_norm"]))
                optimizer.step()

                row = {
                    "seed": seed,
                    "fold": fold,
                    "condition": condition_name,
                    "stage": stage_idx,
                    "batch": global_batch,
                    "success": sum(r["success"] for r in batch_records) / len(batch_records),
                    "return": sum(r["return"] for r in batch_records) / len(batch_records),
                    "total_length": sum(r["total_length"] for r in batch_records) / len(batch_records),
                    "initial_length": sum(r["lengths"]["initial_length"] for r in batch_records) / len(batch_records),
                    "feedback_length": sum(r["lengths"]["feedback_length"] for r in batch_records) / len(batch_records),
                    "response_length": sum(r["lengths"]["response_length"] for r in batch_records) / len(batch_records),
                }
                writer.writerow(row)
                if global_batch % int(cfg["training"]["log_interval"]) == 0:
                    print(
                        f"{condition_name} fold={fold} seed={seed} batch={global_batch} "
                        f"success={row['success']:.3f} return={row['return']:.3f}"
                    )

    checkpoint = {
        "model_config": model_cfg.__dict__,
        "condition": condition_from_config(cfg, condition_name),
        "population": [agent.state_dict() for agent in population],
    }
    torch.save(checkpoint, out_dir / "population.pt")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"condition": condition_name, "fold": fold, "seed": seed}, f, indent=2)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Experiment 5 population models.")
    parser.add_argument("--config", default="configs/exp5_population_duplex_quick.yaml")
    parser.add_argument("--condition", default="duplex_direct")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Run all configured seeds/folds for the condition.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.all:
        for fold in cfg["training"]["folds"]:
            for seed in cfg["training"]["seeds"]:
                train_condition(cfg, args.condition, int(fold), int(seed))
    else:
        train_condition(cfg, args.condition, args.fold, args.seed)


if __name__ == "__main__":
    main()

