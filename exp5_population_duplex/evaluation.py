"""Evaluation utilities and CLI for Experiment 5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from .agents import ModelConfig, build_population, empty_message, message_content_length, require_torch, sample_message
from .config import condition_from_config, load_config
from .messages import apply_content_noise, empty_feedback as make_empty_feedback
from .training import _tensor_object, _tensor_world
from .world import CandidateWorld, WorldSpec, fixed_eval_worlds


def _torch_imports():
    require_torch()
    import torch

    return torch


def load_population(checkpoint_path: str | Path, device):
    torch = _torch_imports()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**checkpoint["model_config"])
    population = build_population(model_cfg, len(checkpoint["population"]), device)
    for agent, state in zip(population, checkpoint["population"]):
        agent.load_state_dict(state)
        agent.eval()
    return population, checkpoint["condition"]


def _noise(message, vocabulary_size: int, p_noise: float, rng: random.Random, device):
    torch = _torch_imports()
    noisy, stats = apply_content_noise(message.detach().cpu().tolist(), vocabulary_size, p_noise, rng)
    return torch.tensor(noisy, dtype=torch.long, device=device), stats


def evaluate_episode(
    population,
    world: CandidateWorld,
    condition: Dict[str, Any],
    informant_id: int,
    receiver_id: int,
    rng: random.Random,
    device,
    p_noise_override: Optional[float] = None,
    feedback_ablation: str = "none",
    feedback_override=None,
):
    torch = _torch_imports()
    informant = population[informant_id]
    receiver = population[receiver_id]
    kind = condition["kind"]
    vocab_size = int(condition["vocabulary_size"])
    l_initial = int(condition.get("l_initial", 0))
    l_feedback = int(condition.get("l_feedback", 0))
    l_response = int(condition.get("l_response", 0))
    p_noise = float(condition.get("p_noise", 0.0) if p_noise_override is None else p_noise_override)
    lambda_cost = float(condition.get("lambda_cost", 0.0))

    target = _tensor_object(world.target, device)
    candidates = _tensor_world(world, device)
    noise_content = 0
    noise_corrupted = 0

    with torch.no_grad():
        if kind == "no_comm":
            m_initial = empty_message(0, device)
        else:
            logits, _ = informant.initial_logits(target, l_initial)
            m_initial = sample_message(logits, vocab_size, greedy=True)["message"]
        m_initial_rx, stats = _noise(m_initial, vocab_size, p_noise, rng, device)
        noise_content += stats.content_tokens
        noise_corrupted += stats.corrupted_tokens

        if kind == "duplex":
            if feedback_override is not None:
                q_feedback = feedback_override
            elif feedback_ablation == "empty":
                q_feedback = torch.tensor(make_empty_feedback(l_feedback), dtype=torch.long, device=device)
            else:
                logits, _ = receiver.feedback_logits(candidates, m_initial_rx, l_feedback)
                q_feedback = sample_message(logits, vocab_size, greedy=True)["message"]
        else:
            q_feedback = empty_message(l_feedback, device)

        q_feedback_rx, stats = _noise(q_feedback, vocab_size, p_noise, rng, device)
        noise_content += stats.content_tokens
        noise_corrupted += stats.corrupted_tokens

        if kind == "duplex":
            logits, _ = informant.response_logits(target, m_initial, q_feedback_rx, l_response)
            m_response = sample_message(logits, vocab_size, greedy=True)["message"]
        else:
            m_response = empty_message(l_response, device)
        m_response_rx, stats = _noise(m_response, vocab_size, p_noise, rng, device)
        noise_content += stats.content_tokens
        noise_corrupted += stats.corrupted_tokens

        logits, _ = receiver.selection_logits(candidates, m_initial_rx, q_feedback, m_response_rx)
        action = int(torch.argmax(logits).item())

    lengths = {
        "initial_length": message_content_length(m_initial),
        "feedback_length": message_content_length(q_feedback),
        "response_length": message_content_length(m_response),
    }
    total_length = sum(lengths.values())
    success = float(action == world.target_index)
    return {
        "success": success,
        "return": success - lambda_cost * total_length,
        "action": action,
        "target_index": world.target_index,
        "avg_overlap": world.avg_overlap,
        "total_length": total_length,
        **lengths,
        "noise_content_tokens": noise_content,
        "noise_corrupted_tokens": noise_corrupted,
        "noise_corruption_rate": 0.0 if noise_content == 0 else noise_corrupted / noise_content,
        "initial_message": " ".join(map(str, m_initial.detach().cpu().tolist())),
        "feedback_message": " ".join(map(str, q_feedback.detach().cpu().tolist())),
        "response_message": " ".join(map(str, m_response.detach().cpu().tolist())),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [
        "success",
        "return",
        "total_length",
        "initial_length",
        "feedback_length",
        "response_length",
        "noise_corruption_rate",
    ]
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def evaluate_checkpoint(
    cfg: Dict[str, Any],
    checkpoint_path: str | Path,
    fold: int,
    seed: int,
    split: str,
    output_dir: str | Path,
    p_noise: Optional[float] = None,
    feedback_ablation: str = "none",
) -> Path:
    torch = _torch_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    population, condition = load_population(checkpoint_path, device)
    rng = random.Random(seed + 4242)
    spec = WorldSpec(**{k: int(v) for k, v in cfg["world"].items()})
    worlds = fixed_eval_worlds(spec, fold, split, seed, int(cfg["evaluation"]["episodes"]))

    rows: List[Dict[str, Any]] = []
    pair_schedule = [tuple(rng.sample(range(len(population)), 2)) for _ in worlds]
    feedback_overrides = None

    if feedback_ablation == "shuffled":
        feedback_rows = []
        for world, (informant_id, receiver_id) in zip(worlds, pair_schedule):
            feedback_rows.append(
                evaluate_episode(
                    population,
                    world,
                    condition,
                    informant_id,
                    receiver_id,
                    rng,
                    device,
                    p_noise_override=p_noise,
                    feedback_ablation="none",
                )["feedback_message"]
            )
        shuffled = list(feedback_rows)
        rng.shuffle(shuffled)
        feedback_overrides = [
            torch.tensor([int(tok) for tok in item.split()] if item else [], dtype=torch.long, device=device)
            for item in shuffled
        ]

    for episode, world in enumerate(worlds):
        informant_id, receiver_id = pair_schedule[episode]
        row = evaluate_episode(
            population,
            world,
            condition,
            informant_id,
            receiver_id,
            rng,
            device,
            p_noise_override=p_noise,
            feedback_ablation=feedback_ablation,
            feedback_override=feedback_overrides[episode] if feedback_overrides is not None else None,
        )
        row.update(
            {
                "episode": episode,
                "seed": seed,
                "fold": fold,
                "split": split,
                "informant_id": informant_id,
                "receiver_id": receiver_id,
                "feedback_ablation": feedback_ablation,
                "p_noise": condition.get("p_noise", 0.0) if p_noise is None else p_noise,
            }
        )
        rows.append(row)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / f"evaluation_{split}_{feedback_ablation}.csv"
    with open(records_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(output / f"metrics_{split}_{feedback_ablation}.json", "w", encoding="utf-8") as f:
        json.dump(summarize(rows), f, indent=2)
    return records_path


def crossplay_matrix(
    cfg: Dict[str, Any],
    checkpoint_path: str | Path,
    fold: int,
    seed: int,
    split: str,
    output_dir: str | Path,
) -> Path:
    torch = _torch_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    population, condition = load_population(checkpoint_path, device)
    rng = random.Random(seed + 999)
    spec = WorldSpec(**{k: int(v) for k, v in cfg["world"].items()})
    worlds = fixed_eval_worlds(spec, fold, split, seed, int(cfg["evaluation"]["episodes"]))

    matrix: List[List[float]] = []
    for i in range(len(population)):
        row_values = []
        for j in range(len(population)):
            rows = [
                evaluate_episode(population, world, condition, i, j, rng, device)
                for world in worlds
            ]
            row_values.append(summarize(rows)["success"])
        matrix.append(row_values)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"crossplay_{split}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 5 checkpoints.")
    parser.add_argument("--config", default="configs/exp5_population_duplex_quick.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["id", "ood", "mixed"], default="id")
    parser.add_argument("--output", default="results/exp5_population_duplex/eval")
    parser.add_argument("--p_noise", type=float, default=None)
    parser.add_argument("--feedback_ablation", choices=["none", "empty", "shuffled"], default="none")
    parser.add_argument("--crossplay", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.crossplay:
        path = crossplay_matrix(cfg, args.checkpoint, args.fold, args.seed, args.split, args.output)
    else:
        path = evaluate_checkpoint(
            cfg,
            args.checkpoint,
            args.fold,
            args.seed,
            args.split,
            args.output,
            p_noise=args.p_noise,
            feedback_ablation=args.feedback_ablation,
        )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
