"""Analysis, statistics, plots, and tables for Experiment 5."""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .messages import FIRST_CONTENT_TOKEN
from .world import Object, WorldSpec, all_objects, overlap_count


def mean_ci(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    m = mean(values)
    sd = pstdev(values) if len(values) > 1 else 0.0
    return {"mean": m, "std": sd, "ci95": 1.96 * sd / sqrt(len(values)), "n": len(values)}


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def token_entropy(messages: Iterable[str], vocabulary_size: int) -> float:
    counts = np.zeros(vocabulary_size, dtype=float)
    total = 0
    for message in messages:
        for tok in message.split():
            token = int(tok)
            if token >= FIRST_CONTENT_TOKEN:
                counts[token] += 1
                total += 1
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def mutual_information(tokens: Sequence[int], attrs: Sequence[int], vocab: int, attr_dim: int) -> float:
    joint = np.zeros((vocab, attr_dim), dtype=float)
    for token, attr in zip(tokens, attrs):
        if token >= FIRST_CONTENT_TOKEN:
            joint[token, attr] += 1
    total = joint.sum()
    if total == 0:
        return 0.0
    p_joint = joint / total
    p_t = p_joint.sum(axis=1, keepdims=True)
    p_a = p_joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_joint / (p_t * p_a)
        ratio[p_joint == 0] = 1.0
        return float((p_joint * np.log2(ratio)).sum())


def _structured_edge(a: Object, b: Object) -> bool:
    return overlap_count(a, b) >= 1


def conflict_graph_stats(spec: WorldSpec) -> Dict[str, float | int]:
    nodes = all_objects(spec)
    adjacency = {obj: set() for obj in nodes}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if _structured_edge(a, b):
                adjacency[a].add(b)
                adjacency[b].add(a)
    edge_count = sum(len(v) for v in adjacency.values()) // 2
    possible = len(nodes) * (len(nodes) - 1) // 2

    ordered = sorted(nodes, key=lambda obj: len(adjacency[obj]), reverse=True)
    colors: Dict[Object, int] = {}
    for node in ordered:
        used = {colors[nbr] for nbr in adjacency[node] if nbr in colors}
        color = 0
        while color in used:
            color += 1
        colors[node] = color

    best_clique = 0

    def bronk(r: set[Object], p: set[Object], x: set[Object]) -> None:
        nonlocal best_clique
        if not p and not x:
            best_clique = max(best_clique, len(r))
            return
        if len(r) + len(p) <= best_clique:
            return
        pivot_candidates = p | x
        pivot = max(pivot_candidates, key=lambda obj: len(adjacency[obj])) if pivot_candidates else None
        candidates = p - (adjacency[pivot] if pivot is not None else set())
        for v in list(candidates):
            bronk(r | {v}, p & adjacency[v], x & adjacency[v])
            p.remove(v)
            x.add(v)

    bronk(set(), set(nodes), set())
    return {
        "nodes": len(nodes),
        "edges": edge_count,
        "density": edge_count / possible,
        "greedy_coloring_upper_bound": max(colors.values()) + 1,
        "maximum_clique_lower_bound": best_clique,
    }


def make_learning_curve_plot(training_files: Sequence[str], output: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    for path in training_files:
        rows = read_csv_rows(path)
        xs = [int(r["batch"]) for r in rows]
        ys = [float(r["success"]) for r in rows]
        label = Path(path).parents[2].name if len(Path(path).parents) > 2 else Path(path).stem
        plt.plot(xs, ys, label=label, alpha=0.8)
    plt.xlabel("Training batch")
    plt.ylabel("Task success")
    plt.title("Experiment 5 learning curves (raw per-seed traces)")
    plt.legend()
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def make_crossplay_heatmap(matrix_csv: str | Path, output: str | Path, title: str) -> None:
    matrix = np.loadtxt(matrix_csv, delimiter=",")
    plt.figure(figsize=(5.5, 4.8))
    im = plt.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, label="Task success")
    plt.xlabel("Receiver agent")
    plt.ylabel("Informant agent")
    plt.title(title)
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def generate_tables(cfg: Dict, results_dir: str | Path, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "table1_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump({"world": cfg["world"], "model": cfg["model"], "training": cfg["training"]}, f, indent=2)

    metric_rows = []
    for path in Path(results_dir).rglob("metrics_*.json"):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["source"] = str(path)
        metric_rows.append(row)

    if metric_rows:
        keys = sorted({key for row in metric_rows for key in row})
        with open(output / "table2_main_condition_comparison.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metric_rows)

    graph = conflict_graph_stats(WorldSpec(**{k: int(v) for k, v in cfg["world"].items()}))
    with open(output / "table10_conflict_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Experiment 5 analyses.")
    parser.add_argument("--config", default="configs/exp5_population_duplex_quick.yaml")
    parser.add_argument("--results", default="results/exp5_population_duplex")
    parser.add_argument("--output", default="results/exp5_population_duplex/analysis")
    parser.add_argument("--learning_files", nargs="*", default=[])
    parser.add_argument("--crossplay_csv", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.learning_files:
        make_learning_curve_plot(args.learning_files, output / "fig1_learning_curves.png")
    if args.crossplay_csv:
        make_crossplay_heatmap(args.crossplay_csv, output / "fig2_crossplay_heatmap.png", "Experiment 5 cross-play matrix")
    generate_tables(cfg, args.results, output / "tables")

    graph = conflict_graph_stats(WorldSpec(**{k: int(v) for k, v in cfg["world"].items()}))
    with open(output / "conflict_graph_summary.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"Wrote analysis artifacts to {output}")


if __name__ == "__main__":
    main()

