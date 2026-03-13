# Experiments: Police–Civilian Bomb GridWorld

This folder contains the earlier multi-agent GridWorld experiments where agents learn to **find and defuse a bomb** with optional **discrete communication**.

## Scenario (current implementation)

- **Agents**: 1 police (agent id `0`) + `n_civilians` civilians (ids `1..`)
- **Bomb**: one bomb spawns each episode at a random location
- **Observations**: local vision (radius configurable); observations include:
  - `self_pos`
  - `visible` cells (bomb presence and agent presence)
  - `last_message` (if communication enabled)
- **Goal**: police defuses bomb by stepping onto it
- **Communication** (optional): agents can send discrete tokens each step

> Note: The environment and defaults live in `env/police_bomb_env.py`. If you change environment defaults, retrain models for fair comparisons.

## Setup

Install requirements from the repo root:

```bash
pip install -r requirements.txt
```

### Pygame (optional visualization)

On macOS, installing `pygame` from source may fail due to missing SDL headers. The easiest option is:

```bash
pip install pygame-ce
```

(`pygame-ce` provides an import-compatible `pygame`.)

## Train baseline (no communication)

Trains independent Q-learning agents **without** messages and saves:

- `results/baseline/model.pkl`
- `results/baseline/learning_curve.png`
- `results/baseline/success_rate.png`

```bash
python -m experiments.train_baseline --episodes 20000 --plot
```

Custom output directory:

```bash
python -m experiments.train_baseline --episodes 20000 --plot --output results/baseline_run1
```

## Train with communication

Trains independent Q-learning agents **with** messages and saves:

- `results/comm/model.pkl`
- `results/comm/learning_curve.png`
- `results/comm/success_rate.png`

```bash
python -m experiments.train_comm --episodes 20000 --plot
```

Custom output directory:

```bash
python -m experiments.train_comm --episodes 20000 --plot --output results/comm_run1
```

## Run inference (render the live episode)

Text rendering:

```bash
python -m experiments.inference --model results/comm/model.pkl --episodes 3 --render text --delay 0.2
```

Pygame rendering:

```bash
python -m experiments.inference --model results/comm/model.pkl --episodes 3 --render pygame --delay 0.2
```

## Plot baseline vs communication on the same scale

This runs both trainings (no-plot mode), then creates shared-axis plots:

- `results/combined_learning_curve.png`
- `results/combined_success_rate.png`

```bash
python -m experiments.compare_learning_curves --episodes 20000
```

## Sweep number of civilians (compare multiple settings)

Trains baseline and comm for a list of civilian counts and plots success curves:

- `results/compare_civilians_success.png`

```bash
python -m experiments.compare_civilians --episodes 20000 --civilians 1 2 3
```

## Tips

- **Longer training**: Q-learning in larger/harder variants can require 50k–100k episodes for clean curves.
- **Stuck policies**: `QAgent` uses epsilon-greedy exploration with decay; if you still see looping behavior, consider increasing exploration (epsilon) or environment penalties for wasted steps.

