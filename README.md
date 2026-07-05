# Emergent Communication in Limited-Visibility Resource Collection

This project implements a reinforcement learning experiment where multiple agents operate in a partially observable grid world and learn whether and how to communicate under bandwidth constraints.

## Key Components

- **GridWorld environment** with limited agent vision (radius 1)
- **Independent Q-learning baseline** (no communication)
- **Communication-enabled agents** with a small discrete vocabulary
- **Metrics & plotting utilities** for analyzing behavior, communication entropy, and task efficiency

## Getting Started

1. Create a Python environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the baseline training (no communication):

```bash
python -m experiments.train_baseline --episodes 200 --plot
```

The run will save a trained model at `results/baseline/model.pkl`.

3. Run the communication-enabled training:

```bash
python -m experiments.train_comm --episodes 200 --plot
```

The run will save a trained model at `results/comm/model.pkl`.

4. Run inference using a saved model:

Text rendering (console):

```bash
python -m experiments.inference --model results/comm/model.pkl --episodes 1 --render text
```

Graphical rendering (pygame):

```bash
python -m experiments.inference --model results/comm/model.pkl --episodes 1 --render pygame
```

If `pygame` is not installed, it will fall back to text rendering.

## Notes

- The environment is intentionally simple to focus on communication dynamics.
- This repository is set up to be extended for additional experiments described in `docs/experiment_plan.md`.

## Experiment 5

The population-level duplex communication experiment lives in `exp5_population_duplex/` with configs in `configs/`.

Quick smoke test:

```bash
python -m exp5_population_duplex.training --config configs/exp5_population_duplex_quick.yaml --condition duplex_direct --fold 0 --seed 0
```

See `exp5_population_duplex/README.md` for the full runbook.
