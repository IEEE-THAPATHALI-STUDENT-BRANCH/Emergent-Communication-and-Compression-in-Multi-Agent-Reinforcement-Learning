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
python -m experiments.train_baseline --episodes 200
```

3. Run the communication-enabled training:

```bash
python -m experiments.train_comm --episodes 200
```

## Notes

- The environment is intentionally simple to focus on communication dynamics.
- This repository is set up to be extended for additional experiments described in `docs/experiment_plan.md`.
