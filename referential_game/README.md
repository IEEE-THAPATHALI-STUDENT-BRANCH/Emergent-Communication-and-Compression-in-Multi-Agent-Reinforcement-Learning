# Referential Game: Emergent Communication

This folder contains a minimal, research-oriented **referential game** setup for studying **emergent communication** between two agents:

- **Speaker** observes only the *target object*.
- **Listener** observes a *set of candidate objects* and receives the Speaker’s message.
- Both receive reward \(1\) if the Listener picks the target, else \(0\).

The message is a discrete sequence of tokens with configurable:

- **`vocabulary_size`**
- **`message_length`**

## Setup

Create/activate your Python environment, then install dependencies.

### Install PyTorch (CPU)

If you do not have PyTorch installed, use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Install pygame (optional, for visualization)

On macOS, `pygame` may require SDL headers. The easiest option is:

```bash
pip install pygame-ce
```

(`pygame-ce` installs an import-compatible `pygame`.)

## Configuration

Edit `referential_game/config.json` to change:

- **World**: `num_objects`, attribute sets (`colors`, `shapes`, `sizes`)
- **Channel**: `vocabulary_size`, `message_length`
- **Training**: `episodes`, `batch_size`, learning rates, etc.
- **Checkpoints**: `training.checkpoint_dir`
- **Evaluation**: `evaluation.episodes`

## Training (communication)

Trains **Speaker + Listener** with REINFORCE and saves checkpoints to:

- `referential_game/checkpoints/speaker.pt`
- `referential_game/checkpoints/listener.pt`

```bash
python -m referential_game.train --config referential_game/config.json
```

## Baseline vs communication comparison plot

Trains two settings and plots them on the same axes:

- **Baseline**: Listener gets a constant dummy message (no information).
- **Communication**: Learned Speaker message.

Output:

- `referential_game/results/learning_curve_baseline_vs_comm.png`

```bash
python -m referential_game.train_compare --config referential_game/config.json
```

## Evaluation + language analysis plots

Loads the trained checkpoints and runs evaluation episodes without learning.

Printed metrics include:

- success rate
- token entropy (overall + per position)
- mutual information between tokens and attributes (simple estimate)

Outputs plots to `referential_game/results/`:

- `token_frequencies.png`
- `token_success_rates.png`
- `token_color_heatmap_pos{0,1,2}.png`
- `token_shape_heatmap_pos{0,1,2}.png`
- `token_size_heatmap_pos{0,1,2}.png`

```bash
python -m referential_game.evaluate --config referential_game/config.json
```

## Pygame visualization (live episodes)

Shows:

- candidate objects (drawn as colored shapes sized by the size attribute)
- target object (white outline)
- listener choice (green outline if correct, red if wrong)
- message tokens sent by the Speaker

```bash
python -m referential_game.visualize --config referential_game/config.json
```

## Communication constraints sweep (vocab size × message length)

Runs training for each (vocabulary size, message length) pair and saves a heatmap:

- `referential_game/results/performance_vs_channel_heatmap.png`

```bash
python -m referential_game.train_sweep \
  --config referential_game/config.json \
  --vocab_sizes 4 8 16 \
  --message_lengths 1 2 3
```

## Common outputs

- **Checkpoints**: `referential_game/checkpoints/`
- **Plots / analysis**: `referential_game/results/`

