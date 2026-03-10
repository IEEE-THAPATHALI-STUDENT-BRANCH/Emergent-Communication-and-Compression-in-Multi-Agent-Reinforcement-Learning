# Experiment Plan (Steps 1–7)

This document captures the plan and implementation status for the first 7 steps of the research experiment.

## 1. Define the Environment (Week 1)

- **Grid size**: 7×7
- **Agents**: 3
- **Resources**: configurable (default 2)
- **Vision radius**: 1 cell (agents see a 3×3 window centered on themselves)
- **Episode length**: default 50 steps

### Observations

Each agent receives:

- Own position (x, y)
- Nearby cells within vision radius (3×3 map)
- Resources within vision radius
- (when communication enabled) last received message from other agents

## 2. Define Agent Actions

Each agent can take one of:

- `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`

When communication is enabled, each agent additionally selects one token from a small vocabulary: `{0, 1, 2, 3}`

## 3. Design Reward Function

Rewards used in the implementation:

- +20 when collecting a resource
- -1 per step (time penalty)
- -5 on collision (two agents attempt to occupy same cell)
- -0.2 per message sent (communication cost)

## 4. Build the Baseline Model (No Communication)

Implemented as an **independent Q-learning** agent.

State representation includes:

- Agent position
- Visible resource positions (relative)
- Visible other agents (relative)

## 5. Add Communication Channel

When enabled, each agent selects a token each step.
Tokens are learned through reinforcement learning (no predefined semantics).
Bandwidth is limited to one token per agent per step.

## 6. Implement Communication State

The agent observation now includes:

- `last_received_message` (the token received from other agents in the previous step)

## 7. Train the Communication Model

A training script is provided to train agents with communication enabled.

---

### Next Steps (Beyond Step 7)

- Run controlled experiments for different communication constraints (Steps 8–12)
- Add analysis scripts for entropy / compression metrics (Step 9)
- Produce plots and behavior analysis (Steps 10–11)
- Draft the paper structure (Step 13)
