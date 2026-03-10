"""Independent Q-learning agent for the grid world environment."""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy as np

from env.grid_world import Action


class QAgent:
    def __init__(
        self,
        agent_id: int,
        actions: Tuple[Action, ...],
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
        q_table: Optional[Dict] = None,
        comm_vocab_size: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.comm_vocab_size = comm_vocab_size

        self.q: Dict[Tuple, np.ndarray] = q_table or {}

    def _state_key(self, observation: dict) -> Tuple:
        """Convert observation to a hashable state key.

        State includes:
        - agent position
        - visible resources (relative)
        - visible agents (relative)
        - last received message (if present)
        """

        self_pos = tuple(observation["self_pos"])
        visible = tuple(
            sorted(
                (
                    (cell["pos"][0] - self_pos[0], cell["pos"][1] - self_pos[1]),
                    int(cell["resource"]),
                    int(cell["agent"]),
                )
                for cell in observation["visible"]
            )
        )
        last_msg = observation.get("last_message")
        return (self_pos, visible, last_msg)

    def _action_space_size(self) -> int:
        if self.comm_vocab_size is None:
            return len(self.actions)
        return len(self.actions) * self.comm_vocab_size

    def _decode_action(self, index: int) -> Tuple[int, Optional[int]]:
        """Decode joint action index into (move_action, message_token)."""
        if self.comm_vocab_size is None:
            return index, None
        move_idx = index // self.comm_vocab_size
        msg = index % self.comm_vocab_size
        return move_idx, msg

    def _encode_action(self, move_idx: int, msg: Optional[int] = None) -> int:
        """Encode (move_action, message_token) into a single index."""
        if self.comm_vocab_size is None or msg is None:
            return move_idx
        return move_idx * self.comm_vocab_size + msg

    def act(self, observation: dict, explore: bool = True) -> Tuple[int, Optional[int], int]:
        state = self._state_key(observation)
        n_actions = self._action_space_size()
        if state not in self.q:
            self.q[state] = np.zeros(n_actions, dtype=float)

        if explore and self.rng.random() < self.epsilon:
            index = self.rng.randrange(n_actions)
        else:
            # choose randomly among tied best actions to avoid deterministic stuck behavior
            qvals = self.q[state]
            max_val = np.max(qvals)
            best = np.flatnonzero(np.isclose(qvals, max_val)).tolist()
            index = int(self.rng.choice(best))

        move_idx, msg = self._decode_action(index)
        return move_idx, msg, index

    def learn(
        self,
        observation: dict,
        action_index: int,
        reward: float,
        next_observation: dict,
        done: bool,
    ):
        state = self._state_key(observation)
        next_state = self._state_key(next_observation)
        n_actions = self._action_space_size()
        if next_state not in self.q:
            self.q[next_state] = np.zeros(n_actions, dtype=float)

        td_target = reward + (0 if done else self.gamma * np.max(self.q[next_state]))
        td_error = td_target - self.q[state][action_index]
        self.q[state][action_index] += self.lr * td_error

    def record_episode(self):
        """Optionally called after each episode for epsilon decay or logging."""
        pass
