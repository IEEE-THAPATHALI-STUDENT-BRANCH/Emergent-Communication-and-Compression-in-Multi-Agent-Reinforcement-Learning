"""
Multi-agent referential game environment for emergent communication research.

This module defines `EmergentCommunicationEnv`, a minimal yet extensible
environment where a Speaker and a Listener cooperate to identify a target
object among distractors using a discrete symbolic message.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class WorldConfig:
    """Configuration of the object world.

    Attributes:
        num_objects: Number of candidate objects per episode.
        colors: List of possible color names (used only for readability).
        shapes: List of possible shape names.
        sizes: List of possible size names.
    """

    num_objects: int = 5
    colors: Tuple[str, ...] = ("red", "blue", "green")
    shapes: Tuple[str, ...] = ("square", "circle", "triangle")
    sizes: Tuple[str, ...] = ("small", "large")

    @property
    def color_dim(self) -> int:
        return len(self.colors)

    @property
    def shape_dim(self) -> int:
        return len(self.shapes)

    @property
    def size_dim(self) -> int:
        return len(self.sizes)


@dataclass
class ChannelConfig:
    """Configuration of the communication channel."""

    vocabulary_size: int = 10
    message_length: int = 3


class EmergentCommunicationEnv:
    """One-step referential game environment.

    At the beginning of each episode:
      * A set of N objects is sampled, each with (color, shape, size) attributes.
      * One object is sampled as the target.
      * The Speaker observes only the target object.
      * The Listener observes the list of all objects.

    A single environment step:
      1. The Speaker sends a discrete message (sequence of tokens).
      2. The Listener selects an object index.
      3. The environment returns reward=1 if the selection matches the target,
         otherwise 0. The episode terminates immediately (one-step episode).

    Observations are represented as integer tensors suitable for neural agents.
    """

    def __init__(
        self,
        world_config: WorldConfig | None = None,
        channel_config: ChannelConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.world = world_config or WorldConfig()
        self.channel = channel_config or ChannelConfig()
        self.device = device or torch.device("cpu")

        # Internal state
        self._objects: torch.Tensor | None = None  # (num_objects, 3) int64
        self._target_index: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment and sample a new world.

        Returns:
            A dict with:
                "speaker_obs": tensor of shape (3,) containing the target object
                               attributes as integer codes.
                "listener_obs": tensor of shape (num_objects, 3) containing all
                                objects' attribute codes.
        """
        self._sample_world()

        assert self._objects is not None
        assert self._target_index is not None

        speaker_obs = self._objects[self._target_index].clone()
        listener_obs = self._objects.clone()

        return {
            "speaker_obs": speaker_obs.to(self.device),
            "listener_obs": listener_obs.to(self.device),
        }

    def step(
        self,
        message: torch.Tensor,
        action: int,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """Evaluate a (message, action) pair.

        Args:
            message: Tensor of shape (message_length,) with integer tokens.
                     The current environment is message-agnostic; it simply
                     passes it through for logging. Learning agents will
                     exploit this channel.
            action: Integer index in [0, num_objects-1] chosen by the Listener.

        Returns:
            next_state: A new observation dict for the next episode. Since
                episodes are one-step, this is the reset() observation.
            reward: 1.0 if the guessed index matches the target, else 0.0.
            done: Always True (one-step episode).
            info: Dict with diagnostic information (target index, objects, etc.).
        """
        if self._objects is None or self._target_index is None:
            raise RuntimeError("Env.step() called before reset().")

        correct = int(action == self._target_index)
        reward = float(correct)

        info = {
            "correct": bool(correct),
            "target_index": int(self._target_index),
            "objects": self._objects.clone().cpu().numpy(),
            "message": message.detach().cpu().numpy(),
            "world_config": asdict(self.world),
            "channel_config": asdict(self.channel),
        }

        # One-step episode: immediately reset for the next one.
        next_state = self.reset()
        done = True
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sample_world(self) -> None:
        """Sample a new set of objects and a target index."""
        num_objects = self.world.num_objects

        colors = np.random.randint(0, self.world.color_dim, size=(num_objects, 1))
        shapes = np.random.randint(0, self.world.shape_dim, size=(num_objects, 1))
        sizes = np.random.randint(0, self.world.size_dim, size=(num_objects, 1))

        objects_np = np.concatenate([colors, shapes, sizes], axis=1).astype(np.int64)
        self._objects = torch.from_numpy(objects_np)

        self._target_index = int(np.random.randint(0, num_objects))

