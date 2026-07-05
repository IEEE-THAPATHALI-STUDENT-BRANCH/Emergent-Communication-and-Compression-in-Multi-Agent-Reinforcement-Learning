"""Police-civilian bomb defusal grid-world environment.

The environment is intentionally lightweight: it exposes the small API used by
the Q-learning experiments without requiring a Gym dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import random
from typing import Dict, Iterable, List, Optional, Tuple


Position = Tuple[int, int]


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


@dataclass
class AgentState:
    id: int
    pos: Position
    role: str
    last_message: Optional[int] = None


class GridWorldEnv:
    """Small partially observable grid world with one police-only objective."""

    def __init__(
        self,
        width: int = 7,
        height: int = 7,
        n_agents: Optional[int] = None,
        n_civilians: Optional[int] = None,
        n_resources: int = 1,
        vision_radius: int = 1,
        max_steps: int = 50,
        communication_enabled: bool = False,
        communication_vocab_size: int = 4,
        communication_cost: float = 0.2,
        step_penalty: float = -1.0,
        collision_penalty: float = -5.0,
        resource_reward: float = 20.0,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        if n_civilians is None:
            n_civilians = (n_agents - 1) if n_agents is not None else 2
        self.n_agents = n_agents if n_agents is not None else n_civilians + 1
        self.n_civilians = max(0, self.n_agents - 1)
        self.n_resources = n_resources
        self.vision_radius = vision_radius
        self.max_steps = max_steps
        self.communication_enabled = communication_enabled
        self.communication_vocab_size = communication_vocab_size
        self.communication_cost = communication_cost
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.resource_reward = resource_reward
        self.rng = random.Random(seed)

        self.agents: List[AgentState] = []
        self.bomb_pos: Position = (0, 0)
        self.bomb_defused = False
        self.steps = 0
        self._last_messages: Dict[int, Optional[int]] = {}
        self.reset()

    def reset(self) -> Dict[int, dict]:
        self.steps = 0
        self.bomb_defused = False

        positions = self._sample_unique_positions(self.n_agents + 1)
        self.bomb_pos = positions[0]
        self.agents = [
            AgentState(
                id=aid,
                pos=positions[aid + 1],
                role="police" if aid == 0 else "civilian",
            )
            for aid in range(self.n_agents)
        ]
        self._last_messages = {agent.id: None for agent in self.agents}
        return self._observations()

    def step(
        self,
        actions: Dict[int, Action],
        messages: Optional[Dict[int, int]] = None,
    ) -> Tuple[Dict[int, dict], Dict[int, float], bool, dict]:
        self.steps += 1
        rewards = {agent.id: self.step_penalty for agent in self.agents}

        desired = {
            agent.id: self._move(agent.pos, actions.get(agent.id, Action.STAY))
            for agent in self.agents
        }
        final_positions = {agent.id: desired[agent.id] for agent in self.agents}

        collided = self._collided_agents(desired)
        for agent in self.agents:
            if agent.id in collided:
                final_positions[agent.id] = agent.pos
                rewards[agent.id] += self.collision_penalty

        for agent in self.agents:
            agent.pos = final_positions[agent.id]

        if self.communication_enabled:
            messages = messages or {}
            for agent in self.agents:
                token = int(messages.get(agent.id, 0)) % self.communication_vocab_size
                agent.last_message = token
                self._last_messages[agent.id] = token
                rewards[agent.id] -= self.communication_cost

        if not self.bomb_defused and self.agents[0].pos == self.bomb_pos:
            self.bomb_defused = True
            for aid in rewards:
                rewards[aid] += self.resource_reward

        done = self.bomb_defused or self.steps >= self.max_steps
        info = {
            "bomb_defused": self.bomb_defused,
            "steps": self.steps,
            "collisions": sorted(collided),
            "collected": [self.bomb_pos] if self.bomb_defused else [],
        }
        return self._observations(), rewards, done, info

    def render(self) -> None:
        cells = [["." for _ in range(self.width)] for _ in range(self.height)]
        if not self.bomb_defused:
            bx, by = self.bomb_pos
            cells[by][bx] = "B"
        for agent in self.agents:
            x, y = agent.pos
            cells[y][x] = "P" if agent.role == "police" else str(agent.id)

        print(f"step={self.steps} bomb_defused={self.bomb_defused}")
        for row in cells:
            print(" ".join(row))
        if self.communication_enabled:
            msg = " ".join(
                f"A{agent.id}:{agent.last_message if agent.last_message is not None else '-'}"
                for agent in self.agents
            )
            print(f"messages {msg}")

    def _observations(self) -> Dict[int, dict]:
        return {agent.id: self._observation_for(agent) for agent in self.agents}

    def _observation_for(self, agent: AgentState) -> dict:
        ax, ay = agent.pos
        visible = []
        agent_positions = {other.pos: other.id for other in self.agents if other.id != agent.id}

        for x in range(ax - self.vision_radius, ax + self.vision_radius + 1):
            for y in range(ay - self.vision_radius, ay + self.vision_radius + 1):
                if not self._in_bounds((x, y)):
                    continue
                visible.append(
                    {
                        "pos": (x, y),
                        "resource": int((not self.bomb_defused) and (x, y) == self.bomb_pos),
                        "agent": int((x, y) in agent_positions),
                    }
                )

        obs = {
            "self_pos": agent.pos,
            "visible": visible,
        }
        if self.communication_enabled:
            obs["last_message"] = tuple(
                self._last_messages.get(other.id)
                for other in self.agents
                if other.id != agent.id
            )
        return obs

    def _sample_unique_positions(self, count: int) -> List[Position]:
        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        if count > len(all_positions):
            raise ValueError("Grid is too small for the requested number of objects.")
        return self.rng.sample(all_positions, count)

    def _move(self, pos: Position, action: Action) -> Position:
        x, y = pos
        if action == Action.UP:
            y -= 1
        elif action == Action.DOWN:
            y += 1
        elif action == Action.LEFT:
            x -= 1
        elif action == Action.RIGHT:
            x += 1
        return min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1)

    def _collided_agents(self, desired: Dict[int, Position]) -> set[int]:
        collided: set[int] = set()
        by_destination: Dict[Position, List[int]] = {}
        for aid, pos in desired.items():
            by_destination.setdefault(pos, []).append(aid)
        for aids in by_destination.values():
            if len(aids) > 1:
                collided.update(aids)

        current = {agent.id: agent.pos for agent in self.agents}
        for aid, pos in desired.items():
            for other_id, other_pos in desired.items():
                if aid >= other_id:
                    continue
                if pos == current[other_id] and other_pos == current[aid]:
                    collided.update((aid, other_id))
        return collided

    def _in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

