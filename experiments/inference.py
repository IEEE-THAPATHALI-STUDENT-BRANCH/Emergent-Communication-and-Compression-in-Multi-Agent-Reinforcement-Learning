"""Run an inference episode using a saved Q-learning model (pickle file).

This script loads a saved model (Q-tables for each agent + env config) and runs
one or more episodes while rendering the grid world.
"""

import argparse
import os
import pickle
import time

from env.grid_world import GridWorldEnv, Action
from agents.q_agent import QAgent

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore
    _HAS_PYGAME = False


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _init_pygame(env: GridWorldEnv, cell_size: int = 50, margin: int = 2):
    pygame.init()
    window_size = (env.width * cell_size + margin * 2, env.height * cell_size + margin * 2)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("GridWorld Inference")
    return screen, cell_size, margin


def _draw_pygame(env: GridWorldEnv, screen, cell_size: int, margin: int):
    background = (30, 30, 30)
    resource_color = (200, 180, 0)
    agent_colors = [(200, 80, 80), (80, 200, 80), (80, 120, 220), (200, 120, 200), (120, 200, 200)]

    screen.fill(background)

    for x in range(env.width):
        for y in range(env.height):
            rect = pygame.Rect(
                margin + x * cell_size,
                margin + y * cell_size,
                cell_size - margin,
                cell_size - margin,
            )
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)

    for rx, ry in env.resources:
        rect = pygame.Rect(
            margin + rx * cell_size,
            margin + ry * cell_size,
            cell_size - margin,
            cell_size - margin,
        )
        pygame.draw.rect(screen, resource_color, rect)

    for agent in env.agents:
        color = agent_colors[agent.id % len(agent_colors)]
        rect = pygame.Rect(
            margin + agent.pos[0] * cell_size,
            margin + agent.pos[1] * cell_size,
            cell_size - margin,
            cell_size - margin,
        )
        pygame.draw.rect(screen, color, rect)
        label = pygame.font.SysFont(None, 24).render(str(agent.id), True, (255, 255, 255))
        screen.blit(label, (rect.x + 5, rect.y + 5))

    if env.communication_enabled:
        text = " ".join(
            f"A{a.id}:{a.last_message if a.last_message is not None else '-'}" for a in env.agents
        )
        label = pygame.font.SysFont(None, 20).render(text, True, (230, 230, 230))
        screen.blit(label, (margin, env.height * cell_size + margin - 20))

    pygame.display.flip()


def run_inference(
    model_path: str,
    episodes: int = 1,
    render_delay: float = 0.2,
    render_mode: str = "text",
):
    model = load_model(model_path)
    env_kwargs = model["env_kwargs"]
    q_tables = model["q_tables"]

    env = GridWorldEnv(**env_kwargs)
    agents = {
        int(aid): QAgent(
            agent_id=int(aid),
            actions=tuple(Action),
            epsilon=0.0,
            comm_vocab_size=env.communication_vocab_size if env.communication_enabled else None,
            q_table=q_tables[str(aid)],
        )
        for aid in q_tables
    }

    use_pygame = render_mode == "pygame" and _HAS_PYGAME
    if render_mode == "pygame" and not _HAS_PYGAME:
        print("pygame not installed; falling back to text rendering.")

    screen = None
    cell_size = 40
    margin = 2
    if use_pygame:
        screen, cell_size, margin = _init_pygame(env)

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"--- Episode {ep} ---")
        if use_pygame:
            _draw_pygame(env, screen, cell_size, margin)
        else:
            env.render()

        while not done:
            for event in pygame.event.get() if use_pygame else []:
                if event.type == pygame.QUIT:
                    done = True

            actions = {}
            messages = {}

            for aid, agent in agents.items():
                move_idx, msg, _ = agent.act(obs[aid], explore=False)
                actions[aid] = Action(move_idx)
                if env.communication_enabled:
                    token = msg if msg is not None else 0
                    messages[aid] = token

            next_obs, rewards, done, info = env.step(
                actions, messages if env.communication_enabled else None
            )
            obs = next_obs
            total_reward += sum(rewards.values())
            step += 1

            if use_pygame:
                _draw_pygame(env, screen, cell_size, margin)
            else:
                env.render()

            time.sleep(render_delay)

        print(
            f"Episode {ep} finished in {step} steps, total_reward={total_reward:.1f}, collected={len(info['collected'])}"
        )

    if use_pygame:
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Run inference with a saved Q-learning model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model (.pkl)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.25, help="Delay (seconds) between rendered frames")
    parser.add_argument(
        "--render",
        type=str,
        default="text",
        choices=["text", "pygame"],
        help="Render mode: text (console) or pygame (graphical)",
    )
    args = parser.parse_args()

    run_inference(
        args.model,
        episodes=args.episodes,
        render_delay=args.delay,
        render_mode=args.render,
    )


if __name__ == "__main__":
    main()
