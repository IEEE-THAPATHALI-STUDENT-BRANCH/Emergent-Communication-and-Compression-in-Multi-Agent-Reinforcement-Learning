"""
Pygame visualization for the referential game.

Shows:
- the set of objects (color/shape/size)
- the target object (outlined)
- the listener's chosen object (green if correct, red if incorrect)
- the message tokens sent by the speaker

Usage:
    python -m referential_game.visualize --config referential_game/config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pygame
import torch

from .evaluate import load_checkpoints  # re-use loader
from .train import build_env_and_agents, load_config


COLOR_PALETTE = [
    (220, 50, 50),   # red
    (50, 120, 220),  # blue
    (50, 180, 80),   # green
    (230, 200, 60),  # extra if needed
]


def draw_object(screen, obj, x, y, size_px, is_target, is_choice, correct):
    """Draw a single object as a colored shape."""
    color_idx, shape_idx, size_idx = [int(v) for v in obj.tolist()]
    base_color = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]

    # Size scaling
    scale = 0.6 if size_idx == 0 else 0.9
    r = int(size_px * scale / 2)

    cx = x + size_px // 2
    cy = y + size_px // 2

    # Shape: 0=square, 1=circle, 2=triangle
    if shape_idx == 0:
        rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
        pygame.draw.rect(screen, base_color, rect)
    elif shape_idx == 1:
        pygame.draw.circle(screen, base_color, (cx, cy), r)
    else:
        points = [
            (cx, cy - r),
            (cx - r, cy + r),
            (cx + r, cy + r),
        ]
        pygame.draw.polygon(screen, base_color, points)

    # Target outline
    if is_target:
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(x + 2, y + 2, size_px - 4, size_px - 4), 2)

    # Choice outline
    if is_choice:
        color = (0, 240, 0) if correct else (240, 0, 0)
        pygame.draw.rect(screen, color, pygame.Rect(x + 5, y + 5, size_px - 10, size_px - 10), 3)


def visualize(config_path: str) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, speaker, listener = build_env_and_agents(cfg, device)
    load_checkpoints(cfg, speaker, listener, device)

    pygame.init()
    num_objects = env.world.num_objects
    cols = min(num_objects, 5)
    rows = (num_objects + cols - 1) // cols
    cell_size = 80
    margin = 20
    panel_height = 80

    width = cols * cell_size + margin * 2
    height = rows * cell_size + margin * 2 + panel_height

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Referential Game Visualization")
    font = pygame.font.SysFont(None, 24)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Run one episode with greedy policies
        with torch.no_grad():
            obs = env.reset()
            target = obs["speaker_obs"]
            objects = obs["listener_obs"]

            msg_logits, _ = speaker.forward(target.unsqueeze(0))
            msg_logits = msg_logits.squeeze(0)
            message = msg_logits.argmax(dim=-1)

            scores = listener.forward(objects, message)
            choice = int(scores.argmax().item())

            _, reward, _, info = env.step(message, choice)

        screen.fill((30, 30, 30))

        # Draw objects grid
        for idx, obj in enumerate(objects):
            row = idx // cols
            col = idx % cols
            x = margin + col * cell_size
            y = margin + row * cell_size
            is_target = (idx == info["target_index"])
            is_choice = (idx == choice)
            correct = bool(reward > 0.5)
            draw_object(screen, obj.cpu(), x, y, cell_size, is_target, is_choice, correct)

        # Draw message and status text
        msg_str = "Message: " + " ".join(str(int(t)) for t in message.cpu().tolist())
        status_str = f"Choice: {choice} | Target: {info['target_index']} | Reward: {reward:.0f}"

        msg_surf = font.render(msg_str, True, (230, 230, 230))
        status_surf = font.render(status_str, True, (230, 230, 230))
        screen.blit(msg_surf, (margin, rows * cell_size + margin))
        screen.blit(status_surf, (margin, rows * cell_size + margin + 30))

        pygame.display.flip()
        clock.tick(1)  # one episode per second

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize trained referential-game agents with pygame.")
    parser.add_argument(
        "--config",
        type=str,
        default="referential_game/config.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    visualize(args.config)


if __name__ == "__main__":
    main()

