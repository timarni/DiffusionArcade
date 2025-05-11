import pygame
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image

# Game screen
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

# Paddle
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60  # You said you know this
PLAYER_X = 20
CPU_X = SCREEN_WIDTH - PLAYER_X - PADDLE_WIDTH

# Ball
BALL_SIZE = 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pygame.init()
surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))


def render_frame(row, frame_idx):
    surface.fill(BLACK)

    # Extract values
    player_y = row["player_y"]
    cpu_y = row["cpu_y"]
    ball_x = row["ball_x"]
    ball_y = row["ball_y"]

    # Draw player paddle
    pygame.draw.rect(surface, WHITE, pygame.Rect(PLAYER_X, player_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    # Draw CPU paddle
    pygame.draw.rect(surface, WHITE, pygame.Rect(CPU_X, cpu_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    # Draw ball (centered)
    pygame.draw.rect(surface, WHITE, pygame.Rect(ball_x - BALL_SIZE // 2, ball_y - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE))

    # Convert to image
    arr = pygame.surfarray.array3d(surface).transpose((1, 0, 2))  # shape (H, W, 3)
    img = Image.fromarray(arr)
    img.save(f"frames/frame_{frame_idx:05d}.png")


if __name__ == "__main__":
    # df = pd.read_csv("logs/pong_states_2025‑05‑07_10‑34‑43.csv")
    # df = pd.read_csv("logs/pong_states_2025‑05‑07_10‑40‑43.csv")
    df = pd.read_csv("logs/pong_states_2025‑05‑07_14‑46‑46.csv")
    os.makedirs("frames", exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # if i % 100 == 0:
        render_frame(row, i)

