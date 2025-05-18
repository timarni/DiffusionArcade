"""
eval_dqn_pong.py
Run a trained DQN Pong agent against the built-in CPU opponent.

Usage examples
--------------
# 100 headless games (default)
python3 eval_dqn_pong.py --model_path agents/2025-05-18_12-00-00/checkpoint300.pth

# 20 visible games
python3 eval_dqn_pong.py --model_path agents/2025-05-18_12-00-00/checkpoint300.pth --mode visible
"""
import argparse
import random
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
from ple import PLE
from ple.games.pong import Pong
from tqdm import tqdm

from config import Pong as cfg
from dqn import DQN

# -----------------------------------------------------------------------------#
#                          Low-level preprocessing                             #
# -----------------------------------------------------------------------------#
FRAME_STACK_SIZE = cfg["obss_stack_size"]
FRAME_SKIP       = 4
IMG_H            = cfg["input_height"]
IMG_W            = cfg["input_width"]

# -----------------------------------------------------------------------------#
#                          Game specifications                                 #
# -----------------------------------------------------------------------------#
MAX_SCORE = 21
CPU_SPEED, PLAYER_SPEED, BALL_SPEED = 0.25, 0.5, 0.75
POS, NEG = 1, -1
TICK, WIN, LOSS = 0, 0, 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frame_stack: deque[np.ndarray] = deque(maxlen=FRAME_STACK_SIZE)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize (H×W) and map 0-255 → 0-1 float32."""
    if frame.shape[:2] != (IMG_H, IMG_W):  # (H, W)
        frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32) / 255.0


@torch.no_grad()
def init_frame_stack(env) -> torch.Tensor:
    """Fill stack with first frame repeated once per new episode."""
    raw = env.getScreenGrayscale()
    proc = preprocess_frame(raw)
    for _ in range(FRAME_STACK_SIZE):
        frame_stack.append(proc)
    stacked = np.stack(frame_stack, axis=0)
    return torch.from_numpy(stacked).unsqueeze(0).to(device)  # [1, 4, H, W]


@torch.no_grad()
def preprocess(env) -> torch.Tensor:
    """Grab new frame, append to deque and return stacked tensor."""
    raw = env.getScreenGrayscale()
    proc = preprocess_frame(raw)
    frame_stack.append(proc)
    stacked = np.stack(frame_stack, axis=0)
    return torch.from_numpy(stacked).unsqueeze(0).to(device)  # [1, 4, H, W]


# -----------------------------------------------------------------------------#
#                                Test routine                                  #
# -----------------------------------------------------------------------------#
def run_games(model_path: str, n_games: int, display: bool, fps: int = 30) -> tuple[int, int]:
    """Run `n_games` and return (#wins, #losses)."""
    # ---------------- Environment ----------------
    game = Pong(
        width=IMG_W,
        height=IMG_H,
        MAX_SCORE=MAX_SCORE,
        cpu_speed_ratio=CPU_SPEED,
        players_speed_ratio=PLAYER_SPEED,
        ball_speed_ratio=BALL_SPEED,
    )
    env = PLE(
        game,
        fps=fps,
        display_screen=display,
        reward_values=dict(positive=POS, negative=NEG, tick=TICK, loss=LOSS, win=WIN),
    )
    env.init()
    actions = env.getActionSet()  # [K_UP, K_DOWN, None]

    # ---------------- Load network -------------
    dqn = DQN(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device)
    dqn.load_state_dict(ckpt)
    dqn.eval()  # switch to eval; no dropout / batch-norm
    dqn.current_eps = 0.0           # disable ε-greedy explicitly

    wins = losses = 0

    for ep in tqdm(range(n_games)):
        env.reset_game()
        frame_stack.clear()
        state = init_frame_stack(env)

        done = False
        skip_count = 0
        while not done:
            # act every FRAME_SKIP frames (like in training)
            if skip_count == 0:
                with torch.no_grad():
                    action_idx = dqn.act(state, exploit=True).item()
                action = actions[action_idx]

            reward = env.act(action)
            done = env.game_over()
            skip_count = (skip_count + 1) % FRAME_SKIP
            if skip_count == 0 or done:
                state = preprocess(env)

        # -------- episode finished -------------
        agent_score = game.score_counts["agent"]
        cpu_score   = game.score_counts["cpu"]
        if agent_score > cpu_score:
            wins += 1
        else:
            losses += 1

        print(
            f"[{ep + 1:>3}/{n_games}] "
            f"score agent:{agent_score} - cpu:{cpu_score}   "
            f"{'WIN' if agent_score > cpu_score else 'LOSS'}"
        )

    return wins, losses


# -----------------------------------------------------------------------------#
#                                   CLI                                        #
# -----------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained DQN Pong agent.")
    p.add_argument(
        "--model_path",
        required=True,
        help="Path to a .pth checkpoint saved by deepqlearn_pong.py",
    )
    p.add_argument(
        "--mode",
        choices=["headless", "visible"],
        default="headless",
        help="Test mode: 'headless' = 100 games (no display); "
             "'visible' = 20 games (rendered).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "headless":
        n_games, display, fps = 100, False, 1000
    else:                             # visible
        n_games, display, fps = 20, True, 30

    print(
        f"Evaluating checkpoint '{args.model_path}' "
        f"({n_games} games, display={display}) – {datetime.now().isoformat(timespec='seconds')}"
    )
    wins, losses = run_games(args.model_path, n_games, display, fps)

    print("\n==========  FINAL RESULT  ==========")
    print(f"Games played : {n_games}")
    print(f"Agent won    : {wins}")
    print(f"Agent lost   : {losses}")
    print("====================================")


if __name__ == "__main__":
    main()
