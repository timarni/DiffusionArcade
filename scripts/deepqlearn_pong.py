"""
deepqlearn_pong.py
A minimal PLE‑based training loop for Pong that logs every game‑state change and uses a deepq network
"""

import csv
import os
import random
import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from config import Pong as config
from dqn import DQN, ReplayMemory, optimize
from collections import deque
from tqdm import tqdm
from datetime import datetime
from ple import PLE
from ple.games.pong import Pong


### Global Variables
FRAME_STACK_SIZE = config["obss_stack_size"]
frame_stack = deque(maxlen=FRAME_STACK_SIZE)
IMG_H, IMG_W = 84, 84 # Target input size for Pong conv‑net
MAX_STEPS = config["max_steps"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

### Game setting
CPU_SPEED_RATIO = 0.25
PLAYERS_SPEED_RATIO = 0.5
BALL_SPEED_RATIO = 0.75


### Helper functions
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Take a raw grayscale frame from PLE, resize to 84×84, and return
    as a uint8 array.
    """
    # frame.shape == (width, height) for grayscale
    # Resize: cv2 wants (width, height) as (cols, rows)
    frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return frame

def init_frame_stack(env) -> torch.Tensor:
    """
    Call once at the start of each episode to fill the stack
    with the first frame repeated.
    Returns a torch.Tensor of shape [1, 4, 84, 84].
    """
    raw = env.getScreenGrayscale()        # numpy uint8 array
    proc = preprocess_frame(raw)
    for _ in range(FRAME_STACK_SIZE):
        frame_stack.append(proc)
    stacked = np.stack(frame_stack, axis=0)   # (4, 84, 84)
    tensor = torch.from_numpy(stacked).float()  # convert to float32
    tensor = tensor.unsqueeze(0).to(device)     # (1, 4, 84, 84)
    return tensor

def preprocess(env) -> torch.Tensor:
    """
    Call at each step after env.act():
    1. Grab new raw frame
    2. Preprocess & append to stack
    3. Return [1, 4, 84, 84] float32 tensor on device
    """
    raw = env.getScreenGrayscale()
    proc = preprocess_frame(raw)
    frame_stack.append(proc)
    stacked = np.stack(frame_stack, axis=0)      # (4, 84, 84)
    tensor = torch.from_numpy(stacked).float()   # (4, 84, 84)
    tensor = tensor.unsqueeze(0).to(device)      # (1, 4, 84, 84)
    return tensor

def train_agent(fps=1000, display_screen=False):
    ### PLE set‑up
    game = Pong(width=64, height=48, MAX_SCORE=11,
                cpu_speed_ratio=CPU_SPEED_RATIO,
                players_speed_ratio=PLAYERS_SPEED_RATIO,
                ball_speed_ratio=BALL_SPEED_RATIO)
    env = PLE(game, fps=fps, display_screen=display_screen, add_noop_action=False)
    env.init()
    ACTIONS = env.getActionSet()  # [K_UP, K_DOWN, None]
    print("Actions are: ", ACTIONS)
    ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

    ### Initialize DQN and optimizer
    dqn = DQN(config).to(device)
    dqn.train()
    target_dqn = DQN(config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=config["lr"])
    memory = ReplayMemory(config["memory_size"])

    ### Prepare CSV logging
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)
    state_csv = open(f"logs/pong_states_{run_stamp}.csv", "w", newline="")
    state_writer = csv.writer(state_csv)
    state_writer.writerow([
        "episode", "step",
        "player_y", "player_velocity",
        "cpu_y",
        "ball_x", "ball_y",
        "ball_vel_x", "ball_vel_y",
        "reward", "action"
    ])
    ret_csv = (open(f"logs/pong_returns_{run_stamp}.csv", "w", newline=""))
    ret_writer = csv.writer(ret_csv)
    ret_writer.writerow(["episode", "return"])

    ### Training loop
    total_steps = 0
    episode_returns = []

    for ep in tqdm(range(config["n_episodes"])):
        env.reset_game()
        total_reward = 0.0
        frame_stack.clear()
        state = init_frame_stack(env) # fill stack and get first tensor
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            # print(step)
            # 1) pick action
            action_tensor = dqn.act(state)        # [1,1]
            action = ACTIONS[action_tensor.item()]

            # 2) step env
            reward = env.act(action)
            done = env.game_over()
            # if done:
            #     print(game.score_counts)
            total_reward += reward

            # 3) observe next state
            next_state = preprocess(env)

            # 4) store to replay memory
            reward_t = torch.tensor([reward], dtype=torch.float32).to(device)
            memory.push(state.squeeze(0),
                        action_tensor.squeeze(0),
                        next_state.squeeze(0),
                        reward_t)

            # 5) optimize online network
            if total_steps % config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # 6) sync target network
            if total_steps % config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            # 7) log raw game‐state
            gs = env.getGameState()
            state_writer.writerow([
                ep, step,
                gs["player_y"], gs["player_velocity"],
                gs["cpu_y"],
                gs["ball_x"], gs["ball_y"],
                gs["ball_velocity_x"], gs["ball_velocity_y"],
                reward, ACTION_IDX[action],
            ])

            # 8) advance
            state = next_state
            step += 1
            total_steps += 1

            # print(f"game scores: {game.score_counts['agent']} : {game.score_counts['cpu']}")

        # end episode
        episode_returns.append(total_reward)
        ret_writer.writerow([ep, total_reward])

    state_csv.close()
    ret_csv.close()
    print(f"Finished training. Logged states to logs/pong_states_{run_stamp}.csv")

    # Save trained network weights
    os.makedirs("agents", exist_ok=True)
    model_path = f"agents/dqn_pong_{run_stamp}.pth"
    torch.save(dqn.state_dict(), model_path)
    print(f"Trained DQN saved to {model_path}")

    return episode_returns, run_stamp, model_path

def plot_agent_return(episode_returns, run_stamp):
    # Plot return per episode
    window = 50
    returns = np.array(episode_returns, dtype=float)
    # simple centred moving average (same length as data)
    kernel = np.ones(window) / window
    ma = np.convolve(returns, kernel, mode="same")

    plt.figure(figsize=(8, 4))
    plt.plot(returns, label="Episode return", alpha=0.3)
    plt.plot(ma, label=f"{window}-episode moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Learning curve – Pong Q‑learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"logs/return_curve_{run_stamp}.png")
    print(f"Plot return per episode saved to: logs/return_curve_{run_stamp}.png")


def main(fps=1000, display=False):
    # TRAINING
    print("Start training agent...")
    episode_returns, run_stamp, model_path = train_agent(
        fps=fps,
        display_screen=display
    )
    
    # Plot learning curves
    print("\n Generate learning curve plots")
    plot_agent_return(
        episode_returns=episode_returns,
        run_stamp = run_stamp)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

