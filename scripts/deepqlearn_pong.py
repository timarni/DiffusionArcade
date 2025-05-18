"""
deepqlearn_pong.py
A minimal PLE‑based training loop for Pong that uses a deepq network
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
SAVE_SCREEN = config["save_screen"]
FRAME_STACK_SIZE = config["obss_stack_size"]
frame_stack = deque(maxlen=FRAME_STACK_SIZE)
IMG_H = config["input_height"]
IMG_W = config["input_width"]
print(f"Using image size width: {IMG_W} and height: {IMG_H}")
MAX_STEPS = config["max_steps"]
FRAME_SKIP = 4 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

### Game setting
CPU_SPEED_RATIO = 0.5
PLAYERS_SPEED_RATIO = 0.5
BALL_SPEED_RATIO = 0.75

### REWARD
POSITIVE = 1
NEGATIVE = -1
TICK = 0
LOSS = 0
WIN = 0


### Helper functions
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize and map 0‑255 bytes → 0‑1 float32."""
    if frame.shape[:2] != (IMG_W, IMG_H):# (H, W) order
        frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        print(f"DEBUGGING RESIZE: {frame.shape}")
        print(f"DEBUGGING TARGET: {(IMG_H, IMG_W)}")
    frame = frame.astype(np.float32) / 255.0 # normalise here
    # optional: centre to ‑0.5 … +0.5
    # frame -= 0.5
    return frame

def init_frame_stack(env) -> torch.Tensor:
    """
    Call once at the start of each episode to fill the stack
    with the first frame repeated.
    Returns a torch.Tensor of shape [1, 4, IMG_H, IMG_W].
    """
    raw = env.getScreenGrayscale() # numpy uint8 array
    proc = preprocess_frame(raw)
    for _ in range(FRAME_STACK_SIZE):
        frame_stack.append(proc)
    stacked = np.stack(frame_stack, axis=0) # (4, IMG_H, IMG_W)
    tensor = torch.from_numpy(stacked).float() # convert to float32
    tensor = tensor.unsqueeze(0).to(device) # (1, 4, IMG_H, IMG_W)
    return tensor

def preprocess(env) -> torch.Tensor:
    """
    Call at each step after env.act():
    1. Grab new raw frame
    2. Preprocess & append to stack
    3. Return [1, 4, IMG_H, IMG_W] float32 tensor on device
    """
    raw = env.getScreenGrayscale()
    proc = preprocess_frame(raw)
    frame_stack.append(proc)
    stacked = np.stack(frame_stack, axis=0) # (4, IMG_H, IMG_W)
    tensor = torch.from_numpy(stacked).float() # (4, IMG_H, IMG_W)
    tensor = tensor.unsqueeze(0).to(device) # (1, 4, IMG_H, IMG_W)
    return tensor

def train_agent(fps=30, display_screen=False):
    ### Log hyperparameters in terminal
    print(f'''Start training with hyperparameters: LR: {config['lr']}, 
            CPU: {CPU_SPEED_RATIO}, Player: {PLAYERS_SPEED_RATIO},
            Ball: {BALL_SPEED_RATIO}, anneal length: {config['anneal_length']},
            REWARD: {POSITIVE}, {NEGATIVE}, {TICK}, {WIN}, {LOSS}''')
    ### PLE set‑up
    game = Pong(width=IMG_W, height=IMG_H, MAX_SCORE=21,
                cpu_speed_ratio=CPU_SPEED_RATIO,
                players_speed_ratio=PLAYERS_SPEED_RATIO,
                ball_speed_ratio=BALL_SPEED_RATIO)
    env = PLE(game, fps=fps, display_screen=display_screen, reward_values = {
                                                        "positive": POSITIVE,
                                                        "negative": NEGATIVE,
                                                        "tick": TICK,
                                                        "loss": LOSS,
                                                        "win": WIN
                                                        }
            )
    env.init()
    ACTIONS = env.getActionSet()  # [K_UP, K_DOWN, None]
    ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
    assert len(ACTIONS) == config["n_actions"], (
        f"Config says {config['n_actions']} actions, "
        f"but environment returned {len(ACTIONS)}."
    )
    print("Actions are: ", ACTIONS)


    ### Initialize DQN, Target DQN and optimizer
    dqn = DQN(config).to(device)
    dqn.train()
    target_dqn = DQN(config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=config["lr"])
    memory = ReplayMemory(config["memory_size"])
    print(f"eps decay per step: {(dqn.eps_start - dqn.eps_end) / dqn.anneal_length}")

    ### Prepare screen and CSV logging
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"agents/{run_stamp}", exist_ok=True)
    if SAVE_SCREEN:
        screen_folder = f"screens/{run_stamp}"
        os.makedirs(screen_folder, exist_ok=True)
        screen_csv = open(f"screens/{run_stamp}.csv", "w", newline="")
        screen_writer = csv.writer(screen_csv)
        screen_writer.writerow(["episode_step", "action"])
    ret_csv = (open(f"logs/pong_returns_{run_stamp}.csv", "w", newline=""))
    ret_writer = csv.writer(ret_csv)
    ret_writer.writerow(["episode", "return"])

    ### Training loop
    total_steps = 0
    episode_returns = []
    if SAVE_SCREEN:
        nbr_of_screens = 0

    for ep in tqdm(range(config["n_episodes"])):
        # Start new episode
        env.reset_game()
        total_reward = 0.0
        frame_stack.clear()
        state = init_frame_stack(env) # fill stack and get first tensor
        done = False
        step = 0
        skip_count = 0 # counts how many of the skipped frames we have executed
        cum_reward = 0.0 # reward accumulated over those skipped frames


        while not done and step < MAX_STEPS:
            # 1) pick action
            if skip_count == 0:
                action_tensor = dqn.act(state) # shape [1, 1]
                action = ACTIONS[action_tensor.item()]

            
            # Save screen
            if SAVE_SCREEN and ep in config['save_in_episodes'] and step % config['save_screen_freq'] == 0:
                # grab full‑colour frame
                frame_rgb = env.getScreenRGB() # (H, W, 3) uint8, RGB order
                # convert to BGR because OpenCV expects that
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                # build filename: e.g. ep0007_step01234.jpg
                fname  = f"{screen_folder}/ep{ep:04d}_step{step:05d}.jpg"
                cv2.imwrite(fname, frame_bgr) # finally write JPEG
                nbr_of_screens += 1
                screen_writer.writerow([f"{ep}_{step}", ACTION_IDX[action]])

            # 2) step env
            reward = env.act(action)
            done = env.game_over()
            total_reward += reward
            cum_reward += reward
            skip_count += 1

            if skip_count == FRAME_SKIP or done:
                # 3) observe next state
                next_state = preprocess(env)

                # 4) store to replay memory
                reward_t = torch.tensor([cum_reward], dtype=torch.float32).to(device)
                memory.push(state.squeeze(0),
                            action_tensor.squeeze(0),
                            next_state.squeeze(0),
                            reward_t,
                            done)

                # 5) optimize online network
                if total_steps > config['warm_up'] and total_steps % config['train_frequency'] == 0:
                    loss = optimize(dqn, target_dqn, memory, optimizer)

                    if total_steps % 2000 == 0:  # print every 2k steps
                        with torch.no_grad():
                            q_now = dqn(state).abs().mean().item()
                        print(f"[{total_steps:>6}] loss={loss:8.6f}   mean|Q|={q_now:5.3f}   ε={dqn.current_eps:5.3f}")

                # 6) sync target network
                if total_steps > config['warm_up'] and total_steps % config["target_update_frequency"] == 0:
                    print(f"[DQN TARGET UPDATED]")
                    target_dqn.load_state_dict(dqn.state_dict())

                # 7) advance
                state = next_state
                skip_count = 0
                cum_reward = 0

                total_steps += 1
                
                eps_decay = (dqn.eps_start - dqn.eps_end) / dqn.anneal_length
                dqn.current_eps = max(dqn.eps_end, dqn.current_eps - eps_decay)
                
            step += 1 # changed intendation


        print(f"game scores after game {ep} with {step//4} / {total_steps} steps -> {game.score_counts['agent']} : {game.score_counts['cpu']}")
        print(f"Game {ep} with -> epsilon: {dqn.current_eps} and reward: {total_reward}")

        # Save trained network weights
        if ep != 0 and (ep % 50 == 0 or ep == (config['n_episodes']-1)):
            model_path = f"agents/{run_stamp}/checkpoint{ep}.pth"
            torch.save(dqn.state_dict(), model_path)
            print(f"Trained DQN checkpoint saved to {model_path}")


        # end episode
        episode_returns.append(total_reward)
        ret_writer.writerow([ep, total_reward])

    # state_csv.close()
    ret_csv.close()
    if SAVE_SCREEN:
        screen_csv.close()
    # print(f"Finished training. Logged states to logs/pong_states_{run_stamp}.csv")
    print(f"training had {config['n_episodes']} episodes and {total_steps} steps")
    if SAVE_SCREEN:
        print(f"Saved {nbr_of_screens} pictures")

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

