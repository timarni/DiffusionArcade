"""
qlearn_pong.py
A minimal PLE‑based training loop for Pong that logs every game‑state change.

Tested with:
    Python 3.10
    pygame==2.5.*
    PyGame‑Learning‑Environment commit 87caf6d (works with the pip fork `ple`)

Run:
    python3 qlearn_pong.py
    --fps, frames per second for training, default=1000
    parser.add_argument("--display", type=int, default=False)
    parser.add_argument("--eval", type=bool, default=True)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--eval_vis", type=bool, default=False)
"""

import csv
import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import argparse
from collections import defaultdict
from datetime import datetime

from ple import PLE
# from ple.games.pong import Pong
from pong import Pong


### Hyper-parameters for Q-learning
EPISODES = 5000 # nbr of games
MAX_STEPS = 30_000 # max_setps per episode as a safety break
ALPHA = 0.1 # learning‑rate
GAMMA = 0.99 # discount
EPSILON_START = 1.0 # ε‑greedy exploration schedule (with proba ε take a random action and with proba 1-ε action with the highest Q-value)
EPSILON_END = 0.05
EPSILON_DECAY_LEN = EPISODES # linearly decay over whole run
STATE_BINS = (12, 12, 8, 12, 3, 3, 12) # discretisation for y, y‑vel, x, y, vx, vy, y

### Evaluations for debugging
EVAL_AGENT_SCORE = True
EVAL_CPU_SCORE = False

### Game setting
CPU_SPEED_RATIO = 0.25
PLAYERS_SPEED_RATIO = 0.5
BALL_SPEED_RATIO = 0.75


### Helper functions
def discretise(state: dict) -> tuple:
    """Convert the raw dict into a small, hashable tuple.
    Because storing continous values in a Q-table needs a lot of memory"""
    py, pvy = state["player_y"], state["player_velocity"]
    cy = state["cpu_y"]
    bx, by = state["ball_x"], state["ball_y"]
    bvx, bvy = state["ball_velocity_x"], state["ball_velocity_y"]

    # Simple hash: put each value in a rough bucket
    return (
        int(py // (48 / STATE_BINS[0])),
        int(pvy // (15 / STATE_BINS[1])),
        int(cy // (48 / STATE_BINS[2])),
        int(bx // (64 / STATE_BINS[3])),
        int(by // (48 / STATE_BINS[6])),
        int((bvx > 0) - (bvx < 0)), # -1, 0, or +1
        int((bvy > 0) - (bvy < 0)),
    )


def epsilon_by_episode(ep):
    """Give back epsilon corresponding to each episode"""
    frac = max(0, (EPSILON_DECAY_LEN - ep) / EPSILON_DECAY_LEN)
    return EPSILON_END + (EPSILON_START - EPSILON_END) * frac


def train_agent(fps = 1000, display_screen = False, recording = True): # for visual inspection fps = 30 and display_screen = True
    ### PLE set‑up
    game = Pong(width=64, height=48, MAX_SCORE=11,
                cpu_speed_ratio = CPU_SPEED_RATIO, 
                players_speed_ratio = PLAYERS_SPEED_RATIO,
                ball_speed_ratio = BALL_SPEED_RATIO)
    env = PLE(game, fps=fps, display_screen=display_screen)
    env.init()
    ACTIONS = env.getActionSet() # [K_UP, K_DOWN, None]
    ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
    print(f"Game setup completed! Actions are {ACTIONS}")

    ### create Q‑table & logging CSV
    Q = defaultdict(lambda: [0.0] * len(ACTIONS))
    run_stamp = datetime.now().strftime("%Y‑%m‑%d_%H‑%M‑%S")
    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/pong_states_{run_stamp}.csv"

    if recording:
        run_stamp = datetime.now().strftime("%Y‑%m‑%d_%H‑%M‑%S")
        os.makedirs("logs", exist_ok=True)
        csv_path = f"logs/pong_states_{run_stamp}.csv"

        f_csv = open(csv_path, "w", newline="")
        csv_writer = csv.writer(f_csv)
        header = [
            "episode", "step",
            "player_y", "player_vel",
            "cpu_y",
            "ball_x", "ball_y",
            "ball_vel_x", "ball_vel_y",
            "reward", "action"
        ]
        csv_writer.writerow(header)
        ret_path = f"logs/pong_returns_{run_stamp}.csv"
        ret_csv = csv.writer(open(ret_path, "w", newline=""))
        ret_csv.writerow(["episode", "return"])

    # main loop
    total_steps = 0
    episode_returns = [] # used to plot learning curve
    agent_score_per_ep = [] # used to plot scores
    cpu_score_per_ep = [] # used to plot scores

    for ep in tqdm(range(EPISODES)):
        env.reset_game()
        total_reward = 0.0
        state = discretise(env.getGameState())
        done = False
        step = 0
        agent_score_prev = 0
        eps = epsilon_by_episode(ep)

        while not done and step < MAX_STEPS:
            # ε‑greedy policy
            if random.random() < eps: # exploration
                action = random.choice(ACTIONS)
            else: # exploitation
                qs = Q[state]
                action = ACTIONS[max(range(len(qs)), key=qs.__getitem__)]

            reward = env.act(action) # advance one frame based on action chosen and get reward for the action
            total_reward += reward # log reward
            next_state_raw = env.getGameState() # get dict containing information about the world after the step is done
            next_state = discretise(next_state_raw) # discretise to be able to store in Q-table
            done = env.game_over() # check if episode is over (i.e. someone won)

            if EVAL_AGENT_SCORE:
                agent_score_new = game.score_counts["agent"]
                if agent_score_new != agent_score_prev:
                    print(f"Agent score change at eps {ep} and step {step} to: ", agent_score_new)
                    agent_score_prev = agent_score_new

            # Q‑learning update
            best_next = max(Q[next_state])
            Q[state][ACTION_IDX[action]] += ALPHA * (
                reward + GAMMA * best_next - Q[state][ACTION_IDX[action]]
            )
            state = next_state

            # write frame/game state information to csv
            if recording:
                csv_writer.writerow([
                    ep, step,
                    next_state_raw["player_y"],
                    next_state_raw["player_velocity"],
                    next_state_raw["cpu_y"],
                    next_state_raw["ball_x"],
                    next_state_raw["ball_y"],
                    next_state_raw["ball_velocity_x"],
                    next_state_raw["ball_velocity_y"],
                    reward,
                    ACTION_IDX[action],
                ])

            step += 1
            total_steps += 1
        
        episode_returns.append(total_reward)
        agent_score_per_ep.append(game.score_counts["agent"])
        cpu_score_per_ep.append(game.score_counts["cpu"])
        if recording:
            ret_csv.writerow([ep, total_reward])

        if (ep + 1) % 50 == 0:
            print(f"[{ep + 1}/{EPISODES}] ε = {eps:.3f} | steps so far: {total_steps}")
    
    if recording:
        f_csv.close()

    print(f"\nFinished. Every state was appended to {csv_path}")

    agent_path = f"agents/q_table_{run_stamp}.pkl"
    with open(agent_path, "wb") as f:
        pickle.dump(dict(Q), f)

    print(f"The agent policy was stored in {agent_path}")

    return episode_returns, agent_score_per_ep, cpu_score_per_ep, run_stamp, Q


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

def plot_scores_per_episode(scores_agent, scores_cpu, run_stamp):
    assert len(scores_agent) == len(scores_cpu), f"Not same amount of scores for agent ({len(scores_agent)}) as for cpu ({len(scores_cpu)})"

    episodes = list(range(len(scores_agent)))

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores_agent, label='Agent Points', color='blue')
    plt.plot(episodes, scores_cpu, label='CPU Points', color='red')

    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('Points')
    plt.title('Agent vs CPU Points Over Time')
    plt.legend()
    plt.savefig(f"logs/score_curve_{run_stamp}.png")
    plt.grid(True)
    print(f"Plot return per episode saved to: logs/score_curve_{run_stamp}.png")


def evaluate_agent(Q, games = 100, fps = 1000, display = False):
    # 1. Create game & env
    game = Pong(width=64, height=48, MAX_SCORE=11, 
                cpu_speed_ratio = CPU_SPEED_RATIO,
                players_speed_ratio = PLAYERS_SPEED_RATIO,
                ball_speed_ratio = BALL_SPEED_RATIO)
    env = PLE(game, fps=fps, display_screen=display)
    env.init()
    ACTIONS = env.getActionSet() # [K_UP, K_DOWN, None]

    # 2. Pure‑exploitation evaluation loop
    wins = 0
    for ep in tqdm(range(games)):
        env.reset_game()
        state = discretise(env.getGameState())

        while not env.game_over():
            # fallback to [0,0] if the state was never seen during training
            q_values = Q.get(state, [0.0] * len(ACTIONS))
            best_a   = ACTIONS[max(range(len(q_values)), key=q_values.__getitem__)]
            env.act(best_a)
            state = discretise(env.getGameState())

        if game.score_counts["agent"] > game.score_counts["cpu"]:
            print(f"game scores:  {game.score_counts["agent"]} : {game.score_counts["cpu"]}")
            wins += 1

    # 2. Report
    print(f"\nEvaluated {games} episodes")
    print(f"Agent wins : {wins}")
    print(f"Win‑rate   : {wins/games:.2%}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=bool, default=True)
    parser.add_argument("--fps", type=int, default=1000)
    parser.add_argument("--display", type=int, default=False)
    parser.add_argument("--eval", type=bool, default=True)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--eval_vis", type=bool, default=False)
    args = parser.parse_args()

    # TRAINING
    episode_returns, agent_score_per_ep, cpu_score_per_ep, run_stamp, Q = train_agent(
        fps=args.fps,
        display_screen=args.display,
        recording=args.recording
    )

    # EVALUATION
    if args.eval:
        print("\nRunning evaluation...")
        evaluate_agent(
            Q,
        )
    
    # Plot learning curves
    if args.plot:
        print("\n Generate learning curve plots")
        plot_agent_return(
            episode_returns=episode_returns,
            run_stamp = run_stamp)
        plot_scores_per_episode(
            scores_agent = agent_score_per_ep,
            scores_cpu = cpu_score_per_ep, 
            run_stamp = run_stamp
        )
    
    if args.eval_vis:
        print("\nRunning evaluation...")
        evaluate_agent(
            Q,
            fps = 30,
            display = True
        )


if __name__ == "__main__":
    main()

