"""
qlearn_pong.py
A minimal PLE‑based training loop for Pong that logs every game‑state change.

Tested with:
    Python 3.10
    pygame==2.5.*
    PyGame‑Learning‑Environment commit 87caf6d (works with the pip fork `ple`)

Run:
    python3 qlearn_pong.py
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

import pygame
from ple import PLE
# from ple.games.pong import Pong
from pong import Pong

### Hyper-parameters for Q-learning
EPISODES = 2000 # nbr of games
MAX_STEPS = 30_000 # max_setps per episode as a safety break
ALPHA = 1e-01 # learning‑rate
GAMMA = 0.95 # discount
EPSILON_START = 1.0 # ε‑greedy exploration schedule (with proba ε take a random action and with proba 1-ε action with the highest Q-value)
EPSILON_END = 0.1
EPSILON_DECAY_LEN = int(0.9*EPISODES)  # linearly decay over whole run
STATE_BINS = (6, 6, 4, 6, 3, 3, 6)  #(12, 12, 8, 12, 3, 3, 12) # discretisation for y, y‑vel, x, y, vx, vy, y
REWARD_POLICY = 'enhanced_2'

### Evaluations for debugging
EVAL_AGENT_SCORE = True
EVAL_CPU_SCORE = True

### Game setting
CPU_SPEED_RATIO = 0.01  # 0.6
PLAYERS_SPEED_RATIO = 0.015  # 0.4
BALL_SPEED_RATIO = 0.02

### Helper functions
def discretise(state: dict) -> tuple:
    """Convert the raw dict into a small, hashable tuple.
    Because storing continous values in a Q-table needs a lot of memory"""

    def bucketize(val, max_val, bins):
        bin_size = max_val / bins
        return min(bins - 1, max(0, int(val // bin_size)))
    
    py = bucketize(state["player_y"], 48, STATE_BINS[0])
    pvy = bucketize(state["player_velocity"], 15, STATE_BINS[1])
    cy = bucketize(state["cpu_y"], 48, STATE_BINS[2])
    bx = bucketize(state["ball_x"], 64, STATE_BINS[3])
    by = bucketize(state["ball_y"], 48, STATE_BINS[4])
    bvx = int((state["ball_velocity_x"] > 0) - (state["ball_velocity_x"] < 0))
    bvy = int((state["ball_velocity_y"] > 0) - (state["ball_velocity_y"] < 0))

    return (
        py,
        pvy,
        cy,
        bx,
        by,
        bvx,
        bvy
    )

    # Simple hash: put each value in a rough bucket
    # return (
    #     int(py // (48 / STATE_BINS[0])),
    #     int(pvy // (15 / STATE_BINS[1])),
    #     int(cy // (48 / STATE_BINS[2])),
    #     int(bx // (64 / STATE_BINS[3])),
    #     int(by // (48 / STATE_BINS[6])),
    #     int((bvx > 0) - (bvx < 0)), # -1, 0, or +1
    #     int((bvy > 0) - (bvy < 0)),
    # )


def epsilon_by_episode(ep):
    """Give back epsilon corresponding to each episode"""
    frac = max(0, (EPSILON_DECAY_LEN - ep) / EPSILON_DECAY_LEN)
    return EPSILON_END + (EPSILON_START - EPSILON_END) * frac


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


def linear_decay_alpha(ep, total_episodes=1000, start=0.1, end=1e-06):
    alpha = start - (start - end) * (ep / total_episodes)
    return max(end, alpha)


def exp_decay_alpha(ep, total_episodes=500, start=0.1, end=1e-6):
    decay_rate = np.log(end / start) / total_episodes
    return start * np.exp(decay_rate * ep)


def train_agent(fps=30, display_screen=False, recording=True):
    ### PLE set‑up
    game = Pong(width=64, height=48, MAX_SCORE=11, 
                cpu_speed_ratio=CPU_SPEED_RATIO, 
                players_speed_ratio=PLAYERS_SPEED_RATIO,
                ball_speed_ratio=BALL_SPEED_RATIO, 
                reward_policy=REWARD_POLICY)
    env = PLE(game, fps=fps, display_screen=display_screen) # display mode -> set fps=30 and display_screen = True
    env.init()
    ACTIONS = env.getActionSet() # [K_UP, K_DOWN, None]
    ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

    ### create Q‑table & logging CSV
    Q = defaultdict(lambda: [0.0] * len(ACTIONS))

    run_stamp = datetime.now().strftime("%Y‑%m‑%d_%H‑%M‑%S")

    if recording:
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
        eps = epsilon_by_episode(ep)

        agent_score_prev = 0
        cpu_score_prev = 0

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
                    # print(f"Agent score change at eps {ep} and step {step} to: ", agent_score_new)
                    agent_score_prev = agent_score_new

            if EVAL_CPU_SCORE:
                cpu_score_new = game.score_counts["cpu"]
                if cpu_score_new != cpu_score_prev:
                    # print("CPU scored")
                    cpu_score_prev = cpu_score_new

            # Q‑learning update
            best_next = max(Q[next_state])
            ALPHA = exp_decay_alpha(ep, EPISODES, start=0.1, end=1e-03)  #max(1e-06, 0.1 * (0.995 ** ep))
            Q[state][ACTION_IDX[action]] += ALPHA * (
                reward + GAMMA * best_next - Q[state][ACTION_IDX[action]]
            )
            state = next_state

            if recording:
                # write frame/game state information to csv
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
        agent_score_per_ep.append(agent_score_prev)
        cpu_score_per_ep.append(cpu_score_prev)
        if recording:
            ret_csv.writerow([ep, total_reward])

        if (ep + 1) % 50 == 0:
            print(f"[{ep + 1}/{EPISODES}] ε = {eps:.3f}, lr = {ALPHA:.6f} | steps so far: {total_steps}")

    if recording:
        f_csv.close()

        print(f"\nFinished. Every state was appended to {csv_path}")

    os.makedirs("agents", exist_ok=True)
    agent_path = f"agents/q_table_{run_stamp}.pkl"
    with open(agent_path, "wb") as f:
        pickle.dump(dict(Q), f)

    print(f"The agent policy was stored in {agent_path}")

    pygame.quit()

    return episode_returns, agent_score_per_ep, cpu_score_per_ep, run_stamp, Q


def evaluate_agent(Q, games=100, fps=1000, display=False):
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

        print(f"game scores:  {game.score_counts['agent']} : {game.score_counts['cpu']}")
        if game.score_counts["agent"] > game.score_counts["cpu"]:
            # print(f"game scores:  {game.score_counts['agent']} : {game.score_counts['cpu']}")
            wins += 1

    # 2. Report
    print(f"\nEvaluated {games} episodes")
    print(f"Agent wins : {wins}")
    print(f"Win‑rate   : {wins/games:.2%}")

    pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_recording", action="store_true", help="Disable recording (default: enabled)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--display", action="store_true", help="Enable recording (default: disabled)")
    parser.add_argument("--no_eval", action="store_true", help="Disable eval (default: enabled)")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting (default: enabled)")
    parser.add_argument("--eval_vis", action="store_true", help="Enable evaluation visualization (default: disabled)")
    args = parser.parse_args()

    # TRAINING
    episode_returns, agent_score_per_ep, cpu_score_per_ep, run_stamp, Q = train_agent(
        fps=args.fps,
        display_screen=args.display,
        recording=not args.no_recording
    )

    # Plot learning curves
    if not args.no_plot:
        print("\n Generate learning curve plots")
        plot_agent_return(
            episode_returns=episode_returns,
            run_stamp = run_stamp)
        plot_scores_per_episode(
            scores_agent = agent_score_per_ep,
            scores_cpu = cpu_score_per_ep, 
            run_stamp = run_stamp
        )

    # EVALUATION
    if not args.no_eval:
        print("\nRunning evaluation...")
        evaluate_agent(
            Q,
        )
    
    if args.eval_vis:
        print("\nRunning evaluation...")
        evaluate_agent(
            Q,
            fps=30,
            display=True
        )


if __name__ == "__main__":
    main()

