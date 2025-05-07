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
from collections import defaultdict
from datetime import datetime

from ple import PLE
from ple.games.pong import Pong


### Hyper-parameters for Q-learning
EPISODES = 50 # nbr of games
MAX_STEPS = 30_000 # max_setps per episode as a safety break
ALPHA = 1e-04 # learning‑rate
GAMMA = 0.99 # discount
EPSILON_START = 1.0 # ε‑greedy exploration schedule (with proba ε take a random action and with proba 1-ε action with the highest Q-value)
EPSILON_END = 0.05
EPSILON_DECAY_LEN = EPISODES # linearly decay over whole run
STATE_BINS = (12, 12, 8, 12, 3, 3, 12) # discretisation for y, y‑vel, x, y, vx, vy, y

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


def plot_scores_per_epoch(scores_agent, scores_cpu):
    import matplotlib.pyplot as plt

    assert len(scores_agent) == len(scores_cpu), f"Not same amount of scores for agent ({len(scores_agent)}) as for cpu ({len(scores_cpu)})"

    epochs = list(range(len(scores_agent)))

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, scores_agent, label='Agent Points', color='blue')
    plt.plot(epochs, scores_cpu, label='CPU Points', color='red')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Points')
    plt.title('Agent vs CPU Points Over Time')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


### PLE set‑up
game = Pong(width=64, height=48, MAX_SCORE=11, ball_speed_ratio=0.02, cpu_speed_ratio=0.005, players_speed_ratio=0.01)
env = PLE(game, fps=30, display_screen=True) # display mode -> set fps=30 and display_screen = True
env.init()
ACTIONS = env.getActionSet() # [K_UP, K_DOWN, None]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

### create Q‑table & logging CSV
Q = defaultdict(lambda: [0.0] * len(ACTIONS))
run_stamp = datetime.now().strftime("%Y‑%m‑%d_%H‑%M‑%S")
os.makedirs("logs", exist_ok=True)
csv_path = f"logs/pong_states_{run_stamp}.csv"

with open(csv_path, "w", newline="") as f_csv:
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

    agent_score_per_ep = []
    cpu_score_per_ep = []

    # main loop
    total_steps = 0
    for ep in range(EPISODES):
        env.reset_game()
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
            next_state_raw = env.getGameState() # get dict containing information about the world after the step is done
            next_state = discretise(next_state_raw) # discretise to be able to store in Q-table
            done = env.game_over() # check if episode is over (i.e. someone won)

            agent_score_new = game.score_counts["agent"]
            cpu_score_new = game.score_counts["cpu"]

            if agent_score_new != agent_score_prev:
                print("Agent score: ", agent_score_new)
                agent_score_prev = agent_score_new
            if cpu_score_new != cpu_score_prev:
                # print("CPU scored")
                cpu_score_prev = cpu_score_new

            # Q‑learning update
            best_next = max(Q[next_state])
            Q[state][ACTION_IDX[action]] += ALPHA * (
                reward + GAMMA * best_next - Q[state][ACTION_IDX[action]]
            )
            state = next_state

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

        agent_score_per_ep.append(agent_score_prev)
        cpu_score_per_ep.append(cpu_score_prev)

        if (ep + 1) % 50 == 0:
            print(f"[{ep + 1}/{EPISODES}] ε = {eps:.3f} | steps so far: {total_steps}")


plot_scores_per_epoch(agent_score_per_ep, cpu_score_per_ep)



print(f"\nFinished. Every state was appended to {csv_path}")
