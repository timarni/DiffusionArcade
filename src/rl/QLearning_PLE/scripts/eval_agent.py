#!/usr/bin/env python3
"""
eval_agent.py
Evaluate a trained Q‑learning policy for PLE‑Pong.

Usage examples
--------------
# 100 test games, headless (fastest)
python3 eval_agent.py --agent agents/q_table_of_500_eps.pkl

# 20 games and show the window at 30 fps
python3 eval_agent.py --agent agents/q_table_of_500_eps.pkl --games 20 --display
"""
import argparse
import pickle

from ple.ple import PLE
from ple.games.pong import Pong
# from ple.games.pong import Pong

# --- must match training hyper‑params ---------------------------------------
STATE_BINS = (6, 6, 4, 6, 3, 3, 6)  #(12, 12, 8, 12, 3, 3, 12)      # discretisation used for the Q‑table

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
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved Q‑table.")
    parser.add_argument("--agent",  required=True, help="Pickle file containing the Q‑table.")
    parser.add_argument("--games",  type=int, default=100, help="Number of evaluation episodes.")
    parser.add_argument("--fps",    type=int, default=30,  help="FPS if --display is set.")
    parser.add_argument("--display", action="store_true",  help="Render the game window.")
    args = parser.parse_args()

    # 1. Load policy ----------------------------------------------------------
    with open(args.agent, "rb") as f:
        Q = pickle.load(f)                # plain dict from training script

    # 2. Create game & env ----------------------------------------------------
    game = Pong(width=64, height=48, MAX_SCORE=11, 
                cpu_speed_ratio=0.01, players_speed_ratio=0.015, ball_speed_ratio=0.02)
    # game = Pong(width=64, height=48, MAX_SCORE=11, cpu_speed_ratio=0.25, players_speed_ratio=0.5, ball_speed_ratio=1) # needs to be the same
    env = PLE(game, fps=args.fps, display_screen=args.display)
    env.init()
    ACTIONS = env.getActionSet() # [K_UP, K_DOWN]

    # 3. Pure‑exploitation evaluation loop -----------------------------------
    wins = 0
    agent_points = []
    for ep in range(args.games):
        env.reset_game()
        state = discretise(env.getGameState())

        while not env.game_over():
            # fallback to [0,0] if the state was never seen during training
            q_values = Q.get(state, [0.0] * len(ACTIONS))
            best_a   = ACTIONS[max(range(len(q_values)), key=q_values.__getitem__)]
            env.act(best_a)
            state = discretise(env.getGameState())

        print(f"game scores:  {game.score_counts['agent']} : {game.score_counts['cpu']}")
        agent_points.append(game.score_counts["agent"])
        if game.score_counts["agent"] > game.score_counts["cpu"]:
            # print(f"game scores:  {game.score_counts['agent']} : {game.score_counts['cpu']}")
            wins += 1

    # 4. Report ---------------------------------------------------------------
    print(f"\nEvaluated {args.games} episodes")
    print(f"Agent wins : {wins}")
    print(f"Win‑rate   : {wins/args.games:.2%}")
    n = len(agent_points)
    avg_agent_points = sum(agent_points) / float(n)
    print(f"Avg points by agent: {avg_agent_points}")


if __name__ == "__main__":
    main()
