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

from ple import PLE
from ple.games.pong import Pong

# --- must match training hyper‑params ---------------------------------------
STATE_BINS = (12, 12, 8, 12, 3, 3, 12)      # discretisation used for the Q‑table

def discretise(state: dict) -> tuple:
    """Bucket continuous game values into integers (same as training)."""
    py, pvy = state["player_y"], state["player_velocity"]
    cy = state["cpu_y"]
    bx, by = state["ball_x"], state["ball_y"]
    bvx, bvy = state["ball_velocity_x"], state["ball_velocity_y"]
    return (
        int(py  // (48 / STATE_BINS[0])),
        int(pvy // (15 / STATE_BINS[1])),
        int(cy  // (48 / STATE_BINS[2])),
        int(bx  // (64 / STATE_BINS[3])),
        int(by  // (48 / STATE_BINS[6])),
        int((bvx > 0) - (bvx < 0)),   # −1, 0, +1
        int((bvy > 0) - (bvy < 0)),
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
    game = Pong(width=64, height=48, MAX_SCORE=11, cpu_speed_ratio = 0.25, players_speed_ratio = 0.5, ball_speed_ratio = 1) # needs to be the same
    env = PLE(game, fps=args.fps, display_screen=args.display)
    env.init()
    ACTIONS = env.getActionSet() # [K_UP, K_DOWN]

    # 3. Pure‑exploitation evaluation loop -----------------------------------
    wins = 0
    for ep in range(args.games):
        env.reset_game()
        state = discretise(env.getGameState())

        while not env.game_over():
            # fallback to [0,0] if the state was never seen during training
            q_values = Q.get(state, [0.0] * len(ACTIONS))
            best_a   = ACTIONS[max(range(len(q_values)), key=q_values.__getitem__)]
            env.act(best_a)
            state = discretise(env.getGameState())

        print(f"game scores:  {game.score_counts["agent"]} : {game.score_counts["cpu"]}")


        if game.score_counts["agent"] > game.score_counts["cpu"]:
            print(f"game scores:  {game.score_counts["agent"]} : {game.score_counts["cpu"]}")
            wins += 1

    # 4. Report ---------------------------------------------------------------
    print(f"\nEvaluated {args.games} episodes")
    print(f"Agent wins : {wins}")
    print(f"Win‑rate   : {wins/args.games:.2%}")

if __name__ == "__main__":
    main()
