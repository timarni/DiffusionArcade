from ple import PLE
from ple.games.pong import Pong
from policy_network import PolicyNetwork
from ppo_model import PPO

# Hyperparameters
EPISODES = 50 # nbr of games
MAX_STEPS = 30_000 # max_setps per episode as a safety break
ALPHA = 1e-04 # learning‑rate
GAMMA = 0.99 # discount
EPSILON_START = 1.0 # ε‑greedy exploration schedule (with proba ε take a random action and with proba 1-ε action with the highest Q-value)
EPSILON_END = 0.05
EPSILON_DECAY_LEN = EPISODES # linearly decay over whole run
STATE_BINS = (12, 12, 8, 12, 3, 3, 12) # discretisation for y, y‑vel, x, y, vx, vy, y


def main():
    game = Pong(width=64, height=48, MAX_SCORE=11, ball_speed_ratio=0.2, cpu_speed_ratio=0.05, players_speed_ratio=0.01)
    env = PLE(game, fps=1000, display_screen=False) # display mode -> set fps=30 and display_screen = True
    env.init()

    ACTIONS = env.getActionSet() # [K_UP, K_DOWN, None]
    ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

    STATE = env.getGameState()

    # Initialise the PPO network
    policy = PolicyNetwork(len(STATE), len(ACTIONS))

    ppo_model = PPO(env, policy, ACTIONS)

    ppo_model.forward_pass()

    return 

if __name__ == '__main__':
    main()