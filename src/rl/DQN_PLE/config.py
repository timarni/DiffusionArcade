"""
In this file, you may edit the hyperparameters used for different environments.

memory_size: Maximum size of the replay memory.
n_episodes: Number of episodes to train for.
batch_size: Batch size used for training DQN.
target_update_frequency: How often to update the target network.
train_frequency: How often to train the DQN.
gamma: Discount factor.
lr: Learning rate used for optimizer.
eps_start: Starting value for epsilon (linear annealing).
eps_end: Final value for epsilon (linear annealing).
anneal_length: How many steps to anneal epsilon for.
n_actions: The number of actions can easily be accessed with env.action_space.n, but we do
    some manual engineering to account for the fact that Pong has duplicate actions.
"""

Pong = {
    'env_name': "pong",
    'memory_size': 500_000,
    'n_episodes': 500,
    'batch_size': 32,
    'target_update_frequency': 10_000,
    'train_frequency': 4,
    'gamma': 0.99,
    'lr': 1e-4,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'anneal_length': 2e6,
    'n_actions': 3,
    "obss_stack_size": 4,
    'max_steps': 120_000,
    'input_height': 84, # 48
    'input_width': 84, # 64
    'save_screen': True,
    'save_screen_freq': 4,
    'save_in_episodes': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'warm_up': 50_000
}
