from ple import PLE
from ple.games.pong import Pong

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np


from collections import defaultdict


from policy_network import PolicyNetwork, ActorCritic
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

game = Pong(width=64, height=48, MAX_SCORE=11, ball_speed_ratio=0.2, cpu_speed_ratio=0.05, players_speed_ratio=0.01)
env = PLE(game, fps=60, display_screen=False) # display mode -> set fps=30 and display_screen = True
env.init()

ACTIONS = env.getActionSet()
print("ACTIONS:\t", ACTIONS)

def create_agent(env, hidden_dim, dropout):
    input_dim = len(env.getGameState())
    print(env.getGameState())
    action_dim = len(env.getActionSet())-1

    print("AGENT INPUT DIMENSION:\t", input_dim)
    print("AGENT OUTPUT DIMENSION:\t", action_dim)


    actor = PolicyNetwork(input_dim, hidden_dim, action_dim, dropout)
    critic = PolicyNetwork(input_dim, hidden_dim, 1, dropout)

    agent = ActorCritic(actor=actor, critic=critic)

    return agent

def get_returns(rewards, discount_factor):
    """
    Get the returns based on a sequence of rewards and a discount factor
    """
    returns = []
    cum_reward = 0
    for reward in reversed(rewards):
        cum_reward = reward + cum_reward * discount_factor
        returns.insert(0, cum_reward)

    returns = torch.tensor(returns) # Make it a tensor
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)# Normalise
    return returns

def calculate_advantages(returns, values):
    """
    Calculated the advantages based on our returns and our critic values
    """
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def clipped_loss(old_log_action_prob, new_log_action_prob, eps, advantages):
    advantages = advantages.detach()
    policy_ratio = torch.exp(new_log_action_prob)/torch.exp(old_log_action_prob)

    loss_1 = policy_ratio * advantages

    # clamped_policy_ratio = torch.clamp(policy_ratio, 1-eps, 1+eps)

    # loss_2 = clamped_policy_ratio * advantages

    # loss = torch.min(loss_1, loss_2)

    return loss_1

def calculate_losses(loss, entropy, entropy_bonus, returns, value_pred):
    policy_loss = -(loss - entropy * entropy_bonus).sum()
    value_loss = F.mse_loss(returns, value_pred).sum()

    return policy_loss, value_loss

def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    return states, actions, actions_log_probability, values, rewards, done, episode_reward

def forward_pass(env, agent, optimiser, discount_factor):
    """
    Runs the current policy to generate data to train on
    """

    # Get the inital states and actions to start off the model
    states, actions, actions_log_probability, values, rewards, done, episode_reward = init_training()

    env.reset_game()
    state = list(env.getGameState().values())
    agent.train()

    # Run until the game hits a terminal state
    while not done:
        # Add the initial state to the list of states that we have seen
        state = torch.FloatTensor(state).unsqueeze(0)
        states.append(state)

        # Evaluate the action prediction (actor) and the value prediction (critic) with the agent
        action_pred, value_pred = agent(state)
        action_pred = F.softmax(action_pred, dim=-1)

        # and then sample an action from the output actor probabilities
        dist = distributions.Categorical(action_pred)
        action = dist.sample()

        # Calculate the log probabilities of our action
        log_prob_action = dist.log_prob(action)

        # Get the remaining stuff that we need from our environment
        # like the reward for this step, wether we reached a terminal state,
        # and the state of the environment given the last action
        reward = env.act(ACTIONS[action])
        done = env.game_over()
        state = list(env.getGameState().values())

        # Store everything :)
        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward

    # print("ACTION DISTRIBUTION")
    # print("0 \t->\t", list(actions).count(0))
    # print("1 \t->\t", list(actions).count(1))
    # print("2 \t->\t", list(actions).count(2))

    # Turn all the lists into tensors of tensors 
    states = torch.cat(states)
    actions = torch.cat(actions)
    actions_log_probability = torch.cat(actions_log_probability)
    values = torch.cat(values).squeeze(-1)

    # Calculate the returns and then the advantages based on this pass through the network
    returns = get_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    return episode_reward, states, actions, actions_log_probability, advantages, returns

def update_policy(agent, states, actions, actions_log_probs, advantages, returns, total_steps, epsilon, entropy_bonus, optimizer):
    # BATCH_SIZE = 512
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probs = actions_log_probs.detach()
    actions = actions.detach()

    # dataset = TensorDataset(states, actions, actions_log_probs, advantages, returns,)
    for _ in range(total_steps):
        batch_states, batch_actions, batch_actions_log_probs, batch_advantages, batch_returns = states, actions, actions_log_probs, advantages, returns

        batch_action_probs, batch_value_probs = agent(batch_states)
        batch_value_probs = batch_value_probs
        batch_action_probs = F.softmax(batch_action_probs)
        batch_dist = distributions.Categorical(batch_action_probs)

        # print(f"Returns: {batch_returns[:5]}")
        # print(f"Critic Values: {batch_value_probs.squeeze()[:5]}")

        batch_entropy = batch_dist.entropy()
        new_action_log_probs = batch_dist.log_prob(batch_actions)
        surrogate_loss = clipped_loss(batch_actions_log_probs, new_action_log_probs, epsilon, batch_advantages)
        policy_loss, value_loss = calculate_losses(surrogate_loss, batch_entropy, entropy_bonus, batch_returns, batch_value_probs)
        optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        optimizer.step()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / total_steps, total_value_loss / total_steps

def evaluate(env, agent):
    agent.eval()
    rewards = []
    done = False
    episode_reward = 0
    env.reset_game()
    state = list(env.getGameState().values())
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = agent(state)
            action_prob = F.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        reward = env.act(ACTIONS[action])
        state = list(env.getGameState().values())
        done = env.game_over()
        episode_reward += reward
    return episode_reward

def plot_train_rewards(train_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(train_rewards, label='Training Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Training Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(train_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f"train_rewards_{timestr}.png")

def plot_test_rewards(test_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Testing Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Testing Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(test_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f"test_rewards_{timestr}.png")

def run_ppo():
    DISCOUNT_FACTOR = 0.99
    MAX_EPISODES = 100000
    REWARD_THRESHOLD = 15
    PRINT_INTERVAL = 10
    PPO_STEPS = 8 # MAYBE CHANGE THIS
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFF = 0.01
    HIDDEN_DIM = 64
    DROPOUT = 0.2
    LR = 1e-2

    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []
    agent = create_agent(env, HIDDEN_DIM, DROPOUT)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    for episode in tqdm(range(1, MAX_EPISODES+1)):
        train_reward, states, actions, actions_log_prob, advantages, returns = forward_pass(env, agent, optimizer, DISCOUNT_FACTOR)
        policy_loss, value_loss = update_policy(
            agent,
            states,
            actions,
            actions_log_prob,
            advantages,
            returns,
            PPO_STEPS,
            EPSILON,
            ENTROPY_COEFF,
            optimizer
        )
        test_reward = evaluate(env, agent)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
        mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))
        if episode % PRINT_INTERVAL == 0:
            print(f'Episode: {episode:3} | \
                  Mean Train Rewards: {mean_train_rewards:3.1f} \
                  | Mean Test Rewards: {mean_test_rewards:3.1f} \
                  | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                  | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

    plot_train_rewards(train_rewards, REWARD_THRESHOLD)
    plot_test_rewards(test_rewards, REWARD_THRESHOLD)


def main():
    run_ppo()

if __name__=='__main__':
    main()
