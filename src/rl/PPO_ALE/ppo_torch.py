import os
import argparse

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
# from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gym.wrappers import RecordVideo
from reward_wrapper import RewardWrapper

import ale_py
from atariari.benchmark.wrapper import AtariARIWrapper

from policy_network import PolicyNetwork, ActorCritic
from policy_network_conv import PolicyNetworkConv, ActorCriticConv

import time

run_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

def create_agent(n_observation, n_actions, hidden_dim, dropout):

    actor = PolicyNetwork(n_observation, hidden_dim, n_actions, dropout)
    critic = PolicyNetwork(n_observation, hidden_dim, 1, dropout)

    agent = ActorCritic(actor=actor, critic=critic)

    return agent

def create_conv_agent(n_rows, n_cols, channels, hidden_dim, hidden_chan, n_actions, dropout):

    actor = PolicyNetworkConv(channels, 5, 16, n_actions, n_rows, n_cols, hidden_dim, dropout)
    critic = PolicyNetworkConv(channels, 5, 16, 1, n_rows, n_cols, hidden_dim, dropout)

    agent = ActorCriticConv(actor=actor, critic=critic)

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
    returns = (returns - returns.mean()) / (returns.std() + 1e-8) # Normalise
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

    loss_1 = policy_ratio * advantages # Loss 1

    clamped_policy_ratio = torch.clamp(policy_ratio, 1-eps, 1+eps)

    loss_2 = clamped_policy_ratio * advantages # Loss 2 - the clamped one

    loss = torch.min(loss_1, loss_2) # We take the smaller of the two

    return loss

def calculate_losses(loss, entropy, entropy_bonus, returns, value_pred):
    policy_loss = -(loss + entropy * entropy_bonus).sum()
    value_loss = F.mse_loss(returns, value_pred).sum() * 0.5

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

    state, info = env.reset()
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())

    agent.train()

    # Run until the game hits a terminal state
    while not done:
        # Add the initial state to the list of states that we have seen
        state = torch.FloatTensor(list(info['labels'].values()))
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
        state, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated

        # Store everything :)
        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward

    # Turn all the lists into tensors of tensors 
    states = torch.stack(states)
    
    actions = torch.stack(actions)
    actions_log_probability = torch.stack(actions_log_probability)
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
        action_probs, value_probs = agent(states)
        value_probs = value_probs
        action_probs = F.softmax(action_probs)
        dist = distributions.Categorical(action_probs)

        entropy = dist.entropy()
        new_action_log_probs = dist.log_prob(actions)
        surrogate_loss = clipped_loss(actions_log_probs, new_action_log_probs, epsilon, advantages)
        policy_loss, value_loss = calculate_losses(surrogate_loss, entropy, entropy_bonus, returns, value_probs)
        optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        optimizer.step()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / total_steps, total_value_loss / total_steps

def evaluate(env, agent):
    agent.eval()
    done = False
    episode_reward = 0
    state, info = env.reset()
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())

    while not done:
        state = torch.FloatTensor(list(info['labels'].values()))
        with torch.no_grad():
            action_pred, _ = agent(state)
            action_prob = F.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        episode_reward += reward
    return episode_reward

def plot_train_rewards(args, train_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(train_rewards, label='Training Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Training Reward', fontsize=20)
    # plt.hlines(reward_threshold, 0, len(train_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()

    # Ensure directory exists
    output_dir = f"images/{args.env}"
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    plt.savefig(f"{output_dir}/train_rewards_CartPole_{run_stamp}.png")

def plot_test_rewards(args, test_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Testing Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Testing Reward', fontsize=20)
    # plt.hlines(reward_threshold, 0, len(test_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()

    # Ensure directory exists
    output_dir = f"images/{args.env}"
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    plt.savefig(f"{output_dir}/test_rewards_CartPole_{run_stamp}.png")

def run_ppo(env, args):
    DISCOUNT_FACTOR = 0.99
    MAX_EPISODES = args.max_episodes
    REWARD_THRESHOLD = 1000
    PRINT_INTERVAL = 50
    PPO_STEPS = 8 # MAYBE CHANGE THIS
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFF = 0.01
    HIDDEN_DIM = 128
    DROPOUT = 0.1
    LR = 1e-4

    n_actions = env.action_space.n
    env.reset()
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())

    n_observations = len(info['labels'].values())

    print('-' * 50)
    print(f"RUNNING TRAINING FOR ENVIRONMENT -> {args.env}")
    print("MAX EPISODES:\t", MAX_EPISODES)
    print("PRINT INTERVAL:\t", PRINT_INTERVAL)
    print('-' * 50)

    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []
    agent = create_agent(n_observations, n_actions, HIDDEN_DIM, DROPOUT)

    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / MAX_EPISODES)

    for episode in range(1, MAX_EPISODES+1):
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

        scheduler.step()

        if episode % PRINT_INTERVAL == 0:
            print(f'Episode: {episode:3} | \
                  Mean Train Rewards: {mean_train_rewards:3.1f} \
                  | Mean Test Rewards: {mean_test_rewards:3.1f} \
                  | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                  | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

    # plot_train_rewards(args, train_rewards, REWARD_THRESHOLD)
    # plot_test_rewards(args, test_rewards, REWARD_THRESHOLD)

    plot_agent_return(args, train_rewards, run_stamp, is_training_run=True)
    plot_agent_return(args, test_rewards, run_stamp, is_training_run=False)

    os.makedirs("./models", exist_ok=True)
    torch.save(agent.state_dict(), f"./models/model_{args.env}_{run_stamp}.pt")

    env.close()

def plot_agent_return(args, episode_returns, run_stamp, is_training_run):
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
    if is_training_run:
        plt.title("Training curve – Pong PPO")
    else:
        plt.title("Testing curve – Pong PPO")
    plt.legend()
    plt.tight_layout()

    output_dir = f"images/{args.env}"
    os.makedirs(output_dir, exist_ok=True)

    if is_training_run:
        plt.savefig(f"images/{args.env}/return_curve_train_{run_stamp}.png")
        print(f"Plot return per episode saved to: logs/return_curve_train_{run_stamp}.png")
    else:
        plt.savefig(f"images/{args.env}/return_curve_test_{run_stamp}.png")
        print(f"Plot return per episode saved to: logs/return_curve_test_{run_stamp}.png")
        

def main():
    parser = argparse.ArgumentParser(description="A simple script with arguments.")

    parser.add_argument('--env', type=str, required=True, help='Name of the environment to use')
    parser.add_argument('--max-episodes', type=int, required=True, help='Whether you want to see the environment or not')
    parser.add_argument('--human-view', action='store_true', help='Whether you want to see the environment or not')
    parser.add_argument('--record-output', action='store_true', help='Whether you want to see the environment or not')

    args = parser.parse_args()

    match args.env:
        case 'cartpole':
            environment_name = "CartPole-v1"
        case 'acrobot':
            environment_name = "Acrobot-v1"
        case 'pong':
            environment_name = "Pong-v4"

    if args.human_view:
        env = gym.make(environment_name, render_mode="human")
    else:
        env = gym.make(environment_name, render_mode="rgb_array")

    if args.env == 'pong':
        env = AtariARIWrapper(env)
        env = RewardWrapper(env)

    if args.record_output:
        os.makedirs("./videos/run_{run_stamp}", exist_ok=True)
        env = RecordVideo(env=env, video_folder=f"./videos/run_{run_stamp}", name_prefix="test-video", episode_trigger=lambda x: (x+1) % 1000 == 0)

    run_ppo(env, args)

if __name__=='__main__':
    main()
