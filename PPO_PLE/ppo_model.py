import policy_network
import random
from constants import *
from helpers import *

import torch
import torch.nn.functional as F


class PPO():
    def __init__(self, env, policy, actions):
        self.env = env
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.actions = actions

    def forward_pass(self):
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        # main loop
        total_steps = 0
        for ep in range(EPISODES):
            self.env.reset_game()
            done = False
            step = 0
            eps = epsilon_by_episode(ep)

            agent_score_prev = 0
            cpu_score_prev = 0

            while not done and step < MAX_STEPS:
                state = self.env.getGameState()
                state_tensor = torch.tensor(list(state.values()), dtype=torch.float32).unsqueeze(0)  # shape [1, 7]

                # Get action distribution and value
                logits, value = self.policy(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                reward = self.env.act(self.actions[action.item()])
                next_state = self.env.getGameState()
                done = self.env.game_over()

                states.append(torch.tensor(list(next_state.values())))
                actions.append(action)
                rewards.append(torch.tensor([reward], dtype=torch.float32))
                log_probs.append(log_prob)
                values.append(value.squeeze(0))
                dones.append(torch.tensor([done], dtype=torch.float32))

                # agent_score_new = game.score_counts["agent"]
                # cpu_score_new = game.score_counts["cpu"]

                # if agent_score_new != agent_score_prev:
                #     print("Agent score: ", agent_score_new)
                #     agent_score_prev = agent_score_new
                # if cpu_score_new != cpu_score_prev:
                #     # print("CPU scored")
                #     cpu_score_prev = cpu_score_new

                step += 1
                total_steps += 1

            returns = []

            R = 0
            for i in reversed(range(len(rewards))):
                R = rewards[i] + GAMMA * R * (1 - dones[i])
                returns.insert(0, R)

            returns = torch.stack(returns)
            values = torch.stack(values)
            advantages = returns - values.detach()

            # print("STATES:", states)
            # print("STATES TYPE:", type(states))

            # Convert states to tensors
            states_tensor = torch.stack(states)
            actions_tensor = torch.stack(actions)
            old_log_probs_tensor = torch.stack(log_probs)

            advantages_tensor = advantages.detach()
            returns_tensor = returns.detach()

            self.ppo_update(states_tensor, actions_tensor, old_log_probs_tensor, advantages_tensor, returns_tensor)

            if (ep + 1) % 50 == 0:
                print(f"[{ep + 1}/{EPISODES}] ε = {eps:.3f} | steps so far: {total_steps}")
        
    def ppo_update(self, states, actions, log_probs, advantages, returns):
        for _ in range(PPO_EPOCHS):
            logits, values_pred = self.policy(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred.squeeze(), returns.detach())
            total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            values_pred = values_pred.detach()
            new_log_probs = new_log_probs.detach()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        output = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }

        return output


