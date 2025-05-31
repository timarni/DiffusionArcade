import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    @torch.no_grad()
    def push(self, obs, action, next_obs, reward, done):
        if len(self.memory) < self.capacity:
             self.memory.append(None)

        self.memory[self.position] = (
            (obs      * 255).round().to('cpu', dtype=torch.uint8),
            action.to('cpu', dtype=torch.int8),
            (next_obs * 255).round().to('cpu', dtype=torch.uint8),
            reward.to('cpu'),
            done,
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample = random.sample(self.memory[:len(self)], batch_size)
        obs, act, nxt, rew, done = map(list, zip(*sample))

        obs = torch.stack(obs, 0).float().div_(255).to(device, non_blocking=True)
        nxt = torch.stack(nxt, 0).float().div_(255).to(device, non_blocking=True)
        act = torch.stack(act, 0).long().to(device, non_blocking=True)
        rew = torch.stack(rew, 0).squeeze(1).to(device, non_blocking=True)
        done = torch.tensor(done, dtype=torch.bool, device=device)
        return obs, act, nxt, rew, done


def print_frame_stats(t, name="tensor"):
    # t shape: [B, 4, H, W]  (after unsqueeze)
    print(f"{name}: shape {tuple(t.shape)}, "
        f"min {t.min():.3f}, max {t.max():.3f}, "
        f"mean {t.mean():.5f}, non-zeros {(t>0).float().mean()*100:.3f}%")


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.H = env_config["input_height"]
        self.W = env_config["input_width"]

        self.current_eps = self.eps_start

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1) # Changed?

        self.env = env_config["env_name"]
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        conv_out_dim = self._get_conv_out_dim()

        self.fc1 = nn.Linear(conv_out_dim, 512)
        self.fc2 = nn.Linear(512, self.n_actions)
    
    def _get_conv_out_dim(self):
        """
        Runs a dummy batch through conv layers to determine the
        flattened feature size, so nothing is hard-coded.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 4, self.H, self.W) # [B=1, C=4, H, W]
            x = self.relu(self.conv1(dummy))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            return int(np.prod(x.shape[1:])) # C × H × W


    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # Takes an observation tensor and returns a tensor of actions.
        # For example, if the state dimension is 4 and the batch size is 32,
        # the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        
        # Epsilon-greedy exploration.
        if (not exploit) and (random.random() < self.current_eps):
            actions = torch.randint(
                0, self.n_actions, (observation.shape[0], 1)).to(device)
        else:
            actions = torch.argmax(self.forward(
                observation), dim=1).to(device).unsqueeze(1)

        return actions


def optimize(dqn, target_dqn, memory, optimizer):
    """Sample a batch from replay memory and update the online network (batched, fast)."""
    # 1) Skip update if we have too little data
    if len(memory) < dqn.batch_size:
        return

    # 2) Sample and batch‑stack tensors
    obss, actions, next_obss, rewards, dones = memory.sample(dqn.batch_size)

    # --- QUICK HEALTH-CHECK: are Q-values still ~0 ? -------------------------
    if not hasattr(optimize, "tick"):      # static counter on the function
        optimize.tick = 0
    optimize.tick += 1
    if optimize.tick % 400 == 0:           # print every 400 SGD updates
        with torch.no_grad():
            qs = dqn(obss).detach().cpu()
        print(
            f"[Q-stats] mean|Q|={qs.abs().mean():.3f} "
            f"max|Q|={qs.abs().max():.3f}"
        )
    # -------------------------------------------------------------------------


    # 3) Current Q‑values Q(s,a)
    q_pred = dqn(obss) # [B, n_actions]
    q_values = q_pred.gather(1, actions).squeeze(1) # [B]

    with torch.no_grad():
        # Step 1: action selection using online network
        next_actions = dqn(next_obss).argmax(1) # [B]
        # Step 2: action evaluation using target network
        next_q = target_dqn(next_obss).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # [B]
    q_targets = rewards + dqn.gamma * next_q
    q_targets[dones] = rewards[dones]


    # 5) Epsilon annealing -> now moved to main loop

    # 6) Loss, back‑prop, optimizer step
    loss = F.smooth_l1_loss(q_values, q_targets)

    optimizer.zero_grad()
    loss.backward()

    # --- GRADIENT CHECK ----------------------
    grad_norm = torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1)
    if optimize.tick % 400 == 0:
        print(f"[grad] ‖g‖₂ before clip = {grad_norm:6.4f}")
    # -----------------------------------------

    optimizer.step()

    return loss.item()


# ## Old version, where everything was on GPU (bad!! -> needed 58GB of GPU VRAM)

# class ReplayMemory:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def __len__(self):
#         return len(self.memory)

#     def push(self, obs, action, next_obs, reward, done):
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)

#         self.memory[self.position] = (obs, action, next_obs, reward, done)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         """
#         Samples batch_size transitions from the replay memory and returns a tuple
#             (obs, action, next_obs, reward)
#         """
#         sample = random.sample(self.memory, batch_size)
#         return tuple(zip(*sample))


# ## Old version that worked but used too much GPU VRAM

# def optimize(dqn, target_dqn, memory, optimizer):
#     """Sample a batch from replay memory and update the online network (batched, fast)."""
#     # 1) Skip update if we have too little data
#     if len(memory) < dqn.batch_size:
#         return

#     # 2) Sample and batch‑stack tensors
#     obss, actions, next_obss, rewards, dones = memory.sample(dqn.batch_size)

#     obss = torch.stack(obss).to(device) # [B, 4, 84, 84]
#     actions = torch.stack(actions).long().to(device) # [B, 1]  (indices)
#     next_obss = torch.stack(next_obss).to(device) # [B, 4, 84, 84]
#     rewards = torch.stack(rewards).to(device).squeeze(1) # [B]
#     dones = torch.tensor(dones, dtype=torch.bool).to(device)  # [B]
#     # print_frame_stats(obss, "obss")

#     # --- QUICK HEALTH-CHECK: are Q-values still ~0 ? -------------------------
#     if not hasattr(optimize, "tick"):      # static counter on the function
#         optimize.tick = 0
#     optimize.tick += 1
#     if optimize.tick % 400 == 0:           # print every 400 SGD updates
#         with torch.no_grad():
#             qs = dqn(obss).detach().cpu()
#         print(
#             f"[Q-stats] mean|Q|={qs.abs().mean():.3f} "
#             f"max|Q|={qs.abs().max():.3f}"
#         )
#     # -------------------------------------------------------------------------


#     # 3) Current Q‑values Q(s,a)
#     q_pred = dqn(obss) # [B, n_actions]
#     q_values = q_pred.gather(1, actions).squeeze(1) # [B]

#     with torch.no_grad():
#         # Step 1: action selection using online network
#         next_actions = dqn(next_obss).argmax(1) # [B]
#         # Step 2: action evaluation using target network
#         next_q = target_dqn(next_obss).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # [B]
#     q_targets = rewards + dqn.gamma * next_q
#     q_targets[dones] = rewards[dones]


#     # 5) Epsilon annealing -> now moved to main loop

#     # 6) Loss, back‑prop, optimizer step
#     loss = F.smooth_l1_loss(q_values, q_targets)

#     optimizer.zero_grad()
#     loss.backward()

#     # --- GRADIENT CHECK ----------------------
#     grad_norm = torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1)
#     if optimize.tick % 400 == 0:
#         print(f"[grad] ‖g‖₂ before clip = {grad_norm:6.4f}")
#     # -----------------------------------------

#     optimizer.step()

#     return loss.item()