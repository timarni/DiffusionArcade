import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_p):
        super().__init__()

        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_p)

        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.l2(x)

        return x
    
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()  
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)

        return action_pred, value_pred
