import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetworkConv(nn.Module):
    def __init__(self, in_chan, hidden_chan, out_chan, n_actions, n_rows, n_cols, hidden_layer, dropout_p):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.out_chan = out_chan

        self.c1 = nn.Conv2d(in_chan, hidden_chan, 7)
        self.c2 = nn.Conv2d(hidden_chan, out_chan, 7)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_chan, n_rows, n_cols)
            out = self.pool(self.c1(dummy_input))
            out = self.pool(self.c2(out))
            self.flattened_size = out.view(1, -1).shape[1]

        self.l1 = nn.Linear(self.flattened_size, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, n_actions)

        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        x = self.c1(x)
        x = self.pool(x)
        x = self.c2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x
    
class ActorCriticConv(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()  
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)

        return action_pred, value_pred
