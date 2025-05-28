import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()  

        self.l1 = nn.Linear(input_dim, 64)
        self.h1 = nn.ReLU()

        self.policy_head = nn.Linear(64, output_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.h1(x)

        logits = self.policy_head(x)
        value = self.value_head(x)

        return logits, value