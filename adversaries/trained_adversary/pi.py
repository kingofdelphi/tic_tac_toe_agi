import torch
from torch import nn
from torch.distributions.categorical import Categorical
import numpy as np

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),

        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        
        pdparam = self.forward(x)
        for i in range(9):
            if state[i] != 0:
                pdparam[i] = -1e29
        pd = Categorical(logits=pdparam)
        action = pd.sample()
        idx=action.item()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return idx
