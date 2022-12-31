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
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
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

    def get_logits(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)

        # logic to IGNORE invalid moves. Moves using occupied cells are forbidden
        # invalid action masking to avoid sampling invalid actions
        return pdparam.where(x == 0, torch.tensor(-1e18))

    def best_action(self, state):
        pdparam = torch.nn.functional.softmax(self.get_logits(state))

        return np.random.choice(np.flatnonzero(pdparam == pdparam.max()))

    def act(self, state):
        logits = self.get_logits(state)

        pd = Categorical(logits=logits)
        action = pd.sample()
        idx=action.item()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return idx
