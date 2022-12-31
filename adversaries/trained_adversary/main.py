from common import EMPTY_CELL, Player
import torch
import numpy as np
from adversaries.trained_adversary.pi import Pi

class Base():
    def get_action(self, state):
        if self.id != Player.P1:
            state = -state.copy()
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.model(x)

        # logic to IGNORE invalid moves. Moves using occupied cells are forbidden
        choices = []
        vals = []
        for i in range(9):
            if state[i] == 0:
                choices.append(i)
                vals.append(pdparam[i])

        vals = np.array(vals, dtype=np.float32)
        action = np.random.choice(np.flatnonzero(vals == vals.max()))
        action = choices[action]

        assert state[action] == EMPTY_CELL

        return action

class TrainedAdversaryV1(Base):
    def __init__(self, id, name='TrainedV1'):
        self.id = id
        self.name = name
        self.model = Pi(9,9)
        self.model.load_state_dict(torch.load('./adversaries/trained_adversary/models/v1.pt'))
        self.model.eval()