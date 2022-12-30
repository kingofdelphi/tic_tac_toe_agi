from common import EMPTY_CELL, GameState, opponent, EMPTY_CELL, resolve_game_state
import torch
import numpy as np

class TrainedAdversaryV1():
    def __init__(self, id, name='TrainedV1'):
        self.id = id
        self.name = name
        from train import Pi
        self.model = Pi(9,9)
        self.model.load_state_dict(torch.load('./adversaries/trained_adversary/models/v1.pt'), strict=False)
        self.model.eval()

    def get_action(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.model(x)

        # logic to IGNORE invalid moves. Moves using occupied cells are forbidden
        choices = []
        vals = []
        for i in range(9):
            if state[i] == 0:
                choices.append(i)
                vals.append(pdparam[i])

        action = np.argmax(np.array(vals))
        return choices[action]
