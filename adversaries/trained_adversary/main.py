from common import EMPTY_CELL, Player
import torch
import numpy as np
from adversaries.trained_adversary.pi import Pi

class Base():
    def get_action(self, state):
        if self.id != Player.P1:
            state = -state.copy()

        return self.model.best_action(state)

class TrainedAdversaryV1(Base):
    def __init__(self, id, name='TrainedV1'):
        self.id = id
        self.name = name
        self.model = Pi(9,9)
        self.model.load_state_dict(torch.load('./adversaries/trained_adversary/models/v1.pt'))
        self.model.eval()