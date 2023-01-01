from common import EMPTY_CELL, Player
import torch
import numpy as np
from adversaries.trained_adversary.pi import Pi

class Base():
    def __init__(self):
        self.best_only = True
    
    def set_mode(self, best):
        self.best_only = best

    def get_action(self, state):
        #print('MODE', self.best_only)
        if self.id != Player.P1:
            #print('*'*200)
            state = -state.copy()

        return self.model.best_action(state) if self.best_only else self.model.act(state)

class TrainedAdversaryV1(Base):
    def __init__(self, id, name='TrainedV1'):
        super().__init__()
        self.id = id
        self.name = name
        self.model = Pi(9,9)
        self.model.load_state_dict(torch.load('./adversaries/trained_adversary/models/v1.pt'))
        self.model.eval()