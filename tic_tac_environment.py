from pydoc import resolve
from adversaries.not_so_smart_adversary import NotSoSmartAdversary
from gym import Env
from gym.spaces import Box, Discrete
import numpy as np

from common import Player, GameState, EMPTY_CELL, empty_board, make_move
from common import resolve_game_state

# NEVER set zero scores as they're useless in the calculation, 0 signifies no information
SCORE_MAP = {
    '-': -10,
    'W': 10,
    'L': -5,
    'D': 5
}

class TicTacEnv(Env):
    def __init__(self):
        self.observation_space = Box(low=-1, high=1, shape=(9, 1))
        
        # amount of distance travelled 
        self.action_space = Discrete(9)

        self.adversary = NotSoSmartAdversary()
        
        # current state 
        self.reset()

    
    def reset(self):
        self.state = empty_board()
        return self.state


    def step(self, action):
        won = False
        lost = False
        draw = False

        p1 = Player.P1
        opponent = Player.P2

        invalid_move = self.state[action] != EMPTY_CELL
        self.state = make_move(self.state, action, p1)

        if invalid_move: # already used cell
            lost = True
        else:
            gamestate = resolve_game_state(self.state)

            if gamestate == GameState.NoFinishedYet:
                counter_action = self.adversary.get_action(self.state, opponent)
                assert self.state[counter_action] == EMPTY_CELL
                self.state = make_move(self.state, counter_action, opponent)

                gamestate = resolve_game_state(self.state)

            if gamestate == GameState.Player1Win:
                won = True
            elif gamestate == GameState.Player2Win:
                lost = True
            elif gamestate == GameState.Draw:
                draw = True

        tag = ''

        if invalid_move:
            tag='-'
        elif won:
            tag='W'
        elif lost:
            tag='L'
        elif draw:
            tag='D'
            
        reward = SCORE_MAP.get(tag, 0)
        done = won or lost or draw

        return self.state, reward, done, tag
