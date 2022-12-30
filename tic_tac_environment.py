from pydoc import resolve
from adversaries.not_so_smart_adversary import NotSoSmartAdversary
from gym import Env
from gym.spaces import Box, Discrete
import numpy as np

from common import Player, GameState, EMPTY_CELL, empty_board, make_move
from common import resolve_game_state

class GameTag:
    WON = 'W'
    LOST = 'L'
    DRAW = 'D'
    LOST_BY_INVALID_MOVE = '-'
    GAME_RUNNING = ' '

# NEVER set zero scores as they're useless in the calculation, 0 signifies no information
REWARD_MAP = {
    GameTag.LOST_BY_INVALID_MOVE: -10,
    GameTag.WON: 10,
    GameTag.LOST: -5,
    GameTag.DRAW: 5,
    GameTag.GAME_RUNNING: 0
}

class TicTacEnv(Env):
    def __init__(self):
        self.observation_space = Box(low=-1, high=1, shape=(9, 1))
        
        # amount of distance travelled 
        self.action_space = Discrete(9)

        self.me_player_id = Player.P1
        self.adversary = NotSoSmartAdversary(Player.P2)
        
        # current state 
        self.reset()

    
    def reset(self):
        self.state = empty_board()
        return self.state


    def step(self, action):
        won = False
        lost = False
        draw = False

        invalid_move = self.state[action] != EMPTY_CELL
        self.state = make_move(self.state, action, self.me_player_id)

        if invalid_move: # already used cell
            lost = True
        else:
            gamestate = resolve_game_state(self.state)

            if gamestate == GameState.NoFinishedYet:
                counter_action = self.adversary.get_action(self.state)
                assert self.state[counter_action] == EMPTY_CELL
                self.state = make_move(self.state, counter_action, self.adversary.id)

                gamestate = resolve_game_state(self.state)

            if gamestate == GameState.Player1Win:
                won = True
            elif gamestate == GameState.Player2Win:
                lost = True
            elif gamestate == GameState.Draw:
                draw = True

        game_status_tag = GameTag.GAME_RUNNING

        if invalid_move:
            game_status_tag = GameTag.LOST_BY_INVALID_MOVE
        elif won:
            game_status_tag = GameTag.WON
        elif lost:
            game_status_tag = GameTag.LOST
        elif draw:
            game_status_tag = GameTag.DRAW
            
        reward = REWARD_MAP[game_status_tag]
        done = won or lost or draw

        return self.state, reward, done, game_status_tag
