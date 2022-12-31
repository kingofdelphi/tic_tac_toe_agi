from pydoc import resolve
from adversaries.human import HumanAdversary
from adversaries.not_so_smart_adversary import NotSoSmartAdversary
from adversaries.trained_adversary.main import TrainedAdversaryV1
from adversaries.trained_adversary.min_max import MinMaxAdversary
from gym import Env
from gym.spaces import Box, Discrete
import numpy as np

from common import Player, GameState, empty_board, make_move
from common import resolve_game_state

class GameTag:
    WON = 'W'
    LOST = 'L'
    DRAW = 'D'
    GAME_RUNNING = ' '

# NEVER set zero scores as they're useless in the calculation, 0 signifies no information
REWARD_MAP = {
    GameTag.WON: 10,
    GameTag.LOST: -10,
    GameTag.DRAW: 0,
    GameTag.GAME_RUNNING: 0
}

class TicTacEnv(Env):
    def __init__(self):
        self.observation_space = Box(low=-1, high=1, shape=(9, 1))
        
        # amount of distance travelled 
        self.action_space = Discrete(9)

        self.me_player_id = Player.P1
        self.a1 = NotSoSmartAdversary(Player.P2)
        self.a2 = MinMaxAdversary(Player.P2)

        self.adversary = self.a1
        
        # current state 
        self.reset()

    
    def reset(self):
        self.state = empty_board()
        if np.random.rand() < 0.5:
            self.adversary = self.a1
        else:
            self.adversary = self.a2
        if np.random.rand() < 0.5:
            # simulate opponent starts first randomly
            self.state = make_move(self.state, self.adversary.get_action(self.state), self.adversary.id)
        return self.state


    def step(self, action):
        won = False
        lost = False
        draw = False

        self.state = make_move(self.state, action, self.me_player_id)

        gamestate = resolve_game_state(self.state)

        if gamestate == GameState.NoFinishedYet:
            counter_action = self.adversary.get_action(self.state)
            self.state = make_move(self.state, counter_action, self.adversary.id)

            gamestate = resolve_game_state(self.state)

        if gamestate == GameState.Player1Win:
            won = True
        elif gamestate == GameState.Player2Win:
            lost = True
        elif gamestate == GameState.Draw:
            draw = True

        game_status_tag = GameTag.GAME_RUNNING

        if won:
            game_status_tag = GameTag.WON
        elif lost:
            game_status_tag = GameTag.LOST
        elif draw:
            game_status_tag = GameTag.DRAW
            
        reward = REWARD_MAP[game_status_tag]
        done = won or lost or draw
        #print(self.state)
        #input('...')

        return self.state, reward, done, game_status_tag
