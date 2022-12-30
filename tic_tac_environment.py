from pydoc import resolve
from gym import Env
from gym.spaces import Box, Discrete
import numpy as np

from common import Player, GameState, opponent, EMPTY_CELL
from common import resolve_game_state

# NEVER set zero scores as they're useless in the calculation, 0 signifies no information
SCORE_MAP = {
    '-': -10,
    'W': 10,
    'L': -5,
    'D': 5
}

def adversary_move(state, turn):
    candidate_moves = [i for i in range(9) if state[i] == EMPTY_CELL]
    win_action = None
    draw_action = None

    for i in candidate_moves:
        state[i] = turn
        endstate = resolve_game_state(state)
        state[i] = 0
        if endstate == GameState.Draw:
            draw_action = i
        elif endstate in [GameState.Player1Win, GameState.Player2Win]: #adversary must have won, NOTE: only one player in the list can win
            win_action = i

    action = None

    if win_action != None:
        return win_action
    if draw_action != None:
        return draw_action

    # check if the opponent of adversary is trying to win, we want to block such moves to prevent from losing
    prevent_opponent_win_moves = []
    for i in candidate_moves:
        state[i] = opponent(turn)
        endstate = resolve_game_state(state)
        state[i] = 0
        if endstate in [GameState.Player1Win, GameState.Player2Win]: # opponent
            prevent_opponent_win_moves.append(i)

    if prevent_opponent_win_moves:
        return np.random.choice(prevent_opponent_win_moves)

    action = np.random.choice(candidate_moves)
    
    return action

class TicTacEnv(Env):
    def __init__(self):
        self.observation_space = Box(low=-1, high=1, shape=(9, 1))
        
        # amount of distance travelled 
        self.action_space = Discrete(9)
        
        # current state 
        self.reset()

    
    def reset(self):
        self.state = np.array([EMPTY_CELL]*9)
        return self.state


    def step(self, action):
        self.state = np.copy(self.state)
        won = False
        lost = False
        draw = False

        p1 = Player.P1
        opp = Player.P2

        invalid_move = self.state[action] != EMPTY_CELL
        self.state[action] = p1 # set the cell value, override even if invalid

        if invalid_move: # already used cell
            lost = True
        else:
            endstate = resolve_game_state(self.state)

            if endstate == GameState.NoFinishedYet:
                counter_action = adversary_move(self.state, opp)
                assert self.state[counter_action] == EMPTY_CELL
                self.state[counter_action] = opp

                endstate = resolve_game_state(self.state)

            if endstate == GameState.Player1Win:
                won = True
            elif endstate == GameState.Player2Win:
                lost = True
            elif endstate == GameState.Draw:
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
