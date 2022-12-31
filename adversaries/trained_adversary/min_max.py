from common import Player, empty_board, empty_positions, resolve_game_state, GameState
from collections import defaultdict
import numpy as np

# Just takes action from input and returns it
class MinMaxAdversary():
    def __init__(self, id, name='Human'):
        self.id = id
        self.name = name
        self.dp = defaultdict(lambda: None)
        self.build(empty_board(), Player.P1)
        self.build(empty_board(), Player.P2)
    
    def enc(self, board, turn):
        enc=0
        for i in board:
            enc = enc*3+i+1
        enc=enc*3+turn+1
        return enc

    def build(self, board, turn):
        enc = self.enc(board, turn)
        if self.dp[enc] != None:
            return self.dp[enc]
        state = resolve_game_state(board)
        if state == GameState.Draw:
            self.dp[enc] = 0
            return self.dp[enc]
        elif state != GameState.NoFinishedYet:
            self.dp[enc] = -1
            return self.dp[enc]

        cand = empty_positions(board)
        score = -1
        for pos in cand:
            board[pos] = turn
            score = max(score, -self.build(board, -turn))
            board[pos] = 0
        self.dp[enc] = score
        return self.dp[enc]

    def get_action(self, state):
        cand = empty_positions(state)
        win = []
        draw = []
        lose = []

        for pos in cand:
            state[pos] = self.id
            r = self.dp[self.enc(state,-self.id)]
            state[pos] = 0
            if r == -1:
                win.append(pos)
            elif r == 0:
                draw.append(pos)
            else:
                lose.append(pos)

        if win: return np.random.choice(win)
        if draw: return np.random.choice(draw)
        return np.random.choice(lose)
