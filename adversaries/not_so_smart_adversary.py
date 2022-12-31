from common import EMPTY_CELL, GameState, empty_positions, opponent, resolve_game_state
import numpy as np


class NotSoSmartAdversary():
    def __init__(self, id, name='NotSoSmart'):
        self.id = id
        self.name = name

    def _get_action(self, state):
        turn = self.id
        candidate_moves = empty_positions(state)
        win_action = []
        draw_action = []

        for position in candidate_moves:
            state[position] = turn
            endstate = resolve_game_state(state)
            state[position] = 0
            if endstate == GameState.Draw:
                draw_action.append(position)
            elif endstate in [GameState.Player1Win, GameState.Player2Win]: #adversary must have won, NOTE: only one player in the list can win
                win_action.append(position)

        if win_action:
            return np.random.choice(win_action)

        if draw_action:
            return np.random.choice(draw_action)

        # check if the opponent of adversary is trying to win, we want to block such moves to prevent from losing
        prevent_opponent_win_moves = []
        for position in candidate_moves:
            state[position] = opponent(turn)
            endstate = resolve_game_state(state)
            state[position] = 0

            if endstate in [GameState.Player1Win, GameState.Player2Win]: # opponent
                prevent_opponent_win_moves.append(position)

        if prevent_opponent_win_moves:
            return np.random.choice(prevent_opponent_win_moves)

        action = np.random.choice(candidate_moves)
        
        return action

    def get_action(self, state):
        action = self._get_action(state)
        assert state[action] == EMPTY_CELL
        return action

