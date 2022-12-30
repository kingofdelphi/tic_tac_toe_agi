from common import GameState, empty_positions, opponent, resolve_game_state
import numpy as np


class NotSoSmartAdversary():
    def get_action(self, state, turn):
        candidate_moves = empty_positions(state)
        win_action = None
        draw_action = None

        for position in candidate_moves:
            state[position] = turn
            endstate = resolve_game_state(state)
            state[position] = 0
            if endstate == GameState.Draw:
                draw_action = position
            elif endstate in [GameState.Player1Win, GameState.Player2Win]: #adversary must have won, NOTE: only one player in the list can win
                win_action = position

        action = None

        if win_action != None:
            return win_action

        if draw_action != None:
            return draw_action

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


