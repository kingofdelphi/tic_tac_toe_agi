from adversaries.human import HumanAdversary
from adversaries.trained_adversary.main import TrainedAdversaryV1
from common import empty_board, is_cell_occupied, pretty_print_board, EMPTY_CELL, Player, make_move, opponent, resolve_game_state, GameState
import numpy as np

def play():
    board = empty_board()

    ai = TrainedAdversaryV1(Player.P1, 'AI')
    human = HumanAdversary(Player.P2, 'Human')

    turn = ai
    p1, p2 = ai, human

    status = GameState.NoFinishedYet

    while status == GameState.NoFinishedYet:
        turn_id = p1.id
        pretty_print_board(board, turn_id)
        move = p1.get_action(board)
        print(f'{p1.name} made a move in position', move)

        occupied =  is_cell_occupied(board, move)
        board = make_move(board, move, turn_id)

        if occupied:
            status = GameState.Player1Win if turn_id == Player.P2 else GameState.Player2Win
        else:
            status = resolve_game_state(board)

        # swap turns
        p1, p2 = p2, p1

    pretty_print_board(board,turn)

    winner = { GameState.Player1Win: Player.P1, GameState.Player2Win: Player.P2 }.get(status)

    if status == GameState.Draw:
        print(f'It is a draw')
    elif winner == ai.id:
        print(f'AI wins')
    else:
        print('HUman wins')


if __name__ == '__main__':
    play()