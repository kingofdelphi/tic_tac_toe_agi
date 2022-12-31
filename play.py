from adversaries.human import HumanAdversary
from adversaries.not_so_smart_adversary import NotSoSmartAdversary
from adversaries.trained_adversary.main import TrainedAdversaryV1
from adversaries.trained_adversary.min_max import MinMaxAdversary
from common import empty_board, is_cell_occupied, pretty_print_board, EMPTY_CELL, Player, make_move, opponent, resolve_game_state, GameState
import numpy as np

def play():
    board = empty_board()

    p1 = TrainedAdversaryV1(Player.P1, 'AIv1 I')
    #p2 = MinMaxAdversary(Player.P2, 'AIMinMax')
    #p2 = NotSoSmartAdversary(Player.P2, 'Brut')
    #p2 = TrainedAdversaryV1(Player.P2, 'AIv1 II')
    p2 = HumanAdversary(Player.P2, 'Human')

    # p1, p2 = p2, p1
    status = GameState.NoFinishedYet

    while status == GameState.NoFinishedYet:
        turn_id = p1.id
        pretty_print_board(board, p1.name)
        move = p1.get_action(board)
        print(f'{p1.name} made a move in position', move+1)

        occupied =  is_cell_occupied(board, move)
        board = make_move(board, move, turn_id)

        if occupied:
            status = GameState.Player1Win if turn_id == Player.P2 else GameState.Player2Win
        else:
            status = resolve_game_state(board)

        # swap turns
        p1, p2 = p2, p1

    pretty_print_board(board,p1.name)

    winner = { GameState.Player1Win: Player.P1, GameState.Player2Win: Player.P2 }.get(status)

    if status == GameState.Draw:
        print(f'It is a draw')
    elif winner == p1.id:
        print(f'{p1.name} wins')
    else:
        print(f'{p2.name} wins')


if __name__ == '__main__':
    while True:
        play()
        input('Pause...')