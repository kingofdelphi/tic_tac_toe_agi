from adversaries.human import HumanAdversary
from common import empty_board, pretty_print_board, EMPTY_CELL, Player, make_move, opponent, resolve_game_state, GameState
import numpy as np

def play(model):
    board = empty_board()

    AI = Player.P1
    turn = Player.P1
    status = GameState.NoFinishedYet
    human = HumanAdversary()

    while status == GameState.NoFinishedYet:
        pretty_print_board(board,turn)
        move = None
        if turn == AI:
            move = model.best_action(np.array(board))
            print('AI made a move in position', move)
        else:
            move = human.get_action(board, turn)
            print('Human made a move in position', move)
        occupied =  board[move] != EMPTY_CELL
        board = make_move(board, move, turn)
        if occupied:
            status = GameState.Player1Win if turn == Player.P2 else GameState.Player2Win
        else:
            status = resolve_game_state(board)
        turn = opponent(turn)

    pretty_print_board(board,turn)

    winner = { GameState.Player1Win: Player.P1, GameState.Player2Win: Player.P2 }.get(status)

    if status == GameState.Draw:
        print(f'It is a draw')
    elif winner == AI:
        print(f'AI wins')
    else:
        print('You win')
