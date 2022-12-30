from common import pretty_print_board, make_move_1d, EMPTY_CELL, Player, make_move_1d, opponent, resolve_game_state, GameState
import tic_tac_environment
import numpy as np

def play(model):
    board = [EMPTY_CELL]*9
    AI = Player.P1
    turn = Player.P1
    status = GameState.NoFinishedYet
    while status == GameState.NoFinishedYet:
        pretty_print_board(board,turn)
        move = None
        if turn == AI:
            move = model.best_action(np.array(board))
            print('AI made a move in position', move)
        else:
            move = None
            while move == None:
                try:
                    print("Your move:")
                    #move = tic_tac_environment.adversary_move(board, turn)
                    move = int(input()) - 1 
                    if board[move] != EMPTY_CELL:
                        print("Invalid move, select empty position")
                        move = None
                except Exception as e:
                    print("Invalid move entered.", e)
        occupied =  board[move] != EMPTY_CELL
        board = make_move_1d(board, move, turn)
        if occupied:
            status = GameState.Player1Win if turn == Player.P2 else GameState.Player2Win
        else:
            status = resolve_game_state(board)
        turn = opponent(turn)

    pretty_print_board(board,turn)
    if status == GameState.Draw:
        print(f'It is a draw')
    else:
        winner = Player.P1 if status == GameState.Player1Win else Player.P2
        if winner == AI:
            print(f'AI wins')
        else:
            print('You win')
