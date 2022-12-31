from enum import IntEnum
import numpy as np

class Player(IntEnum):
    P1=-1
    P2=1

EMPTY_CELL = 0

def opponent(turn):
    return Player.P1 if turn == Player.P2 else Player.P2

class GameState(IntEnum):
    Player1Win = 0
    Player2Win = 1
    Draw = 2
    NoFinishedYet = 3

# 0 1 2
# 3 4 5
# 6 7 8
def resolve_game_state(board):
    winner = None
    
    # Unrolling for faster training speeds
    if board[0] == board[1] == board[2] and board[0] != EMPTY_CELL:
        winner = board[0]

    if board[3] == board[4] == board[5] and board[3] != EMPTY_CELL:
        winner = board[3]

    if board[6] == board[7] == board[8] and board[6] != EMPTY_CELL:
        winner = board[6]

    if board[0]==board[3]==board[6] and board[0] != EMPTY_CELL:
        winner = board[0]

    if board[1]==board[4]==board[7] and board[1] != EMPTY_CELL:
        winner = board[1]

    if board[2]==board[5]==board[8] and board[2] != EMPTY_CELL:
        winner = board[2]

    if board[0] == board[4] == board[8] and board[0] != EMPTY_CELL: 
        winner = board[0]
    
    if board[2] == board[4] == board[6] and board[2] != EMPTY_CELL:
        winner = board[2]

    if winner:
        return GameState.Player1Win if winner == Player.P1 else GameState.Player2Win

    return GameState.NoFinishedYet if any(i == EMPTY_CELL for i in board) else GameState.Draw

def empty_board():
    return np.array([0]*9)

def empty_positions(board):
    return [i for i in range(9) if board[i] == EMPTY_CELL]

BOARD_MAP = { Player.P1: 'X', Player.P2: 'O', EMPTY_CELL: ' '}

def pretty_print_board(board, turn):
    board = [BOARD_MAP[i] for i in board]
    print(f'Turn: {turn}')
    print('')
    print(' {} | {} | {}'.format(*board[0:3]))
    print('-----------')
    print(' {} | {} | {}'.format(*board[3:6]))
    print('-----------')
    print(' {} | {} | {}'.format(*board[6:9]))

def make_move(board, position, turn, assert_not_occupied=True):
    if assert_not_occupied:
        assert board[position] == EMPTY_CELL

    # copy the board
    result = np.copy(board)
    result[position] = turn
    return result


def is_cell_occupied(board, position):
    return board[position] != EMPTY_CELL
