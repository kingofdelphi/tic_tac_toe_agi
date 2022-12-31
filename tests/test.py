if __name__ == '__main__':
    import sys
    from os.path import dirname
    sys.path.append(dirname(dirname(__file__)))


from common import empty_board, empty_positions, make_move, resolve_game_state, GameState, is_cell_occupied, opponent

def test_empty_board():
    assert list(empty_board()).count(0) == 9

def test_win_1():
    board = [
        1, 1, 1,
        -1, 0, 1,
        -1, 1, 1
    ]
    assert resolve_game_state(board) == GameState.Player2Win
    assert not is_cell_occupied(board, 4)
    assert empty_positions(board) == [4]

def test_win_2():
    board = [
        1, -1, 1,
        -1, -1, 0,
        -1, -1, 1
    ]
    assert is_cell_occupied(board, 1)
    assert not is_cell_occupied(board, 5)
    assert resolve_game_state(board) == GameState.Player1Win
    assert empty_positions(board) == [5]

def test_draw():
    board = [
        -1, -1, 1,
         1, 1, -1,
        -1, -1, 1
    ]
    assert resolve_game_state(board) == GameState.Draw
    assert empty_positions(board) == []

def test_move():
    board = [
        -1, -1, 1,
         0, 1, -1,
        -1, -1, 1
    ]
    new_board = make_move(board, 3, -1)
    assert new_board[3] == -1
    assert board[3] == 0

def test_opponent():
    assert opponent(1) == -1
    assert opponent(-1) == 1

test_empty_board()

test_win_1()
test_win_2()
test_draw()
test_move()
test_opponent()