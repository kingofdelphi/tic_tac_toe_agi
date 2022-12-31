from adversaries.human import HumanAdversary
from adversaries.not_so_smart_adversary import NotSoSmartAdversary
from adversaries.trained_adversary.main import TrainedAdversaryV1
from adversaries.trained_adversary.min_max import MinMaxAdversary
from common import empty_board, pretty_print_board, Player, make_move, resolve_game_state, GameState

def min_max_vs_human():
    p1 = MinMaxAdversary(Player.P2, 'AIMinMax')
    p2 = HumanAdversary(Player.P1, 'Human')

    return p1, p2

def trained_ai_vs_min_max():
    p1 = TrainedAdversaryV1(Player.P1, 'AIv1 I')
    p2 = MinMaxAdversary(Player.P2, 'AIMinMax')

    return p1, p2

def trained_ai_vs_human():
    p1 = TrainedAdversaryV1(Player.P1, 'AIv1 I')
    p2 = HumanAdversary(Player.P2, 'Human')

    return p1, p2

def play():
    board = empty_board()

    p1, p2 = trained_ai_vs_min_max()
    #p1, p2 = trained_ai_vs_human()
    #p1, p2 = min_max_vs_human()

    p1, p2 = p2, p1 #swap for starting player toggle
    status = GameState.NoFinishedYet

    while status == GameState.NoFinishedYet:
        pretty_print_board(board, p1.name)
        move = p1.get_action(board)
        print(f'{p1.name} made a move in position', move+1)

        board = make_move(board, move, p1.id)

        status = resolve_game_state(board)

        # swap turns
        p1, p2 = p2, p1

    pretty_print_board(board, p1.name)

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