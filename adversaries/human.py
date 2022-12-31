from common import EMPTY_CELL, pretty_print_board

# Just takes action from input and returns it
class HumanAdversary():
    def __init__(self, id, name='Human'):
        self.id = id
        self.name = name

    def get_action(self, state):
        while True:
            try:
                pretty_print_board(state, self.name)
                print("Your move, cell indexed 1-9]:")
                move = int(input()) - 1 
                if state[move] == EMPTY_CELL:
                    return move
                print("Invalid move, select empty position")
            except Exception as e:
                print("Invalid move entered.", e)
