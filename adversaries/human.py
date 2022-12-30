from common import EMPTY_CELL

# Just takes action from input and returns it
class HumanAdversary():
    def get_action(self, state, turn):
        while True:
            try:
                print("Your move, cell indexed 1-9]:")
                move = int(input()) - 1 
                if state[move] == EMPTY_CELL:
                    return move
                print("Invalid move, select empty position")
            except Exception as e:
                print("Invalid move entered.", e)
