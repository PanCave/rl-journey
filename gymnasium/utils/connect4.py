from typing import List

ROWS = 6
COLUMNS = 7

class Connect4:
    def __init__(self) -> None:
        self._field = self._create_field()
        self._indexer = [0, 0, 0, 0, 0, 0, 0]
        self.player = 1 # alternating between 1 and -1
        self.game_over = False
        self.legal_moves = self._get_moves()
        self.history = []

    def print_field(self) -> None:
        for row in range(ROWS):
            for column in range(COLUMNS):
                field_entry = self._field[column][5 - row]
                print("o" if field_entry == 1 else "x" if field_entry == -1 else " ", " ", sep="", end="")
            print()
        print("______________")
        print("1 2 3 4 5 6 7 ")

    def move(self, move : int) -> bool:
        if move in self.legal_moves and self._indexer[move] < ROWS:
            self._field[move][self._indexer[move]] = self.player
            self.game_over = self._is_game_over(move)
            self.player *= -1
            self._indexer[move] += 1
            self.legal_moves = self._get_moves()
            self.history.append(move)
            return True
        return False

    @property
    def field(self) -> List[List[int]]:
        return self._field

    def undo(self) -> None:
        if len(self.history) == 0:
            return

        last_move = self.history[-1]
        self._indexer[last_move] -= 1
        self._field[last_move][self._indexer[last_move]] = 0
        self.player *= -1
        self.history.pop()


    def _is_game_over(self, x: int) -> bool:
        y = self._indexer[x]
        offsets = [[0, 1], [1, 1], [1, 0], [1, -1]]

        for offset in offsets:
            counter = 0
            for direction in [1, -1]:
                for step in range(1, 4):
                    step *= direction
                    new_x = x + offset[0] * step
                    new_y = y + offset[1] * step
                    if new_x < 0 or new_x > COLUMNS - 1 or new_y < 0 or new_y > ROWS - 1:
                        break
                    
                    if self._field[new_x][new_y] == self.player:
                        counter += 1
                    else:
                        break

                    if counter == 3:
                        return True
        return False


    def _get_moves(self) -> List[int]:
        legal_moves = []
        for column in range(COLUMNS):
            if self._indexer[column] < ROWS:
                legal_moves.append(column)
        
        return legal_moves

    def _create_field(self) -> List[List[int]]:
        field = []
        for _ in range(COLUMNS):
            field.append([0 for _ in range(ROWS)])
        
        return field