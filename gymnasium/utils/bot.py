import random

from gymnasium.connect4 import C4

class BotInterface:
    def calculate_best_move(self, game: C4) -> int:
        pass

class RandomBot(BotInterface):
    def calculate_best_move(self, game: C4) -> int:
        return game.move(random.choice(game.legal_moves))