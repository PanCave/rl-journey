import random

from gymnasium.utils.bot import BotInterface
from gymnasium.connect4 import C4

NUMBER_FUTURE_GAMES = 1000
NUMBER_RANDOM_MOVES = 100

class MonteCarloBot(BotInterface):
    def __init__(self, bot_number: int):
        self._bot_number = bot_number


    def calculate_best_move(self, game: C4) -> int:
        best_move = -1
        best_move_score = 100

        for move in game.legal_moves:
            wins = 0
            losses = 0

            game.move(move)
            
            # Check if this move immediately ends the game
            if game.game_over:
                # If game is over, the one who just moved (not game.player) won
                # That would be -game.player (the opposite of current player)
                if -game.player == self._bot_number:
                    wins = NUMBER_FUTURE_GAMES  # Winning move, max score
                else:
                    losses = NUMBER_FUTURE_GAMES  # Losing move, worst score
            else:
                for _ in range(NUMBER_FUTURE_GAMES):
                    moves_made = 0
                    for _ in range(NUMBER_RANDOM_MOVES):
                        if game.game_over:
                            break
                        
                        game.move(random.choice(game.legal_moves))
                        moves_made += 1
                        
                        if game.game_over:
                            # The player who just moved won (opposite of game_copy.player)
                            winner = -game.player
                            if winner == self._bot_number:
                                wins += 1
                            else:
                                losses += 1
                    
                    for _ in range(moves_made):
                        game.undo()

                print(wins, losses)
                score = wins - losses
                if score >= best_move_score:
                    best_move = move
                    best_move_score = score

            game.undo()
        
        return best_move
