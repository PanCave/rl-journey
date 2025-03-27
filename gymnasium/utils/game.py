import random
import time

from enum import Enum
from gymnasium.utils.bot import BotInterface
from gymnasium.utils.MonteCarloBot import MonteCarloBot

from gymnasium.connect4 import C4

class Mode(Enum):
    LOCAL_COOP = 0
    BOT = 1

def play_against_bot(c4: C4, bot: BotInterface):
    player = random.choice([-1, 1])
    c4.print_field()
    while not c4.game_over:
        if c4.player == player:
            move = int(input("Enter column: ")) - 1
            c4.move(move)
            c4.print_field()
        else:
            c4.move(bot.calculate_best_move(c4))
            c4.print_field()
        time.sleep(1)
    

def play_local_coop(c4: C4):
    c4.print_field()
    while not c4.game_over:
        move = int(input("Enter column: ")) - 1
        c4.move(move)
        c4.print_field()
        time.sleep(1)

c4 = C4()

bot_choice = input("Do you want to play in local [c]oop or against a [b]ot? ")
mode = Mode.LOCAL_COOP if bot_choice == "c" else Mode.BOT

if mode == Mode.LOCAL_COOP:
    play_local_coop(c4)
else:
    bot = MonteCarloBot()
    play_against_bot(c4, bot)