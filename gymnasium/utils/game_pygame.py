import sys
import time

import pygame

from gymnasium.connect4 import C4
from gymnasium.utils.MonteCarloBot import MonteCarloBot

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREY = (100, 100, 100)
DARK_GREY = (50, 50, 50)
DARK_DARK_GREY = (25, 25, 25)

# Screen dimensions
WIDTH, HEIGHT = 800, 800
ROWS = 6
COLUMNS = 7

# Primitives dimension
CIRCLE_RADIUS = 45

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

c4 = C4()
# for i in range(random.randrange(10) + 5):
#     c4.move(random.choice(c4.legal_moves))
bot = MonteCarloBot(-1)


field = c4.field
running = True
while running:
    screen.fill(DARK_DARK_GREY)

    # if c4.player == -1:
    #     c4_copy = copy.deepcopy(c4)
    #     bot_move = bot.calculate_best_move(c4_copy)
    #     print(bot_move)
    #     if (bot_move != -1):
    #         c4.move(bot_move)
    #     else:
    #         print("Bot has no legal moves")
    #         c4.move(random.choice(c4.legal_moves))
    #     if c4.game_over:
    #         running = False

    mouse_pos = pygame.mouse.get_pos()
    slot = (mouse_pos[0] - 50) // 100
    if slot >= 0 and slot < 7:
        pygame.draw.circle(screen, YELLOW if c4.player == 1 else RED, (slot * 100 + 100, 100), CIRCLE_RADIUS)

    for event in pygame.event.get():
        left, middle, right = pygame.mouse.get_pressed()
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # if c4.player != 1:
            #     continue

            if left:
                c4.move(slot)
                if c4.game_over:
                    running = False
            elif right:
                c4.undo()

    board_rect = pygame.Rect(25, 125, 750, 700)
    pygame.draw.rect(screen, BLUE, board_rect, border_top_left_radius=10, border_top_right_radius=10)

    for row in range(ROWS):
        for column in range(COLUMNS):
            a = field[column]
            color = YELLOW if a[5 - row] == 1 else RED if a[5 - row] == -1 else DARK_GREY
            pygame.draw.circle(screen, color, (column * 100 + 100, row * 100 + 200), CIRCLE_RADIUS)

    pygame.display.flip()

time.sleep(2)

pygame.quit()
sys.exit()