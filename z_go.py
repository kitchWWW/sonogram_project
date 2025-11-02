import pygame
import pyautogui
from PIL import ImageGrab
import time


import mss
import numpy as np

sct = mss.mss()

def get_vertical_line_mss(x, y, height=100, offset=10):
    monitor = {"top": y - offset - height + 1, "left": x, "width": 1, "height": height}
    img = sct.grab(monitor)
    arr = np.array(img)  # shape: (height, 1, 4)
    return [tuple(arr[i, 0][:3]) for i in range(height)]  # Discard alpha









# Config
LINE_HEIGHT = 200
DISPLAY_WIDTH = 300
PIXEL_SIZE = 1  # Display scale factor (1 pixel per actual pixel)
FPS = 60
VERTICAL_OFFSET = 10

# Init display
pygame.init()
screen = pygame.display.set_mode((DISPLAY_WIDTH * PIXEL_SIZE, LINE_HEIGHT * PIXEL_SIZE))
pygame.display.set_caption("Pixel History Viewer")
clock = pygame.time.Clock()

# Store columns of pixel data (each is a list of RGB/RGBA tuples)
columns = []

running = True
while running:
    clock.tick(FPS)

    # Check for quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get mouse position and grab vertical line
    x, y = pyautogui.position()
    pixels = get_vertical_line_mss(x, y, LINE_HEIGHT, VERTICAL_OFFSET)

    # Store column, shift if needed
    columns.append(pixels)
    if len(columns) > DISPLAY_WIDTH:
        columns.pop(0)

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw all columns
    for col_index, col in enumerate(columns):
        for row_index, (r, g, b, *rest) in enumerate(col):  # Ignore alpha if present
            pygame.draw.rect(
                screen,
                (r, g, b),
                pygame.Rect(
                    col_index * PIXEL_SIZE,
                    row_index * PIXEL_SIZE,
                    PIXEL_SIZE,
                    PIXEL_SIZE
                )
            )

    pygame.display.flip()

pygame.quit()
