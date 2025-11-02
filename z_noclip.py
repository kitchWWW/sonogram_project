import pygame
import pyautogui
import mss
import numpy as np
import sounddevice as sd
import threading
import time

# === CONFIG ===
LINE_HEIGHT = 100
DISPLAY_WIDTH = 300
PIXEL_SIZE = 1
FPS = 20
VERTICAL_OFFSET = 10
SAMPLE_RATE = 44100
FRAME_DURATION = 1.0 / FPS  # seconds
BAND_WIDTH = 200  # Each pixel maps to 10 Hz band

# === PYGAME SETUP ===
pygame.init()
screen = pygame.display.set_mode((DISPLAY_WIDTH * PIXEL_SIZE, LINE_HEIGHT * PIXEL_SIZE))
pygame.display.set_caption("Pixel History + Sound")
clock = pygame.time.Clock()

# === MSS SETUP ===
sct = mss.mss()

# === AUDIO STATE ===
current_brightness = np.zeros(LINE_HEIGHT)  # Shared between threads
brightness_lock = threading.Lock()

def get_vertical_line(x, y, height=100, offset=10):
    monitor = {"top": y - offset - height + 1, "left": x, "width": 1, "height": height}
    img = sct.grab(monitor)
    arr = np.array(img)  # shape (height, 1, 4)
    # Normalize darkness (1 = black, 0 = white)
    return 1.0 - np.clip(np.sum(arr[:, 0, :3], axis=1) / (255.0 * 3), 0.0, 1.0)

def generate_band_limited_noise(freq_low, freq_high, duration, sample_rate):
    n_samples = int(duration * sample_rate)
    freqs = np.fft.rfftfreq(n_samples, 1/sample_rate)
    spectrum = np.zeros_like(freqs, dtype=complex)
    # Fill random values in the desired band
    mask = (freqs >= freq_low) & (freqs < freq_high)
    spectrum[mask] = np.random.randn(np.sum(mask)) + 1j * np.random.randn(np.sum(mask))
    # Make it symmetric and real
    waveform = np.fft.irfft(spectrum)
    waveform /= np.max(np.abs(waveform) + 1e-9)
    waveform /= 2
    return waveform

# Pre-generate 100 band-limited noise bases (repeating looped)
# Pre-generate longer noise loops (e.g., 2x expected block size)
NOISE_CACHE_DURATION = FRAME_DURATION * 200
long_noise_len = int(SAMPLE_RATE * NOISE_CACHE_DURATION)

noise_cache = []
for i in range(LINE_HEIGHT):
    f1 = i * BAND_WIDTH
    f2 = f1 + BAND_WIDTH
    base_noise = generate_band_limited_noise(f1, f2, NOISE_CACHE_DURATION, SAMPLE_RATE)
    noise_cache.append(base_noise)

noise_pos = np.zeros(LINE_HEIGHT, dtype=int)

def audio_callback(outdata, frames, time_info, status):
    global noise_pos

    with brightness_lock:
        raw_brightness = current_brightness.copy()

    block = np.zeros(frames, dtype=np.float32)

    for i in range(LINE_HEIGHT):
        buf   = noise_cache[i]
        start = noise_pos[i]
        end   = start + frames

        # wrap-around read
        if end <= long_noise_len:
            segment = buf[start:end]
        else:
            segment = np.concatenate((buf[start:], buf[:end - long_noise_len]))

        block += (1 - raw_brightness[i]) * segment
        noise_pos[i] = (start + frames) % long_noise_len

    outdata[:] = block.reshape(-1, 1)




# Start sound stream
stream = sd.OutputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=1, blocksize=int(SAMPLE_RATE * FRAME_DURATION))
stream.start()

# === VISUAL MEMORY ===
columns = []

running = True
while running:
    clock.tick(FPS)

    # Handle quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get current column of pixels
    x, y = pyautogui.position()
    brightness = get_vertical_line(x, y, LINE_HEIGHT, VERTICAL_OFFSET)

    # Update shared audio data
    with brightness_lock:
        current_brightness = brightness

    # Save to visual history
    rgb_pixels = [(int(255 * (1 - b),),) * 3 for b in brightness]
    columns.append(rgb_pixels)
    if len(columns) > DISPLAY_WIDTH:
        columns.pop(0)

    # Draw visual history
    screen.fill((0, 0, 0))
    for col_index, col in enumerate(columns):
        for row_index, color in enumerate(col):
            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(
                    col_index * PIXEL_SIZE,
                    row_index * PIXEL_SIZE,
                    PIXEL_SIZE,
                    PIXEL_SIZE
                )
            )
    pygame.display.flip()

# === CLEANUP ===
stream.stop()
stream.close()
pygame.quit()
