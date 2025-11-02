import pygame
import pyautogui
import mss
import numpy as np
import sounddevice as sd
import threading
import time


WATERFALL_BORDER = True
LOWER_THRESH     = 0.4
UPPER_THRESH     = 0.9

# === WINDOW / LAYOUT CONFIG ===
WINDOW_WIDTH  = 1300
WINDOW_HEIGHT = 900
WINDOW_BG     = (0, 0, 0)

WATERFALL_X = 40
WATERFALL_Y = 60

# === WATERFALL DISPLAY CONFIG (visual grid only) ===
# Separate the grid geometry from the overall window size
WATERFALL_ROWS       = 100   # vertical bins (was LINE_HEIGHT)
WATERFALL_COLS       = 300   # history width (was DISPLAY_WIDTH)
WATERFALL_CELL_SIZE  = 2     # pixel size per cell (was PIXEL_SIZE)
WATERFALL_BORDER_RGB = (50, 50, 50)

FPS = 30
FRAME_DURATION = 1.0 / FPS

# === SCREEN CAPTURE CONFIG ===
VERTICAL_OFFSET = 3
# Sample exactly as many rows as we draw & sonify so the mapping is 1:1
VERTICAL_SAMPLING_HEIGHT = WATERFALL_ROWS

# === AUDIO CONFIG ===
SAMPLE_RATE = 44100
GAIN_ALPHA  = 0.4   # one-pole smoothing on per-band gains
MASTER_GAIN = 0.9   # overall audio trim

# === PYGAME SETUP ===
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pixel Brightness History Sonification")
clock = pygame.time.Clock()

# Fonts for HUD
FONT_SMALL = pygame.font.SysFont(None, 18)
FONT_MED   = pygame.font.SysFont(None, 22)

# Precomputed rects (handy for HUD layout)
WATERFALL_WIDTH  = WATERFALL_COLS * WATERFALL_CELL_SIZE
WATERFALL_HEIGHT = WATERFALL_ROWS * WATERFALL_CELL_SIZE
WATERFALL_RECT   = pygame.Rect(WATERFALL_X, WATERFALL_Y, WATERFALL_WIDTH, WATERFALL_HEIGHT)

HUD_TOP_H   = 48
HUD_RIGHT_W = 320
HUD_TOP_RECT   = pygame.Rect(0, 0, WINDOW_WIDTH, HUD_TOP_H)
HUD_RIGHT_RECT = pygame.Rect(WATERFALL_X, 500, HUD_RIGHT_W, 320)

# === MSS SETUP ===
sct = mss.mss()

# === AUDIO STATE ===
N_BANDS = WATERFALL_ROWS * 2 # keep audio bands = waterfall rows
current_brightness = np.zeros(N_BANDS, dtype=np.float32)  # shared between threads
brightness_lock = threading.Lock()
smoothed_gain = np.zeros(N_BANDS, dtype=np.float32)

# Audio level for HUD meter
audio_level = 0.0
audio_level_lock = threading.Lock()

# === TRACK/FREEZE STATE ===
FOLLOW_MOUSE = True           # follows the mouse until frozen
TRACK_X, TRACK_Y = 0, 0       # last known / frozen coordinates

# === MOUSE REGION VIEW CONFIG ===
MOUSE_VIEW_W = 300
MOUSE_VIEW_H = 400
MOUSE_VIEW_ABOVE = 250
MOUSE_VIEW_BELOW = 150  # MOUSE_VIEW_ABOVE + BELOW must equal MOUSE_VIEW_H

# Position the preview below the waterfall, aligned left with it
MOUSE_VIEW_X = WATERFALL_X + (WATERFALL_COLS * WATERFALL_CELL_SIZE) + 20
MOUSE_VIEW_Y = WATERFALL_Y
MOUSE_VIEW_BORDER_RGB = (80, 80, 80)

# === THRESHOLD SCALING / MOVEMENT MASK ===
# Map inputs in [LOWER_THRESH, UPPER_THRESH] -> [0, 1], clamp outside.
# If your input is 0..255 instead of 0..1, set INPUT_RANGE_MAX = 255.0
INPUT_RANGE_MAX  = 1.0  # change to 255.0 if your brightness is 0..255

# Only show movement: if current value is within ±SIMILAR_TO_PREV of previous value,
# write 0 (black). Toggle with ONLY_SHOW_MOVEMENT.
ONLY_SHOW_MOVEMENT = False 
SIMILAR_TO_PREV    = 0.10    # in normalized 0..1 units, after threshold scaling

# Internal memory for movement mask (previous *scaled* column)
_prev_scaled = None

def apply_threshold_scale(x, lo=LOWER_THRESH, hi=UPPER_THRESH, in_max=INPUT_RANGE_MAX):
    """
    Piecewise linear scale with clipping:
      x <= lo  -> 0
      x >= hi  -> 1
      between  -> linear (keeps midpoints like 0.5 unchanged when lo=0.2, hi=0.8)
    Then, if ONLY_SHOW_MOVEMENT is True, zero any entries whose change from the previous
    frame is within ±SIMILAR_TO_PREV (after scaling).
    Accepts numpy arrays. Returns 0..1 float array.
    """
    global _prev_scaled

    # Normalize input to 0..1 (handles both 0..1 and 0..255 cases)
    x01 = np.clip(x / float(in_max), 0.0, 1.0)

    # Linear threshold scaling to 0..1
    if hi <= lo:
        y_scaled = x01  # safety: misconfig -> no re-scaling
    else:
        y_scaled = (x01 - lo) / (hi - lo)
        y_scaled = np.clip(y_scaled, 0.0, 1.0)

    # Movement mask (compare to previous *unmasked* scaled values)
    if ONLY_SHOW_MOVEMENT:
        if _prev_scaled is not None and _prev_scaled.shape == y_scaled.shape:
            y_out = y_scaled.copy()
            still_mask = np.abs(y_scaled - _prev_scaled) <= SIMILAR_TO_PREV
            y_out[still_mask] = 1.0
        else:
            y_out = y_scaled.copy()
        _prev_scaled = y_scaled
        return y_out
    else:
        _prev_scaled = y_scaled
        return y_scaled

def get_vertical_line(x, y, height=100, offset=10):
    """Capture a 1px wide, `height` tall strip and return normalized darkness [0..1]."""
    h = int(height)  # ensure integer for MSS
    monitor = {"top": int(y) - offset - h + 1, "left": int(x), "width": 1, "height": h}
    img = sct.grab(monitor)
    arr = np.array(img)  # shape (h, 1, 4)
    # Normalize darkness (1 = black, 0 = white)
    return 1.0 - np.clip(np.sum(arr[:, 0, :3], axis=1) / (255.0 * 3), 0.0, 1.0)

def generate_band_limited_noise(freq_low, freq_high, duration, sample_rate):
    n_samples = int(duration * sample_rate)
    freqs = np.fft.rfftfreq(n_samples, 1 / sample_rate)
    spectrum = np.zeros_like(freqs, dtype=complex)
    mask = (freqs >= freq_low) & (freqs < freq_high)
    spectrum[mask] = np.random.randn(np.sum(mask)) + 1j * np.random.randn(np.sum(mask))
    waveform = np.fft.irfft(spectrum)
    waveform /= (np.max(np.abs(waveform)) + 1e-9)
    waveform /= 5.0
    return waveform.astype(np.float32)

# === PRE-GENERATE NOISE CACHE (log-spaced bands) ===
NOISE_CACHE_DURATION = FRAME_DURATION * 200  # long enough to loop
long_noise_len = int(SAMPLE_RATE * NOISE_CACHE_DURATION)

MIN_FREQ = 20.0
MAX_FREQ = SAMPLE_RATE / 2.0  # Nyquist

band_edges = np.logspace(
    np.log10(MIN_FREQ),
    np.log10(MAX_FREQ),
    num=N_BANDS + 1
)

noise_cache = []
for i in range(N_BANDS):
    f1 = band_edges[i]
    f2 = band_edges[i + 1]
    base_noise = generate_band_limited_noise(f1, f2, NOISE_CACHE_DURATION, SAMPLE_RATE)
    noise_cache.append(base_noise)

noise_pos = np.zeros(N_BANDS, dtype=int)

def audio_callback(outdata, frames, time_info, status):
    global noise_pos, smoothed_gain, audio_level

    # Snapshot target gains safely
    with brightness_lock:
        raw = current_brightness.copy()

    # Match length
    if raw.shape[0] != N_BANDS:
        raw = raw[:N_BANDS] if raw.shape[0] > N_BANDS else np.pad(raw, (0, N_BANDS - raw.shape[0]))

    # --- De-zipper envelope ---
    g_start = smoothed_gain.copy()                           # start-of-block gains
    g_end   = g_start + GAIN_ALPHA * (raw - g_start)         # one-pole update for end-of-block
    smoothed_gain = g_end.copy()                             # persist for next callback

    # Per-sample interpolation factor [0..1) across the block
    t = np.arange(frames, dtype=np.float32) / max(frames, 1)

    block = np.zeros(frames, dtype=np.float32)

    # Sum bands using sample-accurate envelope
    for i in range(N_BANDS):
        buf   = noise_cache[N_BANDS - i - 1]
        start = noise_pos[i]
        end   = start + frames
        if end <= long_noise_len:
            segment = buf[start:end]
        else:
            segment = np.concatenate((buf[start:], buf[:end - long_noise_len]))

        # Interpolated gain for this band across the block:
        # We map "brightness" (your scaled darkness) to volume as (1 - gain)
        gain_line = g_start[i] + (g_end[i] - g_start[i]) * t
        amp = 1.0 - gain_line

        block += amp * segment
        noise_pos[i] = end % long_noise_len

    # Constant-power normalization for many bands
    MIX_NORM = 1.0 / np.sqrt(N_BANDS)
    block *= MIX_NORM

    # Master trim
    block *= MASTER_GAIN

    # Soft limiter + clip guard
    DRIVE = 2.0
    block = np.tanh(DRIVE * block) / np.tanh(DRIVE)
    block = np.clip(block, -0.98, 0.98)

    # HUD RMS
    lvl = float(np.sqrt(np.mean(block.astype(np.float64)**2)))
    with audio_level_lock:
        audio_level = lvl

    outdata[:] = block.reshape(-1, 1)

# Start sound stream
stream = sd.OutputStream(
    callback=audio_callback,
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=int(SAMPLE_RATE * FRAME_DURATION)
)
stream.start()

# === WATERFALL VISUAL STATE ===
# Keep a rolling history of columns; each column is a list of RGB tuples for each row
wf_columns = []

def update_waterfall_history(wf_columns, brightness, max_cols):
    """
    Convert brightness (0..1 darkness) to grayscale RGB and append as a new column.
    Keep at most max_cols columns.
    """
    col = [(int(255 * (1.0 - b)),) * 3 for b in brightness]
    wf_columns.append(col)
    if len(wf_columns) > max_cols:
        wf_columns.pop(0)

def draw_waterfall(surface, columns, pos, cell_size, draw_border=False, border_rgb=(50, 50, 50)):
    """
    Draw the waterfall grid at a given top-left position (pos) with specified cell size.
    """
    x0, y0 = pos
    wf_w = WATERFALL_COLS * cell_size
    wf_h = WATERFALL_ROWS * cell_size

    # Draw each cell
    for c_idx, col in enumerate(columns):
        px_x = x0 + c_idx * cell_size
        if px_x >= x0 + wf_w:
            break
        for r_idx, color in enumerate(col):
            px_y = y0 + r_idx * cell_size
            pygame.draw.rect(
                surface,
                color,
                pygame.Rect(px_x, px_y, cell_size, cell_size)
            )

    if draw_border:
        pygame.draw.rect(surface, border_rgb, pygame.Rect(x0, y0, wf_w, wf_h * 2), width=5)

# === HUD HELPERS ===
def draw_text(surface, text, pos, color=(220, 220, 220), font=FONT_SMALL):
    img = font.render(text, True, color)
    surface.blit(img, pos)

def draw_hud_top(surface, fps, mouse_xy, avg_dark, cols_len):
    # Bar background (slight contrast)
    pygame.draw.rect(surface, (20, 20, 20), HUD_TOP_RECT)

    x_cursor = 12
    y_baseline = 14
    line_gap = 20

    # Left cluster
    draw_text(surface, f"FPS: {fps:5.1f}", (x_cursor, y_baseline))
    draw_text(surface, f"Mouse: {mouse_xy[0]:4d}, {mouse_xy[1]:4d}", (x_cursor, y_baseline + line_gap))
    # (small status line)

    # Center cluster
    cx = 260
    draw_text(surface, f"Rows: {WATERFALL_ROWS}  Cols: {WATERFALL_COLS}", (cx, y_baseline))
    draw_text(surface, f"Avg darkness: {avg_dark:.3f}", (cx, y_baseline + line_gap))

    # Right cluster
    rx = WINDOW_WIDTH - 260
    draw_text(surface, f"SR: {SAMPLE_RATE}  Bands: {N_BANDS}", (rx, y_baseline))
    draw_text(surface, f"α: {GAIN_ALPHA:.3f}  Master: {MASTER_GAIN:.2f}", (rx, y_baseline + line_gap))

def draw_hud_right(surface, audio_lvl):
    # Sidebar background
    pygame.draw.rect(surface, (15, 15, 15), HUD_RIGHT_RECT)

    pad = 16
    x0 = HUD_RIGHT_RECT.left + pad
    y0 = HUD_RIGHT_RECT.top + pad

    # Title
    draw_text(surface, "AUDIO", (x0, y0), font=FONT_MED)
    y0 += 30

    # Level meter
    meter_w = HUD_RIGHT_W - 2 * pad
    meter_h = 20
    pygame.draw.rect(surface, (50, 50, 50), pygame.Rect(x0, y0, meter_w, meter_h), border_radius=4)
    scaled = max(0.0, min(1.0, audio_lvl * 3.0))
    fill_w = int(meter_w * scaled)
    pygame.draw.rect(surface, (180, 180, 180), pygame.Rect(x0, y0, fill_w, meter_h), border_radius=4)
    draw_text(surface, f"RMS: {audio_lvl:.3f}", (x0, y0 + meter_h + 6))
    y0 += meter_h + 28

    # Settings quick view
    draw_text(surface, "SETTINGS", (x0, y0), font=FONT_MED)
    y0 += 24
    draw_text(surface, f"Bands:     {N_BANDS}", (x0, y0)); y0 += 18
    draw_text(surface, f"Cols max:  {WATERFALL_COLS}", (x0, y0)); y0 += 18
    draw_text(surface, f"Cell size: {WATERFALL_CELL_SIZE}", (x0, y0)); y0 += 18
    draw_text(surface, f"Offset:    {VERTICAL_OFFSET}", (x0, y0)); y0 += 18

    # Waterfall rect readout
    y0 += 10
    draw_text(surface, "WATERFALL DISPLAY", (x0, y0), font=FONT_MED); y0 += 24
    draw_text(surface, f"Pos: ({WATERFALL_RECT.x}, {WATERFALL_RECT.y})", (x0, y0)); y0 += 18
    draw_text(surface, f"Size: {WATERFALL_RECT.w}×{WATERFALL_RECT.h}", (x0, y0)); y0 += 18
    draw_text(surface, f"Thresh:   {LOWER_THRESH:.2f}–{UPPER_THRESH:.2f}", (x0, y0)); y0 += 18
    draw_text(surface, f"Movement Only:   {ONLY_SHOW_MOVEMENT}", (x0, y0)); y0 += 18
    draw_text(surface, f"Track mode: {'Frozen' if not FOLLOW_MOUSE else 'Follow'}", (x0, y0)); y0 += 18


def capture_mouse_region(x, y):
    """
    Capture a 300x200 region around (x, y): 150px left/right, 150px above, 50px below.
    Returns an RGB numpy array of shape (200, 300, 3), clamped to screen edges.
    """
    screen_w, screen_h = pyautogui.size()

    left = int(x - MOUSE_VIEW_W // 2)
    top  = int(y - MOUSE_VIEW_ABOVE)

    # Clamp to screen so region is always MOUSE_VIEW_W x MOUSE_VIEW_H
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + MOUSE_VIEW_W > screen_w:
        left = max(0, screen_w - MOUSE_VIEW_W)
    if top + MOUSE_VIEW_H > screen_h:
        top  = max(0, screen_h - MOUSE_VIEW_H)

    monitor = {"left": left, "top": top, "width": MOUSE_VIEW_W, "height": MOUSE_VIEW_H}
    img = sct.grab(monitor)
    arr_bgra = np.array(img)  # H x W x 4 (BGRA)
    arr_rgb  = arr_bgra[:, :, :3][:, :, ::-1]  # BGR -> RGB
    return arr_rgb

def draw_mouse_region_view(surface, x, y):
    """
    Blit the captured region and overlay: center vertical line (mouse x),
    and a rectangle showing the 100px sampling band (ending at y - VERTICAL_OFFSET).
    """
    arr_rgb = capture_mouse_region(x, y)
    arr_rgb = np.ascontiguousarray(arr_rgb, dtype=np.uint8)  # ensure C-contiguous
    h, w = arr_rgb.shape[:2]
    surf = pygame.image.frombuffer(arr_rgb.tobytes(), (w, h), "RGB")

    # Blit preview
    surface.blit(surf, (MOUSE_VIEW_X, MOUSE_VIEW_Y))

    # Overlays
    cx = MOUSE_VIEW_X + MOUSE_VIEW_W * 2 // 2  # small scale fudge retained
    pygame.draw.line(surface, (255, 255, 0),
                     (cx, MOUSE_VIEW_Y + (MOUSE_VIEW_BELOW * 2)),
                     (cx, MOUSE_VIEW_Y + ((MOUSE_VIEW_H - MOUSE_VIEW_BELOW) * 2) - 1), 2)

    topOfLine = MOUSE_VIEW_Y + (MOUSE_VIEW_BELOW * 2)
    bottomOfLine = MOUSE_VIEW_Y + ((MOUSE_VIEW_H - MOUSE_VIEW_BELOW) * 2) - 1
    lineLength = bottomOfLine - topOfLine
    pygame.draw.line(surface, (255, 255, 0), (cx-10, topOfLine), (cx+10, topOfLine), 2)
    pygame.draw.line(surface, (255, 255, 0), (cx-6, topOfLine + (lineLength * 0.25)), (cx+6, topOfLine + (lineLength*0.25)), 2)
    pygame.draw.line(surface, (255, 255, 0), (cx-6, topOfLine + (lineLength * 0.5)), (cx+6, topOfLine + (lineLength*0.5)), 2)
    pygame.draw.line(surface, (255, 255, 0), (cx-6, topOfLine + (lineLength * 0.75)), (cx+6, topOfLine + (lineLength*0.75)), 2)
    pygame.draw.line(surface, (255, 255, 0), (cx-10, topOfLine + (lineLength * 1)), (cx+10, topOfLine + (lineLength*1)), 2)

    # Border & label
    pygame.draw.rect(surface, MOUSE_VIEW_BORDER_RGB,
                     pygame.Rect(MOUSE_VIEW_X, MOUSE_VIEW_Y, MOUSE_VIEW_W*WATERFALL_CELL_SIZE, MOUSE_VIEW_H*WATERFALL_CELL_SIZE), width=5)
    draw_text(surface, "Capture Height: 100", (MOUSE_VIEW_X, MOUSE_VIEW_Y - 28))

running = True
try:
    while running:
        clock.tick(FPS)

        STEP = 0.05  # nudge size for threshold keys

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --- NEW: press 'f' once to freeze current coordinates ---
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                TRACK_X, TRACK_Y = pyautogui.position()
                FOLLOW_MOUSE = not FOLLOW_MOUSE
                print(f"[Freeze] Tracking frozen at ({TRACK_X}, {TRACK_Y})")

            # Toggle only-show-movement
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                ONLY_SHOW_MOVEMENT = not ONLY_SHOW_MOVEMENT
                print(f"[Movement] ONLY_SHOW_MOVEMENT set to ({ONLY_SHOW_MOVEMENT})")

            # --- NEW: threshold bump keys: 1/2 -> lower, 3/4 -> upper ---
            if event.type == pygame.KEYDOWN:
                # 1: lower LOWER_THRESH
                if event.key in (pygame.K_1, pygame.K_KP1):
                    new_lower = max(0.0, LOWER_THRESH - STEP)
                    if new_lower <= UPPER_THRESH and new_lower != LOWER_THRESH:
                        LOWER_THRESH = new_lower
                        print(f"[Thresh] LOWER={LOWER_THRESH:.2f}  UPPER={UPPER_THRESH:.2f}")

                # 2: raise LOWER_THRESH (but not above UPPER_THRESH)
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    new_lower = min(1.0, LOWER_THRESH + STEP)
                    if new_lower <= UPPER_THRESH and new_lower != LOWER_THRESH:
                        LOWER_THRESH = new_lower
                        print(f"[Thresh] LOWER={LOWER_THRESH:.2f}  UPPER={UPPER_THRESH:.2f}")

                # 3: lower UPPER_THRESH (but not below LOWER_THRESH)
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    new_upper = max(0.0, UPPER_THRESH - STEP)
                    if new_upper >= LOWER_THRESH and new_upper != UPPER_THRESH:
                        UPPER_THRESH = new_upper
                        print(f"[Thresh] LOWER={LOWER_THRESH:.2f}  UPPER={UPPER_THRESH:.2f}")

                # 4: raise UPPER_THRESH
                elif event.key in (pygame.K_4, pygame.K_KP4):
                    new_upper = min(1.0, UPPER_THRESH + STEP)
                    if new_upper >= LOWER_THRESH and new_upper != UPPER_THRESH:
                        UPPER_THRESH = new_upper
                        print(f"[Thresh] LOWER={LOWER_THRESH:.2f}  UPPER={UPPER_THRESH:.2f}")


        # Decide which coordinates to use
        if FOLLOW_MOUSE:
            x, y = pyautogui.position()
            TRACK_X, TRACK_Y = x, y   # keep last known for HUD
        else:
            x, y = TRACK_X, TRACK_Y

        # Capture current vertical strip -> brightness vector
        brightness = get_vertical_line(x, y, VERTICAL_SAMPLING_HEIGHT, VERTICAL_OFFSET)
        scaled = apply_threshold_scale(brightness, LOWER_THRESH, UPPER_THRESH, INPUT_RANGE_MAX)
        touse = scaled

        # Update shared audio data
        with brightness_lock:
            current_brightness = touse.astype(np.float32, copy=False)

        # Update visual history
        update_waterfall_history(wf_columns, touse, WATERFALL_COLS)

        # Draw the big window
        screen.fill(WINDOW_BG)

        # HUD: top bar
        fps_now = clock.get_fps()
        avg_dark = float(np.mean(touse)) if touse.size else 0.0
        draw_hud_top(screen, fps_now, (x, y), avg_dark, len(wf_columns))

        # HUD: right sidebar (audio)
        with audio_level_lock:
            lvl = audio_level
        draw_hud_right(screen, lvl)

        # Draw the waterfall at its configured position
        draw_waterfall(
            screen,
            wf_columns,
            pos=(WATERFALL_X, WATERFALL_Y),
            cell_size=WATERFALL_CELL_SIZE,
            draw_border=WATERFALL_BORDER,
            border_rgb=WATERFALL_BORDER_RGB
        )

        # Mouse-region preview (uses the same tracked coords)
        draw_mouse_region_view(screen, x, y)

        pygame.display.flip()
finally:
    # === CLEANUP ===
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    pygame.quit()
