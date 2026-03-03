"""
The Shimmering Field — Live Visualizer
========================================
Runs the Phase 2 Step 4 simulation and renders it in real-time using Pygame.

Usage:
    pip install pygame numpy scipy
    python visualizer.py

Place this file in the same directory as phase2_step4.py.

Controls:
    SPACE      — Pause / Resume
    UP / DOWN  — Speed up / slow down (steps per frame)
    1          — Toggle toxicity overlay
    2          — Toggle organism dots
    3          — Toggle viral particle overlay
    4          — Toggle strata overlay (recent=cyan, intermediate=yellow, ancient=magenta)
    5          — Toggle zone map underlay
    V          — Cycle organism coloring: energy → generation → viral state → age
    R          — Reset simulation
    Q / ESC    — Quit
"""

import numpy as np
import pygame
import sys

# Import the simulation
from phase2_step4 import Config, World


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_SCALE = 5          # Each grid cell = 5x5 pixels (128 * 5 = 640px)
STATS_WIDTH = 320         # Width of the stats panel on the right
FPS = 60                  # Max frames per second
INITIAL_STEPS_PER_FRAME = 1  # Simulation steps per rendered frame

# Color palettes
BG_COLOR = (8, 8, 12)    # Near-black background


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def make_toxic_colormap():
    """
    Generate a 256-entry colormap for toxicity.
    Black (safe) → dark teal → green → yellow → orange → red (lethal).
    """
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.2:
            # Black to dark teal
            s = t / 0.2
            cmap[i] = (0, int(30 * s), int(25 * s))
        elif t < 0.4:
            # Dark teal to green
            s = (t - 0.2) / 0.2
            cmap[i] = (0, int(30 + 90 * s), int(25 - 25 * s + 20 * s))
        elif t < 0.6:
            # Green to yellow
            s = (t - 0.4) / 0.2
            cmap[i] = (int(200 * s), int(120 + 100 * s), 0)
        elif t < 0.8:
            # Yellow to orange
            s = (t - 0.6) / 0.2
            cmap[i] = (200 + int(55 * s), int(220 - 100 * s), 0)
        else:
            # Orange to red
            s = (t - 0.8) / 0.2
            cmap[i] = (255, int(120 - 120 * s), int(40 * s))
    return cmap


def make_viral_colormap():
    """Purple colormap for viral particles: transparent → dim purple → bright magenta."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        cmap[i] = (int(180 * t), int(20 * t), int(220 * t))
    return cmap


TOXIC_CMAP = make_toxic_colormap()
VIRAL_CMAP = make_viral_colormap()


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_heatmap(grid, cmap, vmin=0.0, vmax=1.0):
    """
    Convert a 2D numpy grid into a (H, W, 3) RGB array using the given colormap.
    Values are normalized to [0, 1] based on vmin/vmax, then mapped to 0-255 index.
    """
    normalized = np.clip((grid - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    indices = (normalized * 255).astype(np.uint8)
    return cmap[indices]  # Fancy indexing: (H, W) of uint8 → (H, W, 3)


def render_grid_to_surface(rgb_array, scale):
    """
    Take an (H, W, 3) RGB array and scale it up to a Pygame surface.
    Uses nearest-neighbor scaling (each cell becomes a scale×scale block).
    """
    h, w = rgb_array.shape[:2]
    # Create a small surface from the array
    # pygame.surfarray expects (W, H, 3) — transposed from numpy's (H, W, 3)
    small_surf = pygame.surfarray.make_surface(rgb_array.transpose(1, 0, 2))
    # Scale up with nearest-neighbor
    return pygame.transform.scale(small_surf, (w * scale, h * scale))


def draw_organisms(surface, world, scale, color_mode="energy"):
    """
    Draw each organism as a small dot on the surface.
    Color depends on color_mode:
      - "energy":     dark red (dying) → green (healthy) → bright white (full)
      - "generation": blue (old lineage) → green → yellow (new lineage)
      - "viral":      green (healthy), orange (lysogenic), red (lytic)
      - "age":        white (young) → dim (old)
    """
    if world.pop == 0:
        return

    rows = world.rows
    cols = world.cols
    n = world.pop

    # Pre-compute colors for all organisms
    colors = np.zeros((n, 3), dtype=np.uint8)

    if color_mode == "energy":
        # Normalize energy to [0, 1]
        e_norm = np.clip(world.energy / world.cfg.energy_max, 0, 1)
        colors[:, 0] = (255 * (1.0 - e_norm)).astype(np.uint8)      # Red when low
        colors[:, 1] = (255 * e_norm).astype(np.uint8)               # Green when high
        colors[:, 2] = (180 * np.maximum(0, e_norm - 0.7) / 0.3).astype(np.uint8)  # Blue glow when full

    elif color_mode == "generation":
        max_gen = max(world.generation.max(), 1)
        g_norm = np.clip(world.generation / max_gen, 0, 1)
        colors[:, 0] = (255 * g_norm).astype(np.uint8)               # Red increases
        colors[:, 1] = (180 + 75 * (1.0 - np.abs(g_norm - 0.5) * 2)).astype(np.uint8)
        colors[:, 2] = (255 * (1.0 - g_norm)).astype(np.uint8)       # Blue for old generations

    elif color_mode == "viral":
        # Green = healthy, orange = lysogenic carrier, red = lytic infected
        healthy = (world.viral_load == 0) & (world.lysogenic_strength < 0.01)
        lysogenic = (world.viral_load == 0) & (world.lysogenic_strength >= 0.01)
        lytic = world.viral_load > 0

        colors[healthy, :] = [80, 230, 100]    # Green
        colors[lysogenic, :] = [230, 180, 40]  # Orange/gold
        # Lytic: intensity based on viral_load
        if lytic.any():
            vl = np.clip(world.viral_load[lytic] / world.cfg.viral_burst_threshold, 0, 1)
            colors[lytic, 0] = (150 + 105 * vl).astype(np.uint8)
            colors[lytic, 1] = (40 * (1.0 - vl)).astype(np.uint8)
            colors[lytic, 2] = (40 * (1.0 - vl)).astype(np.uint8)

    elif color_mode == "age":
        a_norm = np.clip(world.age / world.cfg.max_age, 0, 1)
        brightness = (255 * (1.0 - a_norm * 0.7)).astype(np.uint8)
        colors[:, 0] = brightness
        colors[:, 1] = brightness
        colors[:, 2] = (200 * (1.0 - a_norm)).astype(np.uint8)

    # Draw each organism as a small rectangle
    # For large populations, batch drawing with pixel access would be faster,
    # but rect drawing is simpler and fine for ~1000 organisms
    dot_size = max(2, scale - 1)
    for i in range(n):
        x = int(cols[i]) * scale
        y = int(rows[i]) * scale
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        pygame.draw.rect(surface, color, (x, y, dot_size, dot_size))


def draw_stats_panel(surface, world, x_offset, color_mode, steps_per_frame,
                     paused, layers, elapsed_sim_time):
    """
    Draw a text-based stats panel on the right side of the screen.
    Shows population, energy, toxicity, viral stats, controls, etc.
    """
    font = pygame.font.SysFont("monospace", 14)
    small_font = pygame.font.SysFont("monospace", 12)

    # Background for stats panel
    panel_rect = pygame.Rect(x_offset, 0, STATS_WIDTH, surface.get_height())
    pygame.draw.rect(surface, (15, 15, 22), panel_rect)
    pygame.draw.line(surface, (60, 60, 80), (x_offset, 0), (x_offset, surface.get_height()), 2)

    lines = []
    s = world.stats_history[-1] if world.stats_history else {}

    # Title
    lines.append(("THE SHIMMERING FIELD", (200, 180, 255)))
    lines.append((f"Phase 2 Step 4: Viral Archaeology", (140, 130, 170)))
    lines.append(("", None))

    # Simulation state
    state = "▐▐ PAUSED" if paused else f"▶ {steps_per_frame} steps/frame"
    lines.append((f"t = {world.timestep:,}   {state}", (255, 255, 255)))
    lines.append((f"Sim time: {elapsed_sim_time:.1f}s", (150, 150, 150)))
    lines.append(("", None))

    # Population
    lines.append(("─── Population ───", (100, 180, 255)))
    lines.append((f"  Alive:  {world.pop:,}", (255, 255, 255)))
    if s:
        lines.append((f"  Energy: {s.get('avg_energy', 0):5.1f} avg", (180, 230, 180)))
        lines.append((f"  Gen:    {s.get('max_gen', 0)}", (180, 180, 230)))
    lines.append(("", None))

    # Toxicity
    lines.append(("─── Toxicity ───", (255, 180, 80)))
    if s:
        lines.append((f"  Mean: {s.get('toxic_mean', 0):.3f}  Max: {s.get('toxic_max', 0):.2f}", (255, 200, 120)))
        lines.append((f"  Safe:   {s.get('in_low_toxic', 0):4d}  organisms", (100, 255, 100)))
        lines.append((f"  Medium: {s.get('in_med_toxic', 0):4d}  organisms", (255, 255, 100)))
        lines.append((f"  High:   {s.get('in_high_toxic', 0):4d}  organisms", (255, 100, 100)))
    lines.append(("", None))

    # Viral
    lines.append(("─── Viral System ───", (220, 100, 255)))
    if s:
        lines.append((f"  Lytic:     {s.get('n_lytic', 0):4d} infected", (255, 120, 120)))
        lines.append((f"  Lysogenic: {s.get('n_lysogenic', 0):4d} carriers", (230, 180, 60)))
        lines.append((f"  Bursts:    {s.get('total_lytic_deaths', 0):,} total", (200, 100, 100)))
        lines.append((f"  Integr:    {s.get('total_lyso_integrations', 0):,}", (180, 150, 80)))
        lines.append((f"  Activ:     {s.get('total_lyso_activations', 0):,}", (200, 130, 80)))
        lines.append((f"  Particles: {s.get('viral_mean', 0):.2f} avg", (180, 100, 220)))
    lines.append(("", None))

    # Strata
    lines.append(("─── Strata (DNA Layers) ───", (100, 220, 220)))
    if s:
        lines.append((f"  Recent:  {s.get('strata_recent', 0):.2f}  xfer: {s.get('xfer_recent', 0):,}", (120, 230, 230)))
        lines.append((f"  Interm:  {s.get('strata_intermediate', 0):.2f}  xfer: {s.get('xfer_intermediate', 0):,}", (230, 230, 100)))
        lines.append((f"  Ancient: {s.get('strata_ancient', 0):.2f}  xfer: {s.get('xfer_ancient', 0):,}", (230, 100, 230)))
    lines.append(("", None))

    # Layer toggles
    lines.append(("─── Visible Layers ───", (150, 150, 150)))
    for key, name in [("toxic", "1:Toxic"), ("organisms", "2:Organisms"),
                      ("viral", "3:Viral"), ("strata", "4:Strata"), ("zones", "5:Zones")]:
        status = "●" if layers[key] else "○"
        color = (180, 255, 180) if layers[key] else (80, 80, 80)
        lines.append((f"  {status} {name}", color))

    lines.append(("", None))
    cm_names = {"energy": "Energy", "generation": "Generation", "viral": "Viral State", "age": "Age"}
    lines.append((f"  Organism color: {cm_names[color_mode]}", (200, 200, 255)))
    lines.append(("", None))

    # Controls
    lines.append(("─── Controls ───", (120, 120, 120)))
    controls = [
        "SPACE  Pause/Resume",
        "UP/DN  Speed +/-",
        "1-5    Toggle layers",
        "V      Cycle colors",
        "R      Reset",
        "Q/ESC  Quit",
    ]
    for ctrl in controls:
        lines.append((f"  {ctrl}", (100, 100, 110)))

    # Render text lines
    y = 12
    for text, color in lines:
        if color is None:
            y += 6
            continue
        surf = small_font.render(text, True, color)
        surface.blit(surf, (x_offset + 12, y))
        y += 17


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    pygame.display.set_caption("The Shimmering Field — Live Visualizer")

    cfg = Config()
    N = cfg.grid_size  # 128

    grid_px = N * WINDOW_SCALE             # 640
    win_w = grid_px + STATS_WIDTH          # 640 + 320 = 960
    win_h = grid_px                        # 640

    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    # Initialize simulation
    world = World(cfg)

    # State
    paused = False
    steps_per_frame = INITIAL_STEPS_PER_FRAME
    color_mode = "energy"
    color_modes = ["energy", "generation", "viral", "age"]
    color_idx = 0
    layers = {
        "toxic": True,
        "organisms": True,
        "viral": True,
        "strata": False,
        "zones": False,
    }

    import time as _time
    sim_start = _time.time()

    running = True
    while running:
        # ── Handle events ────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    steps_per_frame = min(steps_per_frame + 1, 50)
                elif event.key == pygame.K_DOWN:
                    steps_per_frame = max(steps_per_frame - 1, 1)
                elif event.key == pygame.K_v:
                    color_idx = (color_idx + 1) % len(color_modes)
                    color_mode = color_modes[color_idx]
                elif event.key == pygame.K_r:
                    world = World(cfg)
                    sim_start = _time.time()
                elif event.key == pygame.K_1:
                    layers["toxic"] = not layers["toxic"]
                elif event.key == pygame.K_2:
                    layers["organisms"] = not layers["organisms"]
                elif event.key == pygame.K_3:
                    layers["viral"] = not layers["viral"]
                elif event.key == pygame.K_4:
                    layers["strata"] = not layers["strata"]
                elif event.key == pygame.K_5:
                    layers["zones"] = not layers["zones"]

        # ── Simulation step(s) ───────────────────────────────────────────
        if not paused:
            for _ in range(steps_per_frame):
                world.update()
                if world.pop == 0:
                    paused = True  # Auto-pause on extinction
                    break

        elapsed = _time.time() - sim_start

        # ── Render ───────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Start with a base layer
        base_rgb = np.full((N, N, 3), BG_COLOR, dtype=np.uint8)

        # Layer: Zone map (subtle underlay showing safe/danger zones)
        if layers["zones"]:
            zone_norm = np.clip((world.zone_map - 0.3) / (2.0 - 0.3), 0, 1)
            zone_rgb = np.zeros((N, N, 3), dtype=np.uint8)
            zone_rgb[:, :, 0] = (30 * zone_norm).astype(np.uint8)    # Slight red for amplifying
            zone_rgb[:, :, 1] = (25 * (1.0 - zone_norm)).astype(np.uint8)  # Slight green for absorbing
            zone_rgb[:, :, 2] = 15
            base_rgb = np.maximum(base_rgb, zone_rgb)

        # Layer: Toxicity heatmap
        if layers["toxic"]:
            toxic_rgb = render_heatmap(world.toxic, TOXIC_CMAP, vmin=0, vmax=2.0)
            # Blend: only show where toxicity is meaningful
            mask = world.toxic > 0.02
            base_rgb[mask] = np.clip(
                base_rgb[mask].astype(np.int16) + toxic_rgb[mask].astype(np.int16),
                0, 255).astype(np.uint8)

        # Layer: Strata overlay (shows where buried DNA is concentrated)
        if layers["strata"]:
            strata_rgb = np.zeros((N, N, 3), dtype=np.float64)
            # Recent = cyan, Intermediate = yellow, Ancient = magenta
            recent_w = np.clip(world.strata_weight["recent"] / 3.0, 0, 1)
            inter_w = np.clip(world.strata_weight["intermediate"] / 2.0, 0, 1)
            ancient_w = np.clip(world.strata_weight["ancient"] / 1.5, 0, 1)
            strata_rgb[:, :, 0] += ancient_w * 120 + inter_w * 100
            strata_rgb[:, :, 1] += recent_w * 100 + inter_w * 100
            strata_rgb[:, :, 2] += recent_w * 120 + ancient_w * 120
            base_rgb = np.clip(
                base_rgb.astype(np.int16) + strata_rgb.astype(np.int16),
                0, 255).astype(np.uint8)

        # Layer: Viral particles (purple glow)
        if layers["viral"]:
            viral_rgb = render_heatmap(world.viral_particles, VIRAL_CMAP, vmin=0, vmax=5.0)
            mask_v = world.viral_particles > 0.05
            base_rgb[mask_v] = np.clip(
                base_rgb[mask_v].astype(np.int16) + viral_rgb[mask_v].astype(np.int16),
                0, 255).astype(np.uint8)

        # Blit the composited grid to the screen
        grid_surface = render_grid_to_surface(base_rgb, WINDOW_SCALE)
        screen.blit(grid_surface, (0, 0))

        # Layer: Organisms (drawn as dots on top of everything)
        if layers["organisms"]:
            draw_organisms(screen, world, WINDOW_SCALE, color_mode)

        # Stats panel
        draw_stats_panel(screen, world, grid_px, color_mode, steps_per_frame,
                         paused, layers, elapsed)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
