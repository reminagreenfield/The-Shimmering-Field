"""
The Shimmering Field — Phase 3 Step 4 Live Visualizer
======================================================
Runs the simulation and renders it in real-time using Pygame.

Usage:
    pip install pygame numpy scipy
    python visualizer_p3s4.py

Place this file in the same directory as phase3_step4.py.

Controls:
    SPACE      — Pause / Resume
    UP / DOWN  — Speed up / slow down (steps per frame)
    1          — Toggle toxicity overlay
    2          — Toggle organism dots
    3          — Toggle decomposition overlay
    4          — Toggle viral particle overlay
    5          — Toggle zone map underlay
    V          — Cycle organism coloring:
                   Role → Energy → Modules → Generation → Defense → Age
    R          — Reset simulation
    Q / ESC    — Quit
"""

import numpy as np
import pygame
import sys
import time as _time

from phase3_step4 import (
    Config, World,
    M_PHOTO, M_CHEMO, M_CONSUME, M_MOVE, M_FORAGE, M_DEFENSE, M_DETOX, M_TOXPROD,
    N_MODULES, MODULE_NAMES
)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_SCALE = 5          # Each grid cell = 5×5 pixels → 640px grid
STATS_WIDTH = 340         # Stats panel width
FPS = 60
INITIAL_STEPS_PER_FRAME = 1

BG_COLOR = (8, 8, 12)


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def make_colormap(keypoints):
    """Build a 256-entry RGB colormap from a list of (position, r, g, b) keypoints."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Find bracketing keypoints
        lo = keypoints[0]
        hi = keypoints[-1]
        for k in range(len(keypoints) - 1):
            if keypoints[k][0] <= t <= keypoints[k + 1][0]:
                lo = keypoints[k]
                hi = keypoints[k + 1]
                break
        span = hi[0] - lo[0]
        s = (t - lo[0]) / span if span > 0 else 0
        cmap[i] = [int(lo[j+1] + s * (hi[j+1] - lo[j+1])) for j in range(3)]
    return cmap

TOXIC_CMAP = make_colormap([
    (0.0,  0,   0,   0),
    (0.15, 0,  30,  25),
    (0.3,  0, 100,  20),
    (0.5, 180, 200,  0),
    (0.7, 240, 140,  0),
    (1.0, 255,  40,  30),
])

DECOMP_CMAP = make_colormap([
    (0.0,   0,   0,   0),
    (0.2,  30,  15,   5),
    (0.4,  80,  45,  10),
    (0.6, 140,  90,  20),
    (0.8, 200, 150,  40),
    (1.0, 255, 220, 100),
])

VIRAL_CMAP = make_colormap([
    (0.0,   0,   0,   0),
    (0.3,  60,   8,  80),
    (0.6, 140,  20, 180),
    (1.0, 220,  60, 255),
])


def render_heatmap(grid, cmap, vmin=0.0, vmax=1.0):
    normalized = np.clip((grid - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    indices = (normalized * 255).astype(np.uint8)
    return cmap[indices]


def render_grid_to_surface(rgb_array, scale):
    h, w = rgb_array.shape[:2]
    small_surf = pygame.surfarray.make_surface(rgb_array.transpose(1, 0, 2))
    return pygame.transform.scale(small_surf, (w * scale, h * scale))


# ═══════════════════════════════════════════════════════════════════════════════
# ORGANISM RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

# Role colors
C_PRODUCER    = np.array([60, 210, 90],  dtype=np.uint8)   # Green
C_CARNIVORE   = np.array([230, 55, 55],  dtype=np.uint8)   # Red
C_DETRITIVORE = np.array([180, 130, 50], dtype=np.uint8)   # Brown/amber
C_OMNIVORE    = np.array([100, 140, 230], dtype=np.uint8)  # Blue


def classify_organisms(world):
    """Return role index per organism: 0=producer, 1=carnivore, 2=detritivore, 3=omnivore."""
    n = world.pop
    roles = np.zeros(n, dtype=np.int32)  # default: producer
    has_prod = world.module_present[:, M_PHOTO] | world.module_present[:, M_CHEMO]
    has_cons = world.module_present[:, M_CONSUME]

    omnivore = has_prod & has_cons
    obligate = has_cons & ~has_prod

    roles[omnivore] = 3

    if obligate.any():
        consume_off = 8  # MODULE_WEIGHT_OFFSETS[M_CONSUME]
        dp = 1.0 / (1.0 + np.exp(-world.weights[obligate, consume_off + 2]))
        ob_idx = np.where(obligate)[0]
        roles[ob_idx[dp < 0.5]] = 1   # carnivore
        roles[ob_idx[dp >= 0.5]] = 2  # detritivore

    return roles


def draw_organisms(surface, world, scale, color_mode="role"):
    if world.pop == 0:
        return

    n = world.pop
    rows, cols = world.rows, world.cols
    colors = np.zeros((n, 3), dtype=np.uint8)

    if color_mode == "role":
        roles = classify_organisms(world)
        role_colors = np.array([C_PRODUCER, C_CARNIVORE, C_DETRITIVORE, C_OMNIVORE])
        colors = role_colors[roles]

    elif color_mode == "energy":
        e_norm = np.clip(world.energy / world.cfg.energy_max, 0, 1)
        colors[:, 0] = (220 * (1.0 - e_norm)).astype(np.uint8)
        colors[:, 1] = (230 * e_norm).astype(np.uint8)
        colors[:, 2] = (160 * np.maximum(0, e_norm - 0.6) / 0.4).astype(np.uint8)

    elif color_mode == "modules":
        # Color by module count: fewer=dim, more=bright rainbow
        mc = world.module_present.sum(axis=1).astype(np.float64)
        mc_norm = np.clip((mc - 2) / 5.0, 0, 1)  # 2 modules=0, 7+=1
        colors[:, 0] = (100 + 155 * mc_norm).astype(np.uint8)
        colors[:, 1] = (200 * (1.0 - np.abs(mc_norm - 0.5) * 2)).astype(np.uint8)
        colors[:, 2] = (255 * (1.0 - mc_norm)).astype(np.uint8)

    elif color_mode == "generation":
        max_gen = max(world.generation.max(), 1)
        g_norm = np.clip(world.generation / max_gen, 0, 1)
        colors[:, 0] = (255 * g_norm).astype(np.uint8)
        colors[:, 1] = (180 + 75 * (1.0 - np.abs(g_norm - 0.5) * 2)).astype(np.uint8)
        colors[:, 2] = (255 * (1.0 - g_norm)).astype(np.uint8)

    elif color_mode == "defense":
        # Green=undefended, gold=defended, bright=strong defense
        has_def = world.module_active[:, M_DEFENSE]
        colors[~has_def] = [60, 180, 80]
        if has_def.any():
            dw_off = 24  # MODULE_WEIGHT_OFFSETS[M_DEFENSE]
            shell = 1.0 / (1.0 + np.exp(-world.weights[has_def, dw_off]))
            colors[has_def, 0] = (200 + 55 * shell).astype(np.uint8)
            colors[has_def, 1] = (180 * shell).astype(np.uint8)
            colors[has_def, 2] = 30

    elif color_mode == "age":
        a_norm = np.clip(world.age / world.cfg.max_age, 0, 1)
        brightness = (255 * (1.0 - a_norm * 0.7)).astype(np.uint8)
        colors[:, 0] = brightness
        colors[:, 1] = brightness
        colors[:, 2] = (180 * (1.0 - a_norm)).astype(np.uint8)

    dot_size = max(2, scale - 1)
    for i in range(n):
        x = int(cols[i]) * scale
        y = int(rows[i]) * scale
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        pygame.draw.rect(surface, color, (x, y, dot_size, dot_size))


# ═══════════════════════════════════════════════════════════════════════════════
# STATS PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def draw_stats_panel(surface, world, x_offset, color_mode, steps_per_frame,
                     paused, layers, elapsed):
    font = pygame.font.SysFont("monospace", 12)

    panel_rect = pygame.Rect(x_offset, 0, STATS_WIDTH, surface.get_height())
    pygame.draw.rect(surface, (15, 15, 22), panel_rect)
    pygame.draw.line(surface, (60, 60, 80), (x_offset, 0), (x_offset, surface.get_height()), 2)

    lines = []
    s = world.stats_history[-1] if world.stats_history else {}

    lines.append(("THE SHIMMERING FIELD", (200, 180, 255)))
    lines.append(("Phase 3 Step 4: FORAGE+DEFENSE+DETOX", (140, 130, 170)))
    lines.append(("", None))

    state = "▐▐ PAUSED" if paused else f"▶ {steps_per_frame} steps/frame"
    lines.append((f"t = {world.timestep:,}   {state}", (255, 255, 255)))
    lines.append((f"Sim time: {elapsed:.1f}s", (150, 150, 150)))
    lines.append(("", None))

    # ── Population ──
    lines.append(("─── Population ───", (100, 180, 255)))
    lines.append((f"  Alive:  {world.pop:,}", (255, 255, 255)))
    if s:
        lines.append((f"  Energy: {s.get('avg_energy', 0):5.1f} avg", (180, 230, 180)))
        lines.append((f"  Gen:    {s.get('max_gen', 0)}", (180, 180, 230)))
        lines.append((f"  Modules:{s.get('avg_modules', 0):.2f} avg", (200, 200, 200)))
    lines.append(("", None))

    # ── Ecological Roles ──
    lines.append(("─── Ecology ───", (100, 220, 130)))
    if s:
        r = s.get("roles", {})
        lines.append((f"  ● Producers:    {r.get('producer',0):4d}", (60, 210, 90)))
        lines.append((f"  ● Carnivores:   {r.get('carnivore',0):4d}", (230, 55, 55)))
        lines.append((f"  ● Detritivores: {r.get('detritivore',0):4d}", (180, 130, 50)))
        lines.append((f"  ● Omnivores:    {r.get('omnivore',0):4d}", (100, 140, 230)))
        lines.append((f"  Kills: {s.get('kills',0):3d} this step  ({s.get('total_kills',0):,} total)", (255, 120, 120)))
    lines.append(("", None))

    # ── Module Counts ──
    lines.append(("─── Modules ───", (200, 180, 100)))
    if s:
        mc = s.get("module_counts", {})
        mod_colors = {
            "PHOTO": (100, 220, 100), "CHEMO": (80, 200, 200),
            "CONSUME": (230, 80, 80), "MOVE": (180, 180, 220),
            "FORAGE": (220, 200, 80), "DEFENSE": (200, 160, 60),
            "DETOX": (120, 220, 180), "TOXPROD": (140, 100, 100),
        }
        for name in ["PHOTO", "CHEMO", "CONSUME", "MOVE", "FORAGE", "DEFENSE", "DETOX", "TOXPROD"]:
            c_val = mc.get(name, 0)
            col = mod_colors.get(name, (160, 160, 160))
            lines.append((f"  {name:8s} {c_val:5d}", col))
    lines.append(("", None))

    # ── Environment ──
    lines.append(("─── Environment ───", (255, 180, 80)))
    if s:
        lines.append((f"  Toxic:  {s.get('toxic_mean', 0):.3f} avg", (255, 200, 120)))
        lines.append((f"  Decomp: {s.get('decomp_mean', 0):.2f} avg", (200, 160, 80)))
    lines.append(("", None))

    # ── Layers ──
    lines.append(("─── Layers ───", (150, 150, 150)))
    layer_keys = [("toxic", "1:Toxic"), ("organisms", "2:Organisms"),
                  ("decomp", "3:Decomp"), ("viral", "4:Viral"), ("zones", "5:Zones")]
    for key, name in layer_keys:
        on = layers[key]
        lines.append((f"  {'●' if on else '○'} {name}", (180, 255, 180) if on else (80, 80, 80)))
    lines.append(("", None))

    cm_names = {"role": "Ecological Role", "energy": "Energy", "modules": "Module Count",
                "generation": "Generation", "defense": "Defense", "age": "Age"}
    lines.append((f"  Color: {cm_names.get(color_mode, color_mode)}", (200, 200, 255)))
    lines.append(("", None))

    # ── Role legend (when in role mode) ──
    if color_mode == "role":
        lines.append(("─── Legend ───", (120, 120, 120)))
        lines.append(("  ● Producer", (60, 210, 90)))
        lines.append(("  ● Carnivore", (230, 55, 55)))
        lines.append(("  ● Detritivore", (180, 130, 50)))
        lines.append(("  ● Omnivore", (100, 140, 230)))
        lines.append(("", None))

    # ── Controls ──
    lines.append(("─── Controls ───", (120, 120, 120)))
    for ctrl in ["SPACE  Pause/Resume", "UP/DN  Speed +/-",
                 "1-5    Toggle layers", "V      Cycle colors",
                 "R      Reset", "Q/ESC  Quit"]:
        lines.append((f"  {ctrl}", (100, 100, 110)))

    # Render
    y = 10
    for text, color in lines:
        if color is None:
            y += 5
            continue
        surf = font.render(text, True, color)
        surface.blit(surf, (x_offset + 10, y))
        y += 16


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    pygame.display.set_caption("The Shimmering Field — Phase 3 Step 4")

    cfg = Config()
    N = cfg.grid_size

    grid_px = N * WINDOW_SCALE
    win_w = grid_px + STATS_WIDTH
    win_h = grid_px

    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    world = World(cfg)

    paused = False
    steps_per_frame = INITIAL_STEPS_PER_FRAME
    color_modes = ["role", "energy", "modules", "generation", "defense", "age"]
    color_idx = 0
    color_mode = color_modes[0]
    layers = {
        "toxic": True,
        "organisms": True,
        "decomp": True,
        "viral": False,
        "zones": False,
    }

    sim_start = _time.time()
    running = True

    while running:
        # ── Events ───────────────────────────────────────────────────────
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
                    layers["decomp"] = not layers["decomp"]
                elif event.key == pygame.K_4:
                    layers["viral"] = not layers["viral"]
                elif event.key == pygame.K_5:
                    layers["zones"] = not layers["zones"]

        # ── Simulation ───────────────────────────────────────────────────
        if not paused:
            for _ in range(steps_per_frame):
                world.update()
                if world.pop == 0:
                    paused = True
                    break

        elapsed = _time.time() - sim_start

        # ── Render ───────────────────────────────────────────────────────
        screen.fill(BG_COLOR)
        base_rgb = np.full((N, N, 3), BG_COLOR, dtype=np.uint8)

        # Zone map underlay
        if layers["zones"]:
            zone_norm = np.clip((world.zone_map - 0.3) / (2.0 - 0.3), 0, 1)
            zone_rgb = np.zeros((N, N, 3), dtype=np.uint8)
            zone_rgb[:, :, 0] = (30 * zone_norm).astype(np.uint8)
            zone_rgb[:, :, 1] = (25 * (1.0 - zone_norm)).astype(np.uint8)
            zone_rgb[:, :, 2] = 15
            base_rgb = np.maximum(base_rgb, zone_rgb)

        # Toxicity heatmap
        if layers["toxic"]:
            toxic_rgb = render_heatmap(world.toxic, TOXIC_CMAP, vmin=0, vmax=2.0)
            mask = world.toxic > 0.02
            base_rgb[mask] = np.clip(
                base_rgb[mask].astype(np.int16) + toxic_rgb[mask].astype(np.int16),
                0, 255).astype(np.uint8)

        # Decomposition heatmap (amber/brown — looks like organic matter)
        if layers["decomp"]:
            decomp_rgb = render_heatmap(world.decomposition, DECOMP_CMAP, vmin=0, vmax=8.0)
            mask_d = world.decomposition > 0.05
            base_rgb[mask_d] = np.clip(
                base_rgb[mask_d].astype(np.int16) + decomp_rgb[mask_d].astype(np.int16),
                0, 255).astype(np.uint8)

        # Viral particles (purple glow)
        if layers["viral"]:
            viral_rgb = render_heatmap(world.viral_particles, VIRAL_CMAP, vmin=0, vmax=5.0)
            mask_v = world.viral_particles > 0.05
            base_rgb[mask_v] = np.clip(
                base_rgb[mask_v].astype(np.int16) + viral_rgb[mask_v].astype(np.int16),
                0, 255).astype(np.uint8)

        # Blit composited grid
        grid_surface = render_grid_to_surface(base_rgb, WINDOW_SCALE)
        screen.blit(grid_surface, (0, 0))

        # Organisms
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
