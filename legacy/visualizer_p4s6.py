"""
The Shimmering Field — Phase 4 Step 6 Live Visualizer
======================================================
Runs the simulation and renders it in real-time using Pygame.
Phase 4 Step 6: Sessile-Mobile Divergence + Nutrient Scarcity.

Usage:
    pip install pygame numpy scipy
    python visualizer_p4s6.py

Place this file in the same directory as phase4_step6.py.

Controls:
    SPACE      — Pause / Resume
    UP / DOWN  — Speed up / slow down (steps per frame)
    1          — Toggle toxicity overlay
    2          — Toggle organism dots
    3          — Toggle decomposition overlay
    4          — Toggle viral particle overlay
    5          — Toggle social field overlay
    6          — Toggle mediator field overlay
    7          — Toggle nutrient overlay
    8          — Toggle fungal network overlay
    9          — Toggle ecosystem integrity overlay
    V          — Cycle organism coloring:
                   Role → Mobility → Energy → Modules → Generation
                   → Defense → Social → Manipulation → Hijack
                   → Endosymbiosis → Shedding → Cascade → Development → Age
    R          — Reset simulation
    Q / ESC    — Quit
"""

import numpy as np
import pygame
import sys
import time as _time

from phase4_step6 import (
    Config, World,
    M_PHOTO, M_CHEMO, M_CONSUME, M_MOVE, M_FORAGE, M_DEFENSE, M_DETOX,
    M_TOXPROD, M_VRESIST, M_SOCIAL, M_MEDIATE,
    N_MODULES, MODULE_NAMES, MODULE_WEIGHT_SIZES
)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_SCALE = 8
STATS_WIDTH = 340
FPS = 60
INITIAL_STEPS_PER_FRAME = 1
MIN_WIN_HEIGHT = 1100

BG_COLOR = (8, 8, 12)


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def make_colormap(keypoints):
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        lo, hi = keypoints[0], keypoints[-1]
        for k in range(len(keypoints) - 1):
            if keypoints[k][0] <= t <= keypoints[k + 1][0]:
                lo, hi = keypoints[k], keypoints[k + 1]
                break
        span = hi[0] - lo[0]
        s = (t - lo[0]) / span if span > 0 else 0
        cmap[i] = [int(lo[j+1] + s * (hi[j+1] - lo[j+1])) for j in range(3)]
    return cmap

TOXIC_CMAP = make_colormap([
    (0.0, 0, 0, 0), (0.15, 0, 30, 25), (0.3, 0, 100, 20),
    (0.5, 180, 200, 0), (0.7, 240, 140, 0), (1.0, 255, 40, 30),
])
DECOMP_CMAP = make_colormap([
    (0.0, 0, 0, 0), (0.2, 30, 15, 5), (0.4, 80, 45, 10),
    (0.6, 140, 90, 20), (0.8, 200, 150, 40), (1.0, 255, 220, 100),
])
VIRAL_CMAP = make_colormap([
    (0.0, 0, 0, 0), (0.3, 60, 8, 80), (0.6, 140, 20, 180), (1.0, 220, 60, 255),
])
SOCIAL_CMAP = make_colormap([
    (0.0, 0, 0, 0), (0.3, 10, 40, 60), (0.6, 40, 120, 180), (1.0, 100, 200, 255),
])
MEDIATOR_CMAP = make_colormap([
    (0.0, 0, 0, 0), (0.3, 50, 30, 10), (0.6, 180, 120, 40), (1.0, 255, 220, 100),
])
NUTRIENT_CMAP = make_colormap([
    (0.0, 0, 0, 0), (0.3, 5, 25, 10), (0.6, 20, 80, 30), (1.0, 60, 180, 70),
])
# NEW: Fungal network — warm brown/amber (mycelial web)
FUNGAL_CMAP = make_colormap([
    (0.0, 0, 0, 0), (0.2, 25, 12, 5), (0.4, 80, 40, 10),
    (0.6, 160, 90, 25), (0.8, 220, 150, 50), (1.0, 255, 200, 80),
])
# NEW: Ecosystem integrity — green (healthy) to red (collapsed)
INTEGRITY_CMAP = make_colormap([
    (0.0, 200, 30, 20), (0.2, 220, 80, 20), (0.4, 200, 160, 30),
    (0.6, 120, 200, 50), (0.8, 50, 180, 60), (1.0, 30, 140, 50),
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
# ORGANISM RENDERING — 13 color modes
# ═══════════════════════════════════════════════════════════════════════════════

C_PRODUCER    = np.array([60, 210, 90],  dtype=np.uint8)
C_HERBIVORE   = np.array([200, 100, 200], dtype=np.uint8)
C_CARNIVORE   = np.array([230, 55, 55],  dtype=np.uint8)
C_DETRITIVORE = np.array([180, 130, 50], dtype=np.uint8)
C_OMNIVORE    = np.array([100, 140, 230], dtype=np.uint8)

_CONSUME_OFF = int(sum(MODULE_WEIGHT_SIZES[:M_CONSUME]))
_DEFENSE_OFF = int(sum(MODULE_WEIGHT_SIZES[:M_DEFENSE]))


def classify_organisms(world):
    """0=producer, 1=herbivore, 2=carnivore, 3=detritivore, 4=omnivore."""
    n = world.pop
    roles = np.zeros(n, dtype=np.int32)
    has_prod = world.module_present[:, M_PHOTO] | world.module_present[:, M_CHEMO]
    has_cons = world.module_present[:, M_CONSUME]
    roles[has_prod & has_cons] = 4
    obligate = has_cons & ~has_prod
    if obligate.any():
        dp = 1.0 / (1.0 + np.exp(-world.weights[obligate, _CONSUME_OFF + 2]))
        ps = 1.0 / (1.0 + np.exp(-world.weights[obligate, _CONSUME_OFF + 0]))
        ob_idx = np.where(obligate)[0]
        roles[ob_idx[dp >= 0.5]] = 3
        roles[ob_idx[(dp < 0.5) & (ps < 0.5)]] = 1
        roles[ob_idx[(dp < 0.5) & (ps >= 0.5)]] = 2
    return roles


def draw_organisms(surface, world, scale, color_mode="role", y_offset=0):
    if world.pop == 0:
        return
    n = world.pop
    rows, cols = world.rows, world.cols
    colors = np.zeros((n, 3), dtype=np.uint8)

    if color_mode == "role":
        role_colors = np.array([C_PRODUCER, C_HERBIVORE, C_CARNIVORE, C_DETRITIVORE, C_OMNIVORE])
        colors = role_colors[classify_organisms(world)]

    elif color_mode == "mobility":
        # Sessile-mobile divergence visualization
        has_photo = world.module_present[:, M_PHOTO] | world.module_present[:, M_CHEMO]
        has_consume = world.module_present[:, M_CONSUME]
        has_move = world.module_active[:, M_MOVE]
        # Sessile producer (plant): deep green
        sessile_prod = has_photo & ~has_consume & ~has_move
        colors[sessile_prod] = [30, 160, 50]
        # Mobile producer (rare mobile plant): bright lime
        mobile_prod = has_photo & ~has_consume & has_move
        colors[mobile_prod] = [160, 255, 60]
        # Mobile consumer (animal): warm red-orange
        mobile_cons = has_consume & ~has_photo & has_move
        colors[mobile_cons] = [230, 90, 40]
        # Sessile consumer (rare sessile animal): dark red
        sessile_cons = has_consume & ~has_photo & ~has_move
        colors[sessile_cons] = [140, 30, 30]
        # Mobile omnivore: blue-purple
        mobile_omni = has_photo & has_consume & has_move
        colors[mobile_omni] = [140, 80, 220]
        # Sessile omnivore: muted purple
        sessile_omni = has_photo & has_consume & ~has_move
        colors[sessile_omni] = [90, 50, 140]
        # Size by energy for mobile organisms (bigger dots drawn later)

    elif color_mode == "energy":
        e_norm = np.clip(world.energy / world.cfg.energy_max, 0, 1)
        colors[:, 0] = (220 * (1.0 - e_norm)).astype(np.uint8)
        colors[:, 1] = (230 * e_norm).astype(np.uint8)
        colors[:, 2] = (160 * np.maximum(0, e_norm - 0.6) / 0.4).astype(np.uint8)

    elif color_mode == "modules":
        mc = world.module_present.sum(axis=1).astype(np.float64)
        mc_norm = np.clip((mc - 2) / 6.0, 0, 1)
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
        has_def = world.module_active[:, M_DEFENSE]
        colors[~has_def] = [60, 180, 80]
        if has_def.any():
            shell = 1.0 / (1.0 + np.exp(-world.weights[has_def, _DEFENSE_OFF]))
            colors[has_def, 0] = (200 + 55 * shell).astype(np.uint8)
            colors[has_def, 1] = (180 * shell).astype(np.uint8)
            colors[has_def, 2] = 30

    elif color_mode == "social":
        has_soc = world.module_active[:, M_SOCIAL]
        has_med = world.module_active[:, M_MEDIATE]
        colors[:] = [50, 140, 60]
        if has_soc.any():
            rs = np.clip(world.relationship_score[has_soc] / 3.0, 0, 1)
            colors[has_soc, 0] = (40 + 60 * rs).astype(np.uint8)
            colors[has_soc, 1] = (160 + 90 * rs).astype(np.uint8)
            colors[has_soc, 2] = (180 + 75 * rs).astype(np.uint8)
        if has_med.any():
            colors[has_med] = [255, 210, 80]

    elif color_mode == "manipulation":
        thresh = world.cfg.repro_manip_threshold
        ls = world.lysogenic_strength
        has_vresist = world.module_active[:, M_VRESIST]
        colors[:] = [40, 180, 60]
        carrier = (ls > 0.01) & (ls < thresh)
        if carrier.any():
            c_norm = np.clip(ls[carrier] / thresh, 0, 1)
            colors[carrier, 0] = (80 + 140 * c_norm).astype(np.uint8)
            colors[carrier, 1] = (180 - 40 * c_norm).astype(np.uint8)
            colors[carrier, 2] = (60 - 40 * c_norm).astype(np.uint8)
        manip = ls >= thresh
        if manip.any():
            m_norm = np.clip((ls[manip] - thresh) / (1.0 - thresh + 1e-8), 0, 1)
            colors[manip, 0] = (220 + 35 * m_norm).astype(np.uint8)
            colors[manip, 1] = (100 * (1.0 - m_norm)).astype(np.uint8)
            colors[manip, 2] = (20 * (1.0 - m_norm)).astype(np.uint8)
        if has_vresist.any():
            colors[has_vresist, 0] = np.minimum(255, colors[has_vresist, 0].astype(np.int16) + 40).astype(np.uint8)
            colors[has_vresist, 1] = np.minimum(255, colors[has_vresist, 1].astype(np.int16) + 60).astype(np.uint8)
            colors[has_vresist, 2] = np.minimum(255, colors[has_vresist, 2].astype(np.int16) + 120).astype(np.uint8)

    elif color_mode == "hijack":
        hi = world.hijack_intensity if hasattr(world, 'hijack_intensity') else np.zeros(n)
        vl = world.viral_load
        has_vresist = world.module_active[:, M_VRESIST]
        colors[:] = [40, 160, 150]
        low = (vl > 0) & (hi <= 0.3)
        if low.any():
            vl_norm = np.clip(vl[low] / world.cfg.viral_burst_threshold, 0, 1)
            colors[low, 0] = (100 + 130 * vl_norm).astype(np.uint8)
            colors[low, 1] = (180 - 30 * vl_norm).astype(np.uint8)
            colors[low, 2] = (80 * (1.0 - vl_norm)).astype(np.uint8)
        heavy = hi > 0.3
        if heavy.any():
            h_norm = np.clip((hi[heavy] - 0.3) / 0.7, 0, 1)
            colors[heavy, 0] = (220 + 35 * h_norm).astype(np.uint8)
            colors[heavy, 1] = (120 * (1.0 - h_norm)).astype(np.uint8)
            colors[heavy, 2] = (20 * (1.0 - h_norm)).astype(np.uint8)
        if has_vresist.any():
            colors[has_vresist, 0] = (colors[has_vresist, 0].astype(np.int16) * 0.7).astype(np.uint8)
            colors[has_vresist, 1] = np.minimum(255, colors[has_vresist, 1].astype(np.int16) + 50).astype(np.uint8)
            colors[has_vresist, 2] = np.minimum(255, colors[has_vresist, 2].astype(np.int16) + 80).astype(np.uint8)

    elif color_mode == "endosymbiosis":
        mc = world.merger_count
        rel = world.relationship_score
        has_social = world.module_active[:, M_SOCIAL]
        has_move = world.module_active[:, M_MOVE]
        endo_thresh = world.cfg.endo_relationship_threshold
        colors[:] = [40, 60, 140]
        ready = has_social & (rel >= endo_thresh * 0.5) & (mc == 0)
        if ready.any():
            r_norm = np.clip(rel[ready] / endo_thresh, 0, 1)
            colors[ready, 0] = (40 + 40 * r_norm).astype(np.uint8)
            colors[ready, 1] = (100 + 140 * r_norm).astype(np.uint8)
            colors[ready, 2] = (180 + 75 * r_norm).astype(np.uint8)
        comp1 = mc == 1
        if comp1.any():
            colors[comp1] = [255, 200, 60]
        deep = mc >= 2
        if deep.any():
            d_norm = np.clip((mc[deep] - 1) / 3.0, 0, 1)
            colors[deep, 0] = 255
            colors[deep, 1] = (210 + 45 * d_norm).astype(np.uint8)
            colors[deep, 2] = (80 + 175 * d_norm).astype(np.uint8)
        sessile = ~has_move
        if sessile.any():
            colors[sessile] = (colors[sessile].astype(np.int16) * 0.6).clip(0, 255).astype(np.uint8)

    elif color_mode == "shedding":
        # NEW Phase 4: Module usage / capacity shedding
        #   Bright green = all modules well-used (avg usage > 0.6)
        #   Yellow       = some modules underused (avg 0.3–0.6)
        #   Orange→Red   = heavy dormancy / shedding active
        #   Cyan ring    = has dormant modules (present but inactive)
        present_count = world.module_present.sum(axis=1).astype(np.float64)
        present_count = np.maximum(present_count, 1)
        # Average usage across present modules
        usage_sum = (world.module_usage * world.module_present).sum(axis=1)
        avg_usage = usage_sum / present_count
        dormant = (world.module_present & ~world.module_active).sum(axis=1)

        u_norm = np.clip(avg_usage, 0, 1)
        # Green (high usage) → yellow → red (low usage)
        colors[:, 0] = (60 + 195 * (1.0 - u_norm)).astype(np.uint8)
        colors[:, 1] = (200 * u_norm + 80 * (1.0 - u_norm)).astype(np.uint8)
        colors[:, 2] = (40 * u_norm).astype(np.uint8)

        # Cyan tint for organisms with dormant modules
        has_dormant = dormant > 0
        if has_dormant.any():
            colors[has_dormant, 0] = np.maximum(0, colors[has_dormant, 0].astype(np.int16) - 30).astype(np.uint8)
            colors[has_dormant, 1] = np.minimum(255, colors[has_dormant, 1].astype(np.int16) + 40).astype(np.uint8)
            colors[has_dormant, 2] = np.minimum(255, colors[has_dormant, 2].astype(np.int16) + 100).astype(np.uint8)

    elif color_mode == "cascade":
        # NEW Phase 4: Genomic incompatibility cascade
        #   Green   = phase 0 (no cascade)
        #   Yellow  = phase 1 (metabolic disruption)
        #   Orange  = phase 2 (regulatory breakdown)
        #   Red     = phase 3 (identity dissolution)
        #   White   = high genomic stress (pre-cascade buildup)
        phase = world.genomic_cascade_phase
        stress = world.genomic_stress

        colors[:] = [40, 180, 60]  # Phase 0: healthy green

        p1 = phase == 1
        if p1.any():
            colors[p1] = [220, 200, 40]

        p2 = phase == 2
        if p2.any():
            colors[p2] = [240, 130, 30]

        p3 = phase == 3
        if p3.any():
            colors[p3] = [255, 40, 40]

        # Stress glow on phase-0 organisms (building toward cascade)
        stressed = (phase == 0) & (stress > 1.0)
        if stressed.any():
            s_norm = np.clip((stress[stressed] - 1.0) / 2.0, 0, 1)
            colors[stressed, 0] = (40 + 100 * s_norm).astype(np.uint8)
            colors[stressed, 1] = (180 + 60 * s_norm).astype(np.uint8)
            colors[stressed, 2] = (60 + 140 * s_norm).astype(np.uint8)

    elif color_mode == "development":
        # NEW Phase 4: Developmental dependency
        #   Bright white  = mature (dev_progress >= 1.0)
        #   Blue→Cyan     = immature, progressing (within dev window)
        #   Red           = compromised (past window, not mature)
        #   Dim           = deeply compromised (very low progress past window)
        is_mat = world.is_mature
        dev = world.dev_progress
        past_window = (world.age > world.cfg.dev_window_length) & ~is_mat

        # Mature: bright white-green
        colors[is_mat] = [200, 240, 220]

        # Immature, within window: blue to cyan as progress increases
        in_window = ~is_mat & ~past_window
        if in_window.any():
            p_norm = np.clip(dev[in_window], 0, 1)
            colors[in_window, 0] = (30 + 50 * p_norm).astype(np.uint8)
            colors[in_window, 1] = (80 + 140 * p_norm).astype(np.uint8)
            colors[in_window, 2] = (180 + 75 * p_norm).astype(np.uint8)

        # Compromised: orange to red
        if past_window.any():
            deficit = 1.0 - np.minimum(dev[past_window], 1.0)
            colors[past_window, 0] = (180 + 75 * deficit).astype(np.uint8)
            colors[past_window, 1] = (80 * (1.0 - deficit)).astype(np.uint8)
            colors[past_window, 2] = (30 * (1.0 - deficit)).astype(np.uint8)

    elif color_mode == "age":
        a_norm = np.clip(world.age / world.cfg.max_age, 0, 1)
        brightness = (255 * (1.0 - a_norm * 0.7)).astype(np.uint8)
        colors[:, 0] = brightness
        colors[:, 1] = brightness
        colors[:, 2] = (180 * (1.0 - a_norm)).astype(np.uint8)

    dot_size = max(2, scale - 1)
    has_move_arr = world.module_active[:, M_MOVE] if n > 0 else np.array([], dtype=bool)
    for i in range(n):
        x = int(cols[i]) * scale
        y = int(rows[i]) * scale + y_offset
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        if has_move_arr[i]:
            # Mobile organisms: slightly larger, rounded feel
            sz = dot_size + 1
            pygame.draw.rect(surface, color, (x, y, sz, sz))
        else:
            # Sessile organisms: standard square
            pygame.draw.rect(surface, color, (x, y, dot_size, dot_size))


# ═══════════════════════════════════════════════════════════════════════════════
# STATS PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def draw_stats_panel(surface, world, x_offset, color_mode, steps_per_frame,
                     paused, layers, elapsed, win_h=800):
    font = pygame.font.SysFont("monospace", 11)

    panel_rect = pygame.Rect(x_offset, 0, STATS_WIDTH, win_h)
    pygame.draw.rect(surface, (15, 15, 22), panel_rect)
    pygame.draw.line(surface, (60, 60, 80), (x_offset, 0), (x_offset, win_h), 2)

    lines = []
    s = world.stats_history[-1] if world.stats_history else {}

    lines.append(("THE SHIMMERING FIELD", (200, 180, 255)))
    lines.append(("Phase 4 Step 6: Sessile-Mobile", (140, 130, 170)))
    lines.append(("", None))

    state = "\u2590\u2590 PAUSED" if paused else f"\u25B6 {steps_per_frame} steps/frame"
    lines.append((f"t = {world.timestep:,}   {state}", (255, 255, 255)))
    lines.append((f"Sim time: {elapsed:.1f}s", (150, 150, 150)))
    lines.append(("", None))

    # ── Population ──
    lines.append(("\u2500\u2500 Population \u2500\u2500", (100, 180, 255)))
    lines.append((f"  Alive:  {world.pop:,}", (255, 255, 255)))
    if s:
        lines.append((f"  Energy: {s.get('avg_energy', 0):5.1f}  Gen: {s.get('max_gen', 0)}", (180, 230, 180)))
        lines.append((f"  Modules: {s.get('avg_modules', 0):.2f} avg", (200, 200, 200)))
    lines.append(("", None))

    # ── Ecology ──
    lines.append(("\u2500\u2500 Ecology \u2500\u2500", (100, 220, 130)))
    if s:
        r = s.get("roles", {})
        lines.append((f"  \u25CF Producer:    {r.get('producer',0):4d}", (60, 210, 90)))
        lines.append((f"  \u25CF Herbivore:   {r.get('herbivore',0):4d}", (200, 100, 200)))
        lines.append((f"  \u25CF Carnivore:   {r.get('carnivore',0):4d}", (230, 55, 55)))
        lines.append((f"  \u25CF Detritivore: {r.get('detritivore',0):4d}", (180, 130, 50)))
        lines.append((f"  \u25CF Omnivore:    {r.get('omnivore',0):4d}", (100, 140, 230)))
        lines.append((f"  Kills: {s.get('kills',0):3d}  ({s.get('total_kills',0):,} total)", (255, 120, 120)))
    lines.append(("", None))

    # ── Sessile-Mobile Divergence (NEW) ──
    lines.append(("\u2500\u2500 Body Plans \u2500\u2500", (120, 200, 100)))
    if s:
        sp = s.get('sessile_producers', 0)
        mp = s.get('mobile_producers', 0)
        mc_anim = s.get('mobile_consumers', 0)
        sc = s.get('sessile_consumers', 0)
        lines.append((f"  \u25A0 Sessile prod: {sp:4d}", (30, 160, 50)))
        lines.append((f"  \u25B6 Mobile prod:  {mp:4d}", (160, 255, 60)))
        lines.append((f"  \u25B6 Mobile cons:  {mc_anim:4d}", (230, 90, 40)))
        lines.append((f"  \u25A0 Sessile cons: {sc:4d}", (140, 30, 30)))
        total = max(sp + mp + mc_anim + sc, 1)
        sessile_pct = (sp + sc) / total * 100
        mobile_pct = (mp + mc_anim) / total * 100
        lines.append((f"  Sessile: {sessile_pct:.0f}%  Mobile: {mobile_pct:.0f}%", (180, 200, 180)))
    lines.append(("", None))

    # ── Fungal Networks (NEW headline) ──
    lines.append(("\u2500\u2500 Fungal Network \u2500\u2500", (220, 160, 60)))
    if s:
        fm = s.get('fungal_mean', 0)
        fx = s.get('fungal_max', 0)
        f_color = (255, 200, 80) if fm > 0.01 else (120, 120, 120)
        lines.append((f"  Density: {fm:.4f} avg / {fx:.3f} max", f_color))
        if fx > 1.0:
            lines.append(("  \u2605 MYCELIAL WEB ACTIVE", (255, 220, 100)))
        elif fx > 0.1:
            lines.append(("  \u25CF Fungal growth detected", (200, 160, 60)))
    lines.append(("", None))

    # ── Ecosystem Integrity / Collapse (NEW) ──
    lines.append(("\u2500\u2500 Ecosystem \u2500\u2500", (120, 200, 120)))
    if s:
        avg_int = s.get('avg_integrity', 0)
        collapsed = s.get('collapsed_zones', 0)
        # Color integrity: green if high, red if low
        int_r = int(min(255, max(0, 255 - avg_int * 400)))
        int_g = int(min(255, avg_int * 400))
        lines.append((f"  Integrity: {avg_int:.3f} avg", (int_r, int_g, 60)))
        if collapsed > 0:
            col_color = (255, 60, 60) if collapsed > 500 else (255, 180, 80)
            lines.append((f"  Collapsed: {collapsed:,} cells", col_color))
            if collapsed > 2000:
                lines.append(("  \u26A0 MASS COLLAPSE", (255, 40, 40)))
            elif collapsed > 500:
                lines.append(("  \u26A0 Regional collapse", (255, 140, 60)))
        else:
            lines.append(("  No collapsed zones", (100, 180, 100)))
    lines.append(("", None))

    # ── Capacity Shedding (NEW) ──
    lines.append(("\u2500\u2500 Shedding \u2500\u2500", (180, 200, 140)))
    if s:
        dormant = s.get('dormant_modules', 0)
        avg_use = s.get('avg_usage', 0)
        use_color = (80, 220, 80) if avg_use > 0.5 else (220, 180, 40) if avg_use > 0.2 else (220, 80, 40)
        lines.append((f"  Usage: {avg_use:.3f} avg", use_color))
        d_color = (200, 220, 255) if dormant > 0 else (120, 120, 120)
        lines.append((f"  Dormant: {dormant} modules", d_color))
    lines.append(("", None))

    # ── Genomic Cascade (NEW) ──
    lines.append(("\u2500\u2500 Genomic Cascade \u2500\u2500", (200, 140, 100)))
    if s:
        casc = s.get('cascade_organisms', 0)
        max_ph = s.get('max_cascade_phase', 0)
        avg_gs = s.get('avg_genomic_stress', 0)
        gs_color = (80, 200, 80) if avg_gs < 1.0 else (220, 180, 40) if avg_gs < 2.0 else (240, 80, 40)
        lines.append((f"  Stress: {avg_gs:.3f} avg", gs_color))
        ph_color = [(80, 200, 80), (220, 200, 40), (240, 130, 30), (255, 40, 40)][min(max_ph, 3)]
        lines.append((f"  Cascading: {casc} orgs (max P{max_ph})", ph_color))
        if max_ph >= 3:
            lines.append(("  \u26A0 IDENTITY DISSOLUTION", (255, 40, 40)))
    lines.append(("", None))

    # ── Development (NEW) ──
    lines.append(("\u2500\u2500 Development \u2500\u2500", (140, 180, 220)))
    if s:
        mat_frac = s.get('mature_fraction', 0)
        comp_count = s.get('compromised_count', 0)
        mat_color = (80, 220, 180) if mat_frac > 0.8 else (220, 200, 60) if mat_frac > 0.5 else (220, 80, 60)
        lines.append((f"  Mature: {mat_frac:.1%}", mat_color))
        if comp_count > 0:
            lines.append((f"  Compromised: {comp_count}", (220, 100, 60)))
    lines.append(("", None))

    # ── Endosymbiosis ──
    lines.append(("\u2500\u2500 Endosymbiosis \u2500\u2500", (255, 210, 80)))
    if s:
        total_mrg = s.get('total_mergers', 0)
        comp_count_e = s.get('composite_organisms', 0)
        max_depth = s.get('max_merger_count', 0)
        mrg_color = (255, 220, 100) if total_mrg > 0 else (120, 120, 120)
        lines.append((f"  Mergers: {total_mrg:4d}  Comp: {comp_count_e:4d}  D:{max_depth}", mrg_color))
    lines.append(("", None))

    # ── Parasitic Systems (compact) ──
    lines.append(("\u2500\u2500 Parasitic \u2500\u2500", (200, 80, 200)))
    if s:
        hjk = s.get('hijack_fraction', 0)
        lyso = s.get('lyso_fraction', 0)
        lines.append((f"  Hijacked: {hjk:.1%}  Lysogenic: {lyso:.1%}", (180, 120, 200)))
        if hjk > 0.3:
            lines.append(("  \u26A0 BEHAVIORAL OVERRIDE", (255, 80, 220)))
        if lyso > 0.5:
            lines.append(("  \u26A0 DIVERSITY COLLAPSE RISK", (255, 60, 60)))
    lines.append(("", None))

    # ── Environment ──
    lines.append(("\u2500\u2500 Environment \u2500\u2500", (255, 180, 80)))
    if s:
        lines.append((f"  Toxic: {s.get('toxic_mean', 0):.3f}  Decomp: {s.get('decomp_mean', 0):.2f}", (255, 200, 120)))
        lines.append((f"  Nutrient: {s.get('nutrient_mean', 0):.3f}", (80, 200, 100)))
    lines.append(("", None))

    # ── Layers ──
    lines.append(("\u2500\u2500 Layers \u2500\u2500", (150, 150, 150)))
    layer_keys = [
        ("toxic", "1:Toxic"), ("organisms", "2:Organisms"),
        ("decomp", "3:Decomp"), ("viral", "4:Viral"),
        ("social", "5:Social"), ("mediator", "6:Mediator"),
        ("nutrients", "7:Nutrients"),
        ("fungal", "8:Fungal"), ("integrity", "9:Integrity"),
    ]
    for key, name in layer_keys:
        on = layers[key]
        dot = "\u25CF" if on else "\u25CB"
        lines.append((f"  {dot} {name}", (180, 255, 180) if on else (80, 80, 80)))
    lines.append(("", None))

    cm_names = {
        "role": "Ecological Role", "mobility": "Sessile/Mobile", "energy": "Energy",
        "modules": "Module Count", "generation": "Generation", "defense": "Defense",
        "social": "Social/Mediator", "manipulation": "Manipulation",
        "hijack": "Hijack Intensity", "endosymbiosis": "Endosymbiosis",
        "shedding": "Capacity Shedding", "cascade": "Genomic Cascade",
        "development": "Development", "age": "Age",
    }
    lines.append((f"  Color: {cm_names.get(color_mode, color_mode)}", (200, 200, 255)))
    lines.append(("", None))

    # ── Legend (context-sensitive) ──
    if color_mode == "role":
        lines.append(("\u2500\u2500 Legend \u2500\u2500", (120, 120, 120)))
        lines.append(("  \u25CF Producer", (60, 210, 90)))
        lines.append(("  \u25CF Herbivore", (200, 100, 200)))
        lines.append(("  \u25CF Carnivore", (230, 55, 55)))
        lines.append(("  \u25CF Detritivore", (180, 130, 50)))
        lines.append(("  \u25CF Omnivore", (100, 140, 230)))
    elif color_mode == "mobility":
        lines.append(("\u2500\u2500 Legend \u2500\u2500", (120, 120, 120)))
        lines.append(("  \u25A0 Sessile producer", (30, 160, 50)))
        lines.append(("  \u25B6 Mobile producer", (160, 255, 60)))
        lines.append(("  \u25B6 Mobile consumer", (230, 90, 40)))
        lines.append(("  \u25A0 Sessile consumer", (140, 30, 30)))
        lines.append(("  \u25B6 Mobile omnivore", (140, 80, 220)))
        lines.append(("  \u25A0 Sessile omnivore", (90, 50, 140)))
    elif color_mode == "shedding":
        lines.append(("\u2500\u2500 Legend \u2500\u2500", (120, 120, 120)))
        lines.append(("  \u25CF Well-used", (60, 200, 40)))
        lines.append(("  \u25CF Underused", (220, 180, 40)))
        lines.append(("  \u25CF Shedding", (240, 80, 30)))
        lines.append(("  \u25CF +dormant (cyan tint)", (100, 200, 220)))
    elif color_mode == "cascade":
        lines.append(("\u2500\u2500 Legend \u2500\u2500", (120, 120, 120)))
        lines.append(("  \u25CF Phase 0: Healthy", (40, 180, 60)))
        lines.append(("  \u25CF Phase 1: Metabolic", (220, 200, 40)))
        lines.append(("  \u25CF Phase 2: Regulatory", (240, 130, 30)))
        lines.append(("  \u25CF Phase 3: Dissolution", (255, 40, 40)))
    elif color_mode == "development":
        lines.append(("\u2500\u2500 Legend \u2500\u2500", (120, 120, 120)))
        lines.append(("  \u25CF Mature", (200, 240, 220)))
        lines.append(("  \u25CF Developing", (60, 160, 220)))
        lines.append(("  \u25CF Compromised", (220, 60, 30)))
    elif color_mode == "hijack":
        lines.append(("\u2500\u2500 Legend \u2500\u2500", (120, 120, 120)))
        lines.append(("  \u25CF Uninfected", (40, 160, 150)))
        lines.append(("  \u25CF Low hijack", (200, 170, 50)))
        lines.append(("  \u25CF Heavy hijack", (240, 60, 20)))
    elif color_mode == "endosymbiosis":
        lines.append(("\u2500\u2500 Legend \u2500\u2500", (120, 120, 120)))
        lines.append(("  \u25CF Ordinary", (40, 60, 140)))
        lines.append(("  \u25CF Merger-ready", (80, 220, 255)))
        lines.append(("  \u2605 Composite", (255, 200, 60)))

    # ── Controls ──
    lines.append(("", None))
    lines.append(("\u2500\u2500 Controls \u2500\u2500", (120, 120, 120)))
    for ctrl in ["SPACE  Pause/Resume", "UP/DN  Speed +/-",
                 "1-9    Toggle layers", "V      Cycle colors",
                 "R      Reset", "Q/ESC  Quit"]:
        lines.append((f"  {ctrl}", (100, 100, 110)))

    y = 8
    for text, color in lines:
        if color is None:
            y += 3
            continue
        if y > win_h - 12:
            break
        surf = font.render(text, True, color)
        surface.blit(surf, (x_offset + 10, y))
        y += 13


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    pygame.display.set_caption("The Shimmering Field \u2014 Phase 4 Step 6: Sessile-Mobile Divergence")

    cfg = Config()
    N = cfg.grid_size

    grid_px = N * WINDOW_SCALE
    win_w = grid_px + STATS_WIDTH
    win_h = max(grid_px, MIN_WIN_HEIGHT)
    grid_y_offset = (win_h - grid_px) // 2

    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    world = World(cfg)

    paused = False
    steps_per_frame = INITIAL_STEPS_PER_FRAME
    # 14 color modes
    color_modes = ["role", "mobility", "energy", "modules", "generation", "defense",
                   "social", "manipulation", "hijack", "endosymbiosis",
                   "shedding", "cascade", "development", "age"]
    color_idx = 0
    color_mode = color_modes[0]
    layers = {
        "toxic": True,
        "organisms": True,
        "decomp": False,
        "viral": False,
        "social": False,
        "mediator": False,
        "nutrients": False,
        "fungal": False,
        "integrity": False,
    }

    sim_start = _time.time()
    running = True

    while running:
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
                    layers["social"] = not layers["social"]
                elif event.key == pygame.K_6:
                    layers["mediator"] = not layers["mediator"]
                elif event.key == pygame.K_7:
                    layers["nutrients"] = not layers["nutrients"]
                elif event.key == pygame.K_8:
                    layers["fungal"] = not layers["fungal"]
                elif event.key == pygame.K_9:
                    layers["integrity"] = not layers["integrity"]

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

        if layers["integrity"]:
            int_rgb = render_heatmap(world.ecosystem_integrity, INTEGRITY_CMAP, vmin=0, vmax=0.7)
            mask_i = world.ecosystem_integrity < 0.65
            base_rgb[mask_i] = np.clip(
                base_rgb[mask_i].astype(np.int16) + int_rgb[mask_i].astype(np.int16),
                0, 255).astype(np.uint8)
            # Bright red overlay for collapsed zones
            if world.zone_collapsed.any():
                base_rgb[world.zone_collapsed, 0] = np.minimum(255,
                    base_rgb[world.zone_collapsed, 0].astype(np.int16) + 80).astype(np.uint8)

        if layers["nutrients"]:
            nutr_rgb = render_heatmap(world.nutrients, NUTRIENT_CMAP, vmin=0, vmax=1.0)
            mask_n = world.nutrients > 0.01
            base_rgb[mask_n] = np.clip(
                base_rgb[mask_n].astype(np.int16) + nutr_rgb[mask_n].astype(np.int16),
                0, 255).astype(np.uint8)

        if layers["fungal"]:
            fungal_rgb = render_heatmap(world.fungal_density, FUNGAL_CMAP, vmin=0, vmax=2.0)
            mask_f = world.fungal_density > 0.005
            base_rgb[mask_f] = np.clip(
                base_rgb[mask_f].astype(np.int16) + fungal_rgb[mask_f].astype(np.int16),
                0, 255).astype(np.uint8)

        if layers["toxic"]:
            toxic_rgb = render_heatmap(world.toxic, TOXIC_CMAP, vmin=0, vmax=2.0)
            mask = world.toxic > 0.02
            base_rgb[mask] = np.clip(
                base_rgb[mask].astype(np.int16) + toxic_rgb[mask].astype(np.int16),
                0, 255).astype(np.uint8)

        if layers["decomp"]:
            decomp_rgb = render_heatmap(world.decomposition, DECOMP_CMAP, vmin=0, vmax=8.0)
            mask_d = world.decomposition > 0.05
            base_rgb[mask_d] = np.clip(
                base_rgb[mask_d].astype(np.int16) + decomp_rgb[mask_d].astype(np.int16),
                0, 255).astype(np.uint8)

        if layers["viral"]:
            viral_rgb = render_heatmap(world.viral_particles, VIRAL_CMAP, vmin=0, vmax=5.0)
            mask_v = world.viral_particles > 0.05
            base_rgb[mask_v] = np.clip(
                base_rgb[mask_v].astype(np.int16) + viral_rgb[mask_v].astype(np.int16),
                0, 255).astype(np.uint8)

        if layers["social"]:
            social_sum = world.social_field[:, :, 0] + world.social_field[:, :, 1]
            social_rgb = render_heatmap(social_sum, SOCIAL_CMAP, vmin=0, vmax=3.0)
            mask_s = social_sum > 0.05
            base_rgb[mask_s] = np.clip(
                base_rgb[mask_s].astype(np.int16) + social_rgb[mask_s].astype(np.int16),
                0, 255).astype(np.uint8)

        if layers["mediator"]:
            med_rgb = render_heatmap(world.mediator_field, MEDIATOR_CMAP, vmin=0, vmax=2.0)
            mask_m = world.mediator_field > 0.02
            base_rgb[mask_m] = np.clip(
                base_rgb[mask_m].astype(np.int16) + med_rgb[mask_m].astype(np.int16),
                0, 255).astype(np.uint8)

        grid_surface = render_grid_to_surface(base_rgb, WINDOW_SCALE)
        screen.blit(grid_surface, (0, grid_y_offset))

        if layers["organisms"]:
            draw_organisms(screen, world, WINDOW_SCALE, color_mode, y_offset=grid_y_offset)

        draw_stats_panel(screen, world, grid_px, color_mode, steps_per_frame,
                         paused, layers, elapsed, win_h)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
