"""
The Shimmering Field — Phase 3 Step 9 Live Visualizer
======================================================
Runs the simulation and renders it in real-time using Pygame.
Phase 3 capstone: all modules, all parasitic systems, endosymbiosis.

Usage:
    pip install pygame numpy scipy
    python visualizer_p3s9.py

Place this file in the same directory as phase3_step9.py.

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
    V          — Cycle organism coloring:
                   Role → Energy → Modules → Generation → Defense
                   → Social → Manipulation → Hijack → Endosymbiosis → Age
    R          — Reset simulation
    Q / ESC    — Quit
"""

import numpy as np
import pygame
import sys
import time as _time

from phase3_step9 import (
    Config, World,
    M_PHOTO, M_CHEMO, M_CONSUME, M_MOVE, M_FORAGE, M_DEFENSE, M_DETOX,
    M_TOXPROD, M_VRESIST, M_SOCIAL, M_MEDIATE,
    N_MODULES, MODULE_NAMES, MODULE_WEIGHT_SIZES
)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_SCALE = 5
STATS_WIDTH = 340
FPS = 60
INITIAL_STEPS_PER_FRAME = 1
MIN_WIN_HEIGHT = 800

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


def render_heatmap(grid, cmap, vmin=0.0, vmax=1.0):
    normalized = np.clip((grid - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    indices = (normalized * 255).astype(np.uint8)
    return cmap[indices]


def render_grid_to_surface(rgb_array, scale):
    h, w = rgb_array.shape[:2]
    small_surf = pygame.surfarray.make_surface(rgb_array.transpose(1, 0, 2))
    return pygame.transform.scale(small_surf, (w * scale, h * scale))


# ═══════════════════════════════════════════════════════════════════════════════
# ORGANISM RENDERING — 10 color modes (Step 7 had 8)
# ═══════════════════════════════════════════════════════════════════════════════

C_PRODUCER    = np.array([60, 210, 90],  dtype=np.uint8)
C_HERBIVORE   = np.array([200, 100, 200], dtype=np.uint8)
C_CARNIVORE   = np.array([230, 55, 55],  dtype=np.uint8)
C_DETRITIVORE = np.array([180, 130, 50], dtype=np.uint8)
C_OMNIVORE    = np.array([100, 140, 230], dtype=np.uint8)

# Precompute weight offsets for classification
_CONSUME_OFF = int(sum(MODULE_WEIGHT_SIZES[:M_CONSUME]))
_DEFENSE_OFF = int(sum(MODULE_WEIGHT_SIZES[:M_DEFENSE]))


def classify_organisms(world):
    """0=producer, 1=herbivore, 2=carnivore, 3=detritivore, 4=omnivore."""
    n = world.pop
    roles = np.zeros(n, dtype=np.int32)
    has_prod = world.module_present[:, M_PHOTO] | world.module_present[:, M_CHEMO]
    has_cons = world.module_present[:, M_CONSUME]
    roles[has_prod & has_cons] = 4  # omnivore
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
        # Lysogenic manipulation state:
        #   Green = clean, Yellow = carrier, Orange→Red = manipulated, Cyan = VRESIST
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
        # NEW Step 8: Behavioral hijack intensity visualization
        #   Teal       = uninfected (no viral load)
        #   Yellow     = low hijack (viral_load present, intensity < 0.3)
        #   Orange→Red = heavy hijack (intensity 0.3–1.0, movement overridden)
        #   Dim        = VRESIST organisms (partially resistant, darker tint)
        hi = world.hijack_intensity if hasattr(world, 'hijack_intensity') else np.zeros(n)
        vl = world.viral_load
        has_vresist = world.module_active[:, M_VRESIST]

        # Base: teal for uninfected
        colors[:] = [40, 160, 150]

        # Any viral load but low hijack: yellow-ish
        low = (vl > 0) & (hi <= 0.3)
        if low.any():
            vl_norm = np.clip(vl[low] / world.cfg.viral_burst_threshold, 0, 1)
            colors[low, 0] = (100 + 130 * vl_norm).astype(np.uint8)
            colors[low, 1] = (180 - 30 * vl_norm).astype(np.uint8)
            colors[low, 2] = (80 * (1.0 - vl_norm)).astype(np.uint8)

        # Heavy hijack: orange to red
        heavy = hi > 0.3
        if heavy.any():
            h_norm = np.clip((hi[heavy] - 0.3) / 0.7, 0, 1)
            colors[heavy, 0] = (220 + 35 * h_norm).astype(np.uint8)
            colors[heavy, 1] = (120 * (1.0 - h_norm)).astype(np.uint8)
            colors[heavy, 2] = (20 * (1.0 - h_norm)).astype(np.uint8)

        # VRESIST: darker tint (resistance dampens hijack)
        if has_vresist.any():
            colors[has_vresist, 0] = (colors[has_vresist, 0].astype(np.int16) * 0.7).astype(np.uint8)
            colors[has_vresist, 1] = np.minimum(255, colors[has_vresist, 1].astype(np.int16) + 50).astype(np.uint8)
            colors[has_vresist, 2] = np.minimum(255, colors[has_vresist, 2].astype(np.int16) + 80).astype(np.uint8)

    elif color_mode == "endosymbiosis":
        # NEW Step 9: Endosymbiosis / composite organism visualization
        #   Dim blue   = never merged (merger_count == 0)
        #   Cyan       = has SOCIAL + high relationship (merger-ready)
        #   Gold       = first-generation composite (merger_count == 1)
        #   White-gold = deep composite lineage (merger_count >= 2)
        #   Ring       = sessile organisms (no MOVE module)
        mc = world.merger_count
        rel = world.relationship_score
        has_social = world.module_active[:, M_SOCIAL]
        has_move = world.module_active[:, M_MOVE]
        endo_thresh = world.cfg.endo_relationship_threshold

        # Base: dim blue for ordinary organisms
        colors[:] = [40, 60, 140]

        # Merger-ready: SOCIAL + high relationship → cyan glow
        ready = has_social & (rel >= endo_thresh * 0.5) & (mc == 0)
        if ready.any():
            r_norm = np.clip(rel[ready] / endo_thresh, 0, 1)
            colors[ready, 0] = (40 + 40 * r_norm).astype(np.uint8)
            colors[ready, 1] = (100 + 140 * r_norm).astype(np.uint8)
            colors[ready, 2] = (180 + 75 * r_norm).astype(np.uint8)

        # First-generation composites: gold
        comp1 = mc == 1
        if comp1.any():
            colors[comp1] = [255, 200, 60]

        # Deep composites: white-gold, brighter with depth
        deep = mc >= 2
        if deep.any():
            d_norm = np.clip((mc[deep] - 1) / 3.0, 0, 1)
            colors[deep, 0] = 255
            colors[deep, 1] = (210 + 45 * d_norm).astype(np.uint8)
            colors[deep, 2] = (80 + 175 * d_norm).astype(np.uint8)

        # Sessile organisms: dim down slightly (rooted, can't seek partners)
        sessile = ~has_move
        if sessile.any():
            colors[sessile] = (colors[sessile].astype(np.int16) * 0.6).clip(0, 255).astype(np.uint8)

    elif color_mode == "age":
        a_norm = np.clip(world.age / world.cfg.max_age, 0, 1)
        brightness = (255 * (1.0 - a_norm * 0.7)).astype(np.uint8)
        colors[:, 0] = brightness
        colors[:, 1] = brightness
        colors[:, 2] = (180 * (1.0 - a_norm)).astype(np.uint8)

    dot_size = max(2, scale - 1)
    for i in range(n):
        x = int(cols[i]) * scale
        y = int(rows[i]) * scale + y_offset
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
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
    lines.append(("Phase 3 Step 9: Endosymbiosis", (140, 130, 170)))
    lines.append(("", None))

    state = "▐▐ PAUSED" if paused else f"▶ {steps_per_frame} steps/frame"
    lines.append((f"t = {world.timestep:,}   {state}", (255, 255, 255)))
    lines.append((f"Sim time: {elapsed:.1f}s", (150, 150, 150)))
    lines.append(("", None))

    # ── Population ──
    lines.append(("── Population ──", (100, 180, 255)))
    lines.append((f"  Alive:  {world.pop:,}", (255, 255, 255)))
    if s:
        lines.append((f"  Energy: {s.get('avg_energy', 0):5.1f}  Gen: {s.get('max_gen', 0)}", (180, 230, 180)))
        lines.append((f"  Modules: {s.get('avg_modules', 0):.2f} avg", (200, 200, 200)))
    lines.append(("", None))

    # ── Ecology (5 roles) ──
    lines.append(("── Ecology ──", (100, 220, 130)))
    if s:
        r = s.get("roles", {})
        lines.append((f"  ● Producer:    {r.get('producer',0):4d}", (60, 210, 90)))
        lines.append((f"  ● Herbivore:   {r.get('herbivore',0):4d}", (200, 100, 200)))
        lines.append((f"  ● Carnivore:   {r.get('carnivore',0):4d}", (230, 55, 55)))
        lines.append((f"  ● Detritivore: {r.get('detritivore',0):4d}", (180, 130, 50)))
        lines.append((f"  ● Omnivore:    {r.get('omnivore',0):4d}", (100, 140, 230)))
        lines.append((f"  Kills: {s.get('kills',0):3d}  ({s.get('total_kills',0):,} total)", (255, 120, 120)))
    lines.append(("", None))

    # ── Modules ──
    lines.append(("── Modules ──", (200, 180, 100)))
    if s:
        mc = s.get("module_counts", {})
        mod_colors = {
            "PHOTO": (100, 220, 100), "CHEMO": (80, 200, 200),
            "CONSUME": (230, 80, 80), "MOVE": (180, 180, 220),
            "FORAGE": (220, 200, 80), "DEFENSE": (200, 160, 60),
            "DETOX": (120, 220, 180), "VRESIST": (160, 130, 220),
            "SOCIAL": (100, 200, 255), "MEDIATE": (255, 210, 80),
        }
        for name in MODULE_NAMES:
            if name == "TOXPROD":
                continue
            col = mod_colors.get(name, (160, 160, 160))
            lines.append((f"  {name:8s} {mc.get(name, 0):5d}", col))
    lines.append(("", None))

    # ── Endosymbiosis (NEW — the headline feature of Step 9) ──
    lines.append(("── Endosymbiosis ──", (255, 210, 80)))
    if s:
        total_mrg = s.get('total_mergers', 0)
        comp_count = s.get('composite_organisms', 0)
        max_depth = s.get('max_merger_count', 0)
        # Gold text for active mergers
        mrg_color = (255, 220, 100) if total_mrg > 0 else (140, 140, 140)
        lines.append((f"  Mergers:     {total_mrg:4d} total", mrg_color))
        comp_color = (255, 200, 60) if comp_count > 0 else (140, 140, 140)
        lines.append((f"  Composites:  {comp_count:4d} alive", comp_color))
        depth_color = (255, 230, 180) if max_depth >= 2 else (180, 180, 180)
        lines.append((f"  Max depth:   {max_depth:4d}", depth_color))
        # Milestone banners
        if max_depth >= 3:
            lines.append(("  ★ DEEP SYMBIOTIC LINEAGE", (255, 255, 200)))
        elif total_mrg >= 10:
            lines.append(("  ● Active merger ecology", (255, 220, 140)))
    lines.append(("", None))

    # ── Hijacking (Step 8) ──
    lines.append(("── Hijacking ──", (200, 80, 200)))
    if s:
        hjk_frac = s.get('hijack_fraction', 0)
        hjk_steps = s.get('hijacked_steps', 0)
        # Purple intensity based on fraction
        hjk_r = int(min(255, 120 + hjk_frac * 300))
        hjk_b = int(min(255, 140 + hjk_frac * 200))
        lines.append((f"  Hijacked:    {hjk_frac:.1%} of pop", (hjk_r, 80, hjk_b)))
        lines.append((f"  Total steps: {hjk_steps:,}", (180, 120, 200)))
        if hjk_frac > 0.3:
            lines.append(("  ⚠ BEHAVIORAL OVERRIDE", (255, 80, 220)))
    lines.append(("", None))

    # ── Manipulation (Step 7) ──
    lines.append(("── Manipulation ──", (255, 100, 60)))
    if s:
        lyso_frac = s.get('lyso_fraction', 0)
        manip_births = s.get('manipulated_births', 0)
        frac_r = int(min(255, lyso_frac * 400))
        frac_g = int(max(0, 255 - lyso_frac * 400))
        lines.append((f"  Lysogenic:  {lyso_frac:.1%} of pop", (frac_r, frac_g, 60)))
        lines.append((f"  Manip births: {manip_births:,} total", (255, 160, 100)))
        if lyso_frac > 0.5:
            lines.append(("  ⚠ DIVERSITY COLLAPSE RISK", (255, 60, 60)))
        elif lyso_frac > 0.3:
            lines.append(("  ⚠ Homogenization spreading", (255, 180, 80)))
    lines.append(("", None))

    # ── Social / Mediator ──
    lines.append(("── Social Systems ──", (100, 200, 255)))
    if s:
        lines.append((f"  Relationship: {s.get('avg_relationship', 0):.2f} avg / {s.get('max_relationship', 0):.2f} max", (130, 210, 255)))
        lines.append((f"  Mediator:     {s.get('mediator_field_mean', 0):.3f}", (255, 210, 80)))
        lines.append((f"  Immune exp:   {s.get('avg_immune_exp', 0):.3f}", (160, 130, 220)))
    lines.append(("", None))

    # ── Environment ──
    lines.append(("── Environment ──", (255, 180, 80)))
    if s:
        lines.append((f"  Toxic:    {s.get('toxic_mean', 0):.3f}", (255, 200, 120)))
        lines.append((f"  Decomp:   {s.get('decomp_mean', 0):.2f}", (200, 160, 80)))
        lines.append((f"  Nutrient: {s.get('nutrient_mean', 0):.3f}", (80, 200, 100)))
    lines.append(("", None))

    # ── Layers ──
    lines.append(("── Layers ──", (150, 150, 150)))
    layer_keys = [
        ("toxic", "1:Toxic"), ("organisms", "2:Organisms"),
        ("decomp", "3:Decomp"), ("viral", "4:Viral"),
        ("social", "5:Social"), ("mediator", "6:Mediator"),
        ("nutrients", "7:Nutrients"),
    ]
    for key, name in layer_keys:
        on = layers[key]
        lines.append((f"  {'●' if on else '○'} {name}", (180, 255, 180) if on else (80, 80, 80)))
    lines.append(("", None))

    cm_names = {
        "role": "Ecological Role", "energy": "Energy", "modules": "Module Count",
        "generation": "Generation", "defense": "Defense", "social": "Social/Mediator",
        "manipulation": "Manipulation", "hijack": "Hijack Intensity",
        "endosymbiosis": "Endosymbiosis", "age": "Age",
    }
    lines.append((f"  Color: {cm_names.get(color_mode, color_mode)}", (200, 200, 255)))
    lines.append(("", None))

    # ── Legend (context-sensitive) ──
    if color_mode == "role":
        lines.append(("── Legend ──", (120, 120, 120)))
        lines.append(("  ● Producer", (60, 210, 90)))
        lines.append(("  ● Herbivore", (200, 100, 200)))
        lines.append(("  ● Carnivore", (230, 55, 55)))
        lines.append(("  ● Detritivore", (180, 130, 50)))
        lines.append(("  ● Omnivore", (100, 140, 230)))
        lines.append(("", None))
    elif color_mode == "manipulation":
        lines.append(("── Legend ──", (120, 120, 120)))
        lines.append(("  ● Clean", (40, 180, 60)))
        lines.append(("  ● Carrier", (200, 160, 30)))
        lines.append(("  ● Manipulated", (240, 60, 20)))
        lines.append(("  ● +VRESIST (cyan tint)", (100, 180, 220)))
        lines.append(("", None))
    elif color_mode == "hijack":
        lines.append(("── Legend ──", (120, 120, 120)))
        lines.append(("  ● Uninfected", (40, 160, 150)))
        lines.append(("  ● Low hijack", (200, 170, 50)))
        lines.append(("  ● Heavy hijack", (240, 60, 20)))
        lines.append(("  ● VRESIST (cool tint)", (50, 150, 180)))
        lines.append(("", None))
    elif color_mode == "endosymbiosis":
        lines.append(("── Legend ──", (120, 120, 120)))
        lines.append(("  ● Ordinary", (40, 60, 140)))
        lines.append(("  ● Merger-ready", (80, 220, 255)))
        lines.append(("  ★ Composite (1st gen)", (255, 200, 60)))
        lines.append(("  ★ Deep composite (2+)", (255, 240, 200)))
        lines.append(("  dim = sessile", (80, 80, 80)))
        lines.append(("", None))

    # ── Controls ──
    lines.append(("── Controls ──", (120, 120, 120)))
    for ctrl in ["SPACE  Pause/Resume", "UP/DN  Speed +/-",
                 "1-7    Toggle layers", "V      Cycle colors",
                 "R      Reset", "Q/ESC  Quit"]:
        lines.append((f"  {ctrl}", (100, 100, 110)))

    y = 8
    for text, color in lines:
        if color is None:
            y += 4
            continue
        surf = font.render(text, True, color)
        surface.blit(surf, (x_offset + 10, y))
        y += 15


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    pygame.display.set_caption("The Shimmering Field — Phase 3 Step 9: Endosymbiosis")

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
    # 10 color modes: 8 from Step 7 + hijack (Step 8) + endosymbiosis (Step 9)
    color_modes = ["role", "energy", "modules", "generation", "defense",
                   "social", "manipulation", "hijack", "endosymbiosis", "age"]
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

        if layers["nutrients"]:
            nutr_rgb = render_heatmap(world.nutrients, NUTRIENT_CMAP, vmin=0, vmax=1.0)
            mask_n = world.nutrients > 0.01
            base_rgb[mask_n] = np.clip(
                base_rgb[mask_n].astype(np.int16) + nutr_rgb[mask_n].astype(np.int16),
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
