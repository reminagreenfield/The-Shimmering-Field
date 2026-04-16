# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Agent-based artificial life simulation (Python/Pygame) with a Houdini art extraction pipeline.

## Commands

```bash
# Install dependencies
pip install pygame numpy scipy

# Run with interactive Pygame visualizer
python run_shimmering_field.py

# Run headless
python -c "from shimmering_field import Config, run_simulation; run_simulation()"
```

No test suite or linter is configured.

### Visualizer controls

SPACE pause/resume, UP/DOWN speed, 1-9 toggle overlays (toxicity, organisms, decomposition, viral, social, mediator, nutrients, fungal, integrity), V cycle coloring modes, R reset, Q/ESC quit.

## Architecture

Mixin-based `World` class inheriting from 11 mixin classes, each owning one ecological subsystem. New mechanics get their own mixin unless they tightly couple to an existing one — if extending, state why.

Grid: 80×80, ~400–1200 organisms, real-time Pygame visualization.

### Subsystem update order (in `world.update()`)

density → prey scent → sense environment → hijacking → energy acquisition → toxic damage → detox → genomic incompatibility → development → metabolic costs → movement → HGT → viral infection → viral manipulation → predation + shedding → endosymbiosis → social → reproduction → collapse/integrity → aging → death removal → environment diffusion → stats

### Key files

- `run_shimmering_field.py` — top-level entry point
- `shimmering_field/run.py` — creates World + Visualizer, runs simulation loop
- `shimmering_field/world.py` — World class composing all mixins, main `update()` loop
- `shimmering_field/config.py` — all parameters as class attributes on `Config`
- `shimmering_field/constants.py` — module definitions (names, weight counts, costs)
- `shimmering_field/visualizer.py` — Pygame real-time GUI
- Subsystem mixins: `environment.py`, `energy.py`, `predation.py`, `movement.py`, `reproduction.py`, `lifecycle.py`, `viral.py`, `social.py`, `endosymbiosis.py`, `collapse.py`, `stats.py`
- `legacy/` — previous phase implementations (Phases 1-3) with `_explained.md` docs
- `houdini_files/` — Houdini project + exported organism data JSON for 3D visualization

## Genome Structure

Two-layer genome per organism:
- **Module layer** — which capabilities the organism has (11 types)
- **Weight layer** — parameters governing how those capabilities function

Module types: PHOTO, CHEMO, CONSUME, MOVE, FORAGE, DEFENSE, VRESIST, DETOX, TOXPROD, SOCIAL, MEDIATE

Trophic role (producer, herbivore, carnivore, detritivore, omnivore) is **emergent from module inventory**. Never assign it directly.

## Interaction Chains — READ BEFORE MODIFYING ANYTHING

These subsystems compound. Before changing any one of them, trace how the change propagates through all five:

1. **Horizontal gene transfer (HGT)** — primary innovation engine, environmentally modulated rates
2. **Viral dynamics** — prophage integration, stratified temporal expression (recent/intermediate/ancient strata), stress-triggered lytic activation, behavioral hijacking
3. **Capacity shedding** — unused modules degrade (behavioral > metabolic > structural). Irreversible loss recoverable only via HGT
4. **Endosymbiosis** — five-stage pathway with constraint satisfaction (social bonding, metabolic complementarity, co-location, moderate stress, sufficient energy)
5. **Toxic accumulation** — endogenous environmental toxins from metabolism, decomposition layer

The design principle: complexity arises from interaction, not from complicated individual rules. Simple local rules, complex emergent behavior.

## Performance

Real-time visualization is a hard constraint. Prefer vectorized numpy operations over Python loops. Profile before and after changes that touch per-organism or per-cell logic.

## JSON Export Contract

Organism data exports must carry **full state** for the Houdini pipeline:
- Module inventory (list of active module types)
- Weight values (full weight layer)
- Lineage (parent IDs, generation)
- Viral strata (recent/intermediate/ancient, domestication status)
- Endosymbiotic history (partner IDs, merger stage, composite flag)
- Energy, age, position
- Trophic role (as derived, for convenience)
- Hijack state, shedding state, cascade risk

If you add a new attribute to organisms, add it to the export. The Houdini pipeline cannot use what it cannot see.

## Six Compounding Failure Modes

1. Mediator network collapse (Ophiocordyceps) — hijacking worse than absence
2. Developmental symbiont absence (bobtail squid) — developmental windows, specificity checkpoints
3. Reproductive manipulation (Wolbachia) — offspring viability manipulation, diversity collapse
4. Temporal viral expression (HERV) — stress-activated ancient strata with domesticated function
5. Capacity shedding (Buchnera) — irreversible genome reduction
6. Genomic cascade — compounding failures triggering ecosystem collapse

Key insight: success and vulnerability are the same thing at different timescales.

## Houdini Pipeline (21.0)

JSON organism data → point cloud / attribute import → procedural geometry (space colonization branching, module-specific structures) → VDB membrane wrapping → material assignment (gold armature, pearlescent/agate body) → render or fabrication export.

Aesthetic: complexity is internal. Clean translucent membrane over dense branching internals. Module count drives structural density, not surface decoration.

Artworks are documents, not illustrations. Unbroken causal chain from simulation event to physical object.

## What Does Not Yet Exist

- Neural network weight layers (currently flat parameter vectors)
- Full five-stage endosymbiosis with composite identity emergence
- Temporal viral strata with stratified activation
- Dependency network tracking and cascade propagation
- Interpretability layer (monitoring, event flagging, structure library)
- JAX/TPU parallel world execution
- Art extraction pipeline (event capture → 3D geometry → fabrication)

## Biology

All mechanics are grounded in peer-reviewed literature (Margulis 1967 through 2025). If a proposed mechanic contradicts the biological models, flag it. Do not invent biology.

## Code Style

- No preamble in responses. Show code, not descriptions of code.
- When debugging, trace the causal chain through interaction paths. Don't patch symptoms.
- When something looks wrong (performance, correctness, architecture), say so directly.
