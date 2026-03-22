# The Shimmering Field — Phase 4 Step 5: Complete Technical Reference

## What This File Is

`phase4_step5.py` is a 2,762-line NumPy-based artificial life simulation. It models a 128×128 grid where digital organisms with modular genomes compete, cooperate, parasitize, merge, collapse, and recover. This is the capstone of Phase 4, which adds five failure/recovery systems on top of the Phase 3 ecology. The file is fully self-contained — one `Config` class, one `World` class, one `run_simulation()` entry point.

The simulation runs for 10,000 timesteps. Each timestep executes ~22 operations in a fixed causal order. There are no neural networks, no gradient descent — evolution happens through mutation, horizontal gene transfer, viral integration, and endosymbiotic merger. The organisms are arrays of floats, not objects.

---

## Architecture Overview

The simulation has accumulated systems across four phases:

**Phase 1** (Steps 1–3): Grid, light, toxicity, basic photosynthesis, movement, reproduction, death/decomposition.

**Phase 2** (Steps 1–4): Horizontal gene transfer (3-stratum geological system), viral dynamics (lytic/lysogenic), predation with CONSUME module.

**Phase 3** (Steps 1–9): Nine specialized modules (FORAGE, DEFENSE, DETOX, VRESIST, SOCIAL, MEDIATE) plus reproductive manipulation (Wolbachia-style), behavioral hijacking (Toxoplasma-style), and endosymbiosis (organism mergers).

**Phase 4** (Steps 1–5, THIS FILE): Five failure/recovery systems that make the ecosystem both more fragile and more resilient:

| Step | System | What It Does |
|------|--------|-------------|
| 1 | Capacity Shedding | Unused modules decay → dormancy → permanent loss |
| 2 | Genomic Incompatibility | Too much HGT → 3-phase cascade breakdown |
| 3 | Developmental Dependency | Newborns need compatible neighbors to mature |
| 4 | Nonlinear Collapse | Per-cell ecosystem integrity tracking + zone collapse |
| 5 | Fungal Networks | Decomposition-fueled grid infrastructure for nutrient/genome transport |

These five systems interact with each other and with the Phase 3 parasitic stack. Five failure modes can compound — each overlapping failure multiplies energy costs by 1.2×. Two simultaneous failures = 1.44× costs, three = 1.73×, all five = 2.49×.

---

## The Grid

The world is a 128×128 toroidal grid. Each cell holds continuous-valued environmental layers:

| Layer | Range | Dynamics |
|-------|-------|----------|
| `light` | 0.05–1.0 | Static gradient, bright at top, dim at bottom |
| `toxic` | 0–5.0 | Produced by organisms (TOXPROD), diffuses, decays. Zone map modulates production rates |
| `nutrients` | 0–3.0 | Slow baseline regen + faster regen from decomposition and DETOX byproducts |
| `decomposition` | 0–30.0 | Deposited on death, decays slowly (0.998×/step), diffuses locally |
| `decomp_scent` | ≥0 | Gaussian blur of decomposition (σ=5), used for detritivore navigation |
| `density` | 0–N | Integer count of organisms per cell, recomputed each step |
| `social_field` | 0–10.0 | Two channels: producer signal and consumer signal. Decays 50%/step |
| `mediator_field` | 0–10.0 | Pollination/dispersal service availability. Decays 97%/step |
| `viral_particles` | 0–20.0 | Free-floating viral load. Diffuses (0.08), decays (0.01/step) |
| `ecosystem_integrity` | 0–1.0 | **Phase 4**: Weighted health score per cell (smoothed 7×7) |
| `zone_collapsed` | bool | **Phase 4**: True if integrity dropped below 0.28 in a populated area |
| `fungal_density` | 0–5.0 | **Phase 4 Step 5**: Mycelial network density. Grows from decomp, diffuses, decays |

Additionally, three stratified fragment pools store genome fragments at different geological depths (recent/intermediate/ancient), each with weight values, module presence, and blended genome vectors. These are the substrate for horizontal gene transfer.

The `zone_map` divides the grid into 8×8 zones with varying toxic production multipliers (0.3–2.0), smoothed with a uniform filter. This creates a heterogeneous landscape: some zones are naturally toxic, others are clean.

---

## The Genome

Each organism carries a two-layer genome:

**Layer 1: Module presence** — A boolean vector of length 11. Each position represents one module capability. An organism can have any combination of modules, but each adds maintenance and expression costs.

**Layer 2: Evolvable weights** — A float vector of length 48 (36 module-specific weights + 4 standalone parameters + 8 reserved). These weights are interpreted through sigmoid/tanh activations to produce continuous behavioral parameters. They mutate, blend during HGT, and recombine during endosymbiosis.

### The 11 Modules

| ID | Name | Weights | Role | Maintenance + Expression Cost |
|----|------|---------|------|------------------------------|
| 0 | PHOTO | 4 | Photosynthesis. Light → energy. Sessile producers get +50% bonus | 0.30 |
| 1 | CHEMO | 4 | Chemosynthesis. Toxin → energy via Michaelis-Menten kinetics | 0.37 |
| 2 | CONSUME | 4 | Predation + detritivory. prey_selectivity controls herbivore/carnivore gradient | 0.20 |
| 3 | MOVE | 8 | Locomotion. 8 weights control multi-factor movement decisions. Cost scales with total module count | 0.14 |
| 4 | FORAGE | 4 | Enhanced resource gathering. Extraction bonus, storage capacity, cooperative signaling | 0.18 |
| 5 | DEFENSE | 4 | Anti-predation. Shell (reduces hit probability), camouflage (skip chance), counter-attack | 0.28 |
| 6 | DETOX | 4 | Toxin metabolism. Removes toxins from environment + gains energy from them | 0.30 |
| 7 | TOXPROD | 0 | Universal toxic waste. Always present, no evolvable weights, never sheds | 0.05 |
| 8 | VRESIST | 4 | Viral immune system. Specificity/breadth tradeoff, immune memory from surviving infections | 0.23 |
| 9 | SOCIAL | 4 | Social signaling. Identity signal, compatibility assessment, relationship building | 0.15 |
| 10 | MEDIATE | 4 | Pollination/dispersal. Helps neighbors reproduce, supports development, earns passive energy | 0.11 |

**Phase 4 module states**: In Phase 4, each module has three possible states — *absent* (not in genome), *active* (present and functioning), or *dormant* (present but deactivated by capacity shedding). Dormant modules still cost maintenance but not expression. Stress can reactivate dormant modules.

### Standalone Parameters

Four additional evolvable parameters not tied to any module:

| Index | Name | Function |
|-------|------|----------|
| 0 | Transfer Receptivity | How readily the organism absorbs HGT fragments (sigmoid → 0–1) |
| 1 | Transfer Selectivity | How picky — higher selectivity requires closer genome distance for absorption |
| 2 | Viral Resistance | Baseline infection resistance for organisms without VRESIST (~0.3 effective) |
| 3 | Lysogenic Suppression | Baseline ability to suppress dormant viral DNA activation (~0.3 effective) |

### Metabolic Interference

Organisms with both producer modules (PHOTO/CHEMO) and CONSUME are classified as omnivores. They pay an asymmetric penalty: production efficiency drops to 85% and hunting efficiency drops to 30%. Obligate consumers (CONSUME without producers) get a +50% specialist bonus. This drives role differentiation — generalists pay a tax, specialists are rewarded.

---

## Ecological Roles

Organisms are classified into five roles based on their module inventory and CONSUME weight parameters:

| Role | Definition | Typical Strategy |
|------|-----------|-----------------|
| **Producer** | Has PHOTO or CHEMO, no CONSUME | Sessile or mobile photosynthesizer/chemosynthesizer |
| **Herbivore** | Obligate consumer, low decomp_preference, low prey_selectivity | Hunts producers. Extracts more energy per kill (1.2×) from abundant easy prey |
| **Carnivore** | Obligate consumer, low decomp_preference, high prey_selectivity | Hunts other consumers. Higher success bonus (+15%) against harder prey, but extracts less (0.9×) |
| **Detritivore** | Obligate consumer, high decomp_preference | Scavenges decomposition. Reliable energy from carrion, no predation needed |
| **Omnivore** | Has both producer and CONSUME modules | Jack-of-all-trades with metabolic interference penalty |

The herbivore/carnivore distinction is continuous, not binary. The `prey_selectivity` weight (sigmoid to 0–1) determines where on the gradient an organism falls. At 0, it's a pure herbivore preferring producers; at 1, a pure carnivore preferring consumers.

---

## The Update Loop

Each timestep executes these operations in order:

```
 1. _update_density()              — Count organisms per cell
 2. _sense_local()                 — Each organism reads 22 values from its cell
 3. _compute_hijack_intensity()    — Compute viral behavioral override strength
 4. _effective_energy_max()        — FORAGE storage bonus
 5. _acquire_energy()              — Photosynthesis, chemosynthesis, detritivory, FORAGE coop
 6. [hijack energy suppression]    — Heavy hijack reduces energy acquisition up to 70%
 7. ★ _apply_genomic_incompatibility()  — Phase 4 Step 2: cascade effects (every 10 steps)
 8. ★ _update_development()        — Phase 4 Step 3: maturation progress
 9. ★ _compute_dev_compromise()    — Phase 4 Step 3: energy penalty for compromised organisms
10. [add energy, apply toxic damage, apply detox]
11. ★ _compute_compounding_stress() — Phase 4: multiply costs by 1.2^(failure_count)
12. [subtract costs × compound multiplier]
13. _decide_movement() + _execute_movement()  — Weight-based movement with hijack override
14. ★ [dev compromise aging]       — Phase 4 Step 3: compromised organisms age 1.5× faster
15. _update_social_field() + _apply_social_interactions()  — Every 2 steps
16. _update_mediator_field()       — Every 2 steps (includes dev support reward)
17. _apply_nutrient_cycling()      — DETOX/CONSUME/FORAGE nutrient side effects
18. ★ _update_fungal_network()     — Phase 4 Step 5: growth, spread, decay (every 3 steps)
19. ★ _fungal_nutrient_transport() — Phase 4 Step 5: redistribute nutrients along fungi
20. ★ _fungal_toxic_conduit()      — Phase 4 Step 5: accelerate toxic diffusion along fungi
21. ★ _update_ecosystem_integrity() — Phase 4 Step 4: compute health score (every 5 steps)
22. ★ _update_collapse_state()     — Phase 4 Step 4: hysteresis collapse/recovery
23. ★ [collapse energy penalty]    — Phase 4 Step 4: 0.3/step drain in collapsed zones
24. ★ _update_module_usage()       — Phase 4 Step 1: relevance scoring (every 5 steps)
25. ★ [collapse shedding acceleration] — Phase 4: 2× decay in collapsed zones
26. ★ _apply_capacity_shedding()   — Phase 4 Step 1: dormancy + genome loss
27. _predation()                   — Every step
28. _horizontal_transfer()         — Every 5 steps
29. _viral_dynamics()              — Every 3 steps
30. _attempt_endosymbiosis()       — Every 10 steps
31. _reproduce()                   — With collapse penalty + mediator bonus
32. _kill_and_decompose()          — Death, decomp deposit, viral burst, strata update
33. _update_environment()          — Diffusion, decay, sedimentation, clamping
34. _record_stats()                — JSON snapshot
```

Operations marked ★ are Phase 4 additions.

---

## Phase 3 Systems (Inherited)

### Energy Acquisition

**Photosynthesis** (PHOTO): Energy = base (3.0) × efficiency × toxic_penalty × light^sensitivity × storage × shade_from_density × producer_penalty_if_omnivore × forage_bonus. Sessile producers (no MOVE) get a 50% bonus — they're rooted, better oriented.

**Chemosynthesis** (CHEMO): Energy = base (2.2) × efficiency × Michaelis-Menten(toxic, saturation) × shade. Chemosynthesizers thrive in toxic zones — toxin is their food.

**Detritivory** (CONSUME with high decomp_preference): Directly consumes decomposition from the grid. extraction_rate and intake_cap scale with decomp_preference. Consumes the decomposition it eats.

### Predation

Runs every step. Uses a flat spatial index (cell_key → sorted organism indices) for efficient neighbor lookup. The pipeline:

1. Each predator (CONSUME holder) checks nearby cells within hunt radius (1 for sessile, 2 for mobile).
2. Target selection weights by inverse energy (prefer weaker prey) × prey type preference.
3. Success probability = base + hunt_skill − target_difficulty − shell_defense + type_match_bonus.
4. On success: predator gains fraction of prey's energy (modulated by herbivore/carnivore type). Prey dies.
5. On failure: if target has counter-attack, predator takes energy damage.
6. Per-cell kill cap (1) prevents local extinction cascades.
7. Satiation (70% chance) makes fed predators skip their next hunt.
8. Behavioral hijack suppresses all defense values (shell, camo, counter-attack).

Camouflage is pre-rolled per step — each organism has one camo roll that all predators check against.

### Horizontal Gene Transfer (HGT)

Three-stratum geological system. When organisms die, their genome fragments deposit into the "recent" stratum. Over time, fragments sediment through recent → intermediate → ancient layers. Each layer decays at different rates.

Living organisms absorb fragments from the strata they can access:
- **Recent**: everyone can access. Blend rate 10%.
- **Intermediate**: requires local toxic ≥ 0.3. Blend rate 18%.
- **Ancient**: requires local toxic ≥ 0.8. Blend rate 30%.

Transfer receptivity (standalone param, sigmoid) controls probability. Selectivity filters by genome distance — picky organisms only absorb similar DNA. Module acquisition happens probabilistically: if a fragment contains a module the organism lacks, there's a 1.5% chance per transfer event of gaining it.

**Phase 4 Step 5 addition**: Organisms on fungal tissue get up to +0.3 transfer receptivity bonus, capped at 1.0. Fungal networks act as genome highways.

**Phase 4 Step 2 link**: Each HGT event adds genomic stress proportional to stratum depth — recent (1.0×), intermediate (1.5×), ancient (2.5×). This feeds the genomic cascade system.

### Viral Dynamics

Eight initial viral hotspots seed the grid. Viral particles diffuse and decay continuously. Every 3 steps:

**Infection**: Uninfected organisms in cells with viral particles > 0.05 face infection probability scaled by local viral density × (1 − resistance). VRESIST holders have much stronger resistance (up to 0.95 effective) than standalone (~0.3).

**Lytic pathway** (60% of infections): viral_load starts at 0.1, grows by 0.1/step, deals 2.0 × load energy damage per step. At viral_load ≥ 1.0, the organism bursts — dies and releases 8.0 viral particles in a 3-cell radius, carrying its blended genome into the viral pool. Survivors gain immune experience.

**Lysogenic pathway** (40%): Viral genome integrates into the organism's lysogenic_genome at 30% blend rate. lysogenic_strength accumulates. Under toxic stress (> 0.6), lysogenic DNA can activate — blending back into the host's weights and creating a low-level lytic infection (0.2 load).

### Reproductive Manipulation (Wolbachia-style)

Organisms with lysogenic_strength > 0.3 have their offspring subtly modified: up to 15% weight blend toward the viral genome template, boosted transfer receptivity (+0.4), and a viability cost (3.0 energy penalty) for offspring whose genomes diverge too far from the viral template. Self-limiting at 70% population lysogenic fraction. This is a slow, invisible takeover — the virus doesn't kill, it shapes the next generation.

### Behavioral Hijacking (Toxoplasma-style)

Organisms with lytic viral_load > 0.05 experience behavioral override proportional to intensity (0 at load=0.05, maxing at load=0.5). Effects:
- Movement overridden: seek high-density, low-viral areas (spread virus to new hosts)
- Defense suppressed up to 60% (can't protect yourself under hijack)
- VRESIST suppressed up to 40%
- Energy acquisition blocked up to 70% at heavy hijack
- Heavy hijack (>0.7 intensity): erratic movement noise
- Toxic stress amplifies intensity by 1.5×

### Endosymbiosis

Every 10 steps, eligible pairs in the same cell can merge into a composite organism:

**Prerequisites**: relationship_score ≥ 0.6 (built from co-location, accelerated by SOCIAL), energy ≥ 50 (both), no active viral infection, local toxic in sweet spot (0.05–1.0), module complementarity ≥ 1 different module.

**Merger**: Creates one new organism with union of both parents' modules (capped at 8), blended weights (50/50), combined energy (× 1.2 bonus), merger_count incremented. Both originals are removed. VRESIST holders merge at 50% reduced probability (immune system resists foreign integration).

**Module priority when capped**: Energy sources (PHOTO/CHEMO/CONSUME) > SOCIAL > MOVE > TOXPROD > FORAGE/DEFENSE > DETOX > VRESIST > MEDIATE.

### Social System

Social field has two channels: producer signal and consumer signal. All organisms emit weak identity signals (0.3); SOCIAL module holders emit stronger signals (up to 1.0). Signals decay 50% per step.

Relationship score accumulates from compatible co-location. Producers near consumers build faster than same-type. Sessile organisms build 3× faster (permanent neighbors). SOCIAL holders get 5× growth rate. Relationship is the gate to endosymbiosis.

### Mediator System

MEDIATE holders deposit pollination/dispersal signals on the grid. Nearby reproducing organisms get threshold reduction (up to 0.25 per mediator unit). Mediators earn energy rewards for facilitating reproduction (reward_sensitivity weight controls how much). Network coordination: multiple mediators amplify each other's signals.

**Phase 4 Step 3 link**: Mediators near immature organisms earn passive energy (0.15/step, capped at 3 immature neighbors). This creates a development support role.

---

## Phase 4 Systems (New in This File)

### Phase 4 Step 1: Capacity Shedding

**Concept**: Use-it-or-lose-it. Modules that aren't useful in current environmental conditions decay toward dormancy and eventually permanent genome loss. This models the real biological principle of regressive evolution — cave fish losing eyes, parasites shedding metabolic pathways.

**Module usage tracking**: Each module has a usage score [0, 1]. Every 5 steps:
- **Gain**: Active modules in relevant conditions gain 0.015 usage. Relevance is environment-specific:
  - PHOTO: relevant when light > 0.5 AND toxicity < 0.5
  - CHEMO/DETOX: relevant when toxicity > 0.3
  - CONSUME/DEFENSE: scales with local density (more neighbors = more relevant)
  - MOVE: high relevance when crowded (0.8), low when sparse (0.2)
  - FORAGE: relevant when nutrients > 1.0
  - VRESIST: scales with viral particle density (10× amplified)
  - SOCIAL/MEDIATE: scale with density (need neighbors)
  - TOXPROD: always 1.0 (structural, never sheds)
- **Decay**: Three tiers of decay rate per check:
  - Behavioral (MOVE, SOCIAL, MEDIATE): 0.025/check → ~200 steps to deactivation
  - Immune (DEFENSE, VRESIST): 0.012/check → ~400 steps
  - Metabolic (PHOTO, CHEMO, CONSUME, FORAGE, DETOX): 0.008/check → ~600 steps

**State transitions**:
- Usage < 0.15 → module goes **dormant** (present but inactive, still costs maintenance but not expression)
- Usage < 0.03 → module **lost from genome** (irreversible)
- TOXPROD never sheds. Last energy source (PHOTO/CHEMO/CONSUME) is always protected — you can't shed your only way to eat.

**Stress reactivation**: When local toxic > 0.3 or viral particles > 0.1, dormant modules above the loss threshold reactivate with a +0.3 usage boost. Crisis brings back latent capabilities.

**Collapse acceleration**: In collapsed zones (Phase 4 Step 4), shedding decay rates double. Organisms under ecosystem pressure are forced to simplify.

### Phase 4 Step 2: Genomic Incompatibility

**Concept**: Too much horizontal gene transfer builds up "genomic stress" — foreign DNA that the organism can't properly integrate. This models real biological costs of chimeric genomes: regulatory conflicts, protein misfolding, incompatible gene networks.

**Stress accumulation**: Each HGT event adds 0.15 stress, multiplied by stratum depth (recent 1.0×, intermediate 1.5×, ancient 2.5×). Ancient DNA is maximally incompatible.

**Natural decay**: Stress decreases by 0.005/step (accommodation over time).

**Cascade phases** (checked every 10 steps):

| Phase | Threshold | Effect |
|-------|-----------|--------|
| 0 (Healthy) | — | No effects |
| 1 (Metabolic disruption) | effective_stress ≥ 3.0 | Energy acquisition reduced 40% |
| 2 (Regulatory breakdown) | effective_stress ≥ 4.5 | 30% chance per check of random module deactivation |
| 3 (Identity dissolution) | effective_stress ≥ 6.0 | Direct 2.0 energy drain per step (lethal within ~50 steps) |

**Effective stress** = genomic_stress + local_toxic × 1.5. Toxic environments amplify genomic stress — chimeras in toxic zones face compounded pressure.

**Recovery**: When effective stress drops below 0.7× the cascade threshold, the organism steps down one phase.

**Emergency pruning**: At genomic_stress ≥ 4.0, the organism jettisons its least-used modules (synergy with capacity shedding — dormant modules go first). Up to 2 modules lost. Provides 80% stress relief, resets cascade phase to 0. Pruned organisms get +0.5 transfer receptivity — empty genomic slots invite recolonization. This is a desperate survival strategy: lose capabilities to survive.

### Phase 4 Step 3: Developmental Dependency

**Concept**: Newborn organisms need compatible neighbors to develop properly. This models real developmental biology where organisms require environmental signals from other species (gut microbiome colonization, nurse plants in ecology, mycorrhizal inoculation).

**Maturation window**: 75 steps from birth. Organisms accumulate dev_progress toward 1.0:

| Support Source | Progress Rate |
|---------------|--------------|
| Cross-type neighbor (producer near consumer) | 1.0× (full rate of 0.04/step) |
| Same-type neighbor (conspecific) | 0.6× |
| Mediator field | 0.5× |
| Raw co-location (any neighbor) | 0.25× |
| Isolation (no neighbors) | 0.05× (extremely slow solo development) |

Progress sources are additive — a cell with producers, consumers, and mediators gives fast development.

**Maturation**: When dev_progress ≥ 1.0, the organism becomes permanently mature. No further penalties.

**Compromise**: If age exceeds the 75-step window without maturation, the organism is permanently compromised. Penalties scale with how far from maturation (deficit = 1.0 − dev_progress):
- Energy acquisition penalty: up to 50% reduction
- Reproduction threshold: × 1.5 (harder to reproduce)
- Aging speed: × 1.5 (shorter lifespan)

**Founding organisms**: The initial population starts fully mature (they bootstrapped the ecosystem). Endosymbiotic composites are also born mature (they already developed).

**Offspring**: Start at dev_progress = 0, is_mature = False. Must develop through the neighbor system.

### Phase 4 Step 4: Nonlinear Collapse Dynamics

**Concept**: Per-cell ecosystem health tracking with hysteresis. Ecosystems don't fail linearly — they resist until a tipping point, then collapse rapidly, and require sustained recovery to restore.

**Ecosystem integrity** [0, 1]: Computed every 5 steps from active ecological signals:

| Signal | Weight | Source |
|--------|--------|--------|
| Population density | 45% | min(organisms/cell / 3, 1.0) |
| Mediator service | 20% | min(mediator_field / 0.3, 1.0) |
| Producer/consumer diversity | 20% | min(producers, 1) × min(consumers + 0.1, 1) |
| Fungal density | 15% | min(fungal / 2.0, 1.0) |
| − Toxic penalty | 30% | min(toxic / 0.5, 1.0) × 0.3 |
| − Viral penalty | 15% | min(viral / 0.3, 1.0) × 0.15 |

Smoothed over a 7×7 area (uniform_filter) to prevent cell-level noise.

**Empty cells** are treated as pristine (integrity forced above recovery threshold). Only populated areas can collapse.

**Hysteresis collapse/recovery**:
- Integrity drops below **0.28** in a populated area → zone_collapsed = True
- Integrity must exceed **0.42** to recover (gap of 0.14 prevents oscillation)
- The populated area is expanded by a 5-cell maximum_filter to include nearby zones

**Collapsed zone effects**:
- **Reproduction penalty**: Sigmoid function, up to 80% reduction in reproduction success. The sigmoid makes the penalty gradual near the threshold but steep once integrity drops significantly.
- **Energy penalty**: 0.3 energy drain per step (applied every 5 steps when collapse state is checked)
- **Shedding acceleration**: Capacity shedding decay rates double (organisms forced to simplify under pressure)

### Compounding Stress

Five failure modes are tracked per organism per step:

| Failure | Condition |
|---------|-----------|
| Toxic stress | Local toxic > 0.3 |
| Genomic cascade | cascade_phase > 0 |
| Behavioral hijack | hijack_intensity > 0.1 |
| Developmental compromise | Age > 75 steps AND not mature |
| Ecosystem collapse | Organism in collapsed zone |

Each active failure multiplies base energy costs by 1.2×. This is exponential: `cost_multiplier = 1.2^(failure_count)`. An organism with all five failures active pays 2.49× normal energy costs — almost certainly lethal.

### Phase 4 Step 5: Fungal Networks (This Step)

**Concept**: A grid-level mycelial infrastructure that emerges from decomposition and mediates nutrient and genome transport. Not organisms — a distributed system. Fungi are the bridge between death and recovery: they feed on the dead and channel resources to the living.

**Growth** (every 3 steps):
- Fungi grow where decomposition > 0.5. Growth rate = 0.02 × min(decomp/5, 1.0).
- **Surge**: Decomposition > 5.0 (mass death events) triggers 3× accelerated growth.
- Growth **consumes decomposition** — fungi eat 50% of their growth value in decomp.

**Spread**: Gaussian diffusion (σ=1.5, wrap mode). 5% of density moves to neighbors per update. This creates a slow outward expansion from death sites.

**Decay**: Constant maintenance cost of 0.005/step. Fungi need ongoing decomposition to persist — without fresh death, the network decays.

**Nutrient transport**: Gaussian-blurred average of nutrients (σ=3.0) serves as the "target." Nutrients flow from surplus to deficit zones, scaled by local fungal density. A nutrient-rich death zone connected by fungi to a nutrient-poor living zone will transport nutrients along the mycelial highway. Transport rate: 3% of the fungal_density × gradient per update.

**Toxic conduit**: Fungal networks also accelerate toxic diffusion. This is the double-edged sword: fungi transport nutrients (good) but also propagate toxins (dangerous). A toxic event in one zone can spread faster through fungal conduits to connected zones. Conduit rate: 2% of fungal_density × toxic_gradient per update.

**HGT transport boost**: Organisms standing on fungal tissue get up to +0.3 transfer receptivity, calculated as min(fungal_density × 0.2, 0.3). Genome fragments from dead organisms travel along fungal networks and are more readily absorbed by living organisms on the network. This is the mechanism behind "what grows back carries history of what it replaced."

**The central narrative loop**:
```
Mass death → decomp surge → fungal bloom (3× growth)
  → nutrient redistribution along mycelium
    → new organisms colonize nutrient-rich zones
      → absorb HGT fragments from dead populations (fungal boost)
        → genomic stress from foreign DNA
          → capacity shedding / genomic pruning
            → simplified but adapted organisms emerge
              → "what grows back carries history of what it replaced"
```

---

## Initial Population

300 organisms total, seeded as:

| Type | Fraction | Modules | Initial Position | Special Setup |
|------|----------|---------|-----------------|---------------|
| Mobile producers | ~37% | PHOTO + MOVE + TOXPROD | Random across grid | Default genome |
| Sessile producers | ~40% | PHOTO + TOXPROD (no MOVE) | Random across grid | Better photosynthesis (+50%) |
| Carnivores | 15% | CONSUME + MOVE + TOXPROD | Clustered near center | High aggression, low decomp_pref |
| Detritivores | 8% | CONSUME + MOVE + TOXPROD | Clustered near center | High decomp_pref, low aggression |

Decomposition is seeded in a 20×20 area near center so detritivores have initial food. Eight viral hotspots seed the grid with initial viral particles. All founding organisms start fully mature (dev_progress = 1.0).

---

## Reproduction

Requirements: energy ≥ effective_threshold AND age ≥ 8 AND no active viral load.

The effective threshold is dynamic:
- Base: 80.0 energy
- + density penalty: 5.0 per neighbor (crowding suppression)
- − mediator bonus: up to 0.75 reduction per mediator service unit
- ÷ collapse modifier: sigmoid penalty up to 80% in collapsed zones (making threshold effectively unreachable)

Offspring receive:
- Mutated copy of parent's weights (8% mutation rate, Gaussian noise)
- Copy of parent's modules with small chance of gain (0.5%) or loss (0.3%)
- 30% of parent's genomic stress (epigenetic memory of chimeric history)
- 30% of parent's immune experience (partial maternal immunity)
- 80% of parent's lysogenic strength/genome (vertical viral transmission)
- Reproductive manipulation may bias weights toward viral template (up to 15%)
- dev_progress = 0, is_mature = False (must develop through neighbor system)
- module_usage = 1.0 for all present modules (full assumed usefulness at birth)

Placed within 5 cells of parent. Mediators near reproduction events earn energy rewards.

---

## Death and Decomposition

Organisms die from: energy ≤ 0, age ≥ 200, or viral burst (viral_load ≥ 1.0).

On death:
- Decomposition deposited = energy × 0.4 + 2.0 base
- Nutrients deposited = module_count × 0.3 (complex organisms return more)
- Genome fragments deposited into "recent" stratum (weighted average if multiple deaths in same cell)
- Module presence deposited into "recent" stratum module pool
- **Viral burst**: If viral_load ≥ 1.0, organism releases 8.0 viral particles in 3-cell radius, carrying blended genome into viral pool
- **Spontaneous shedding**: Natural deaths have 15% chance of releasing 1.0 viral particles

---

## Sensing

Each organism reads 22 values from the grid at its position:

| Index | Value | Used By |
|-------|-------|---------|
| 0 | Local light | PHOTO, MOVE |
| 1 | Local toxic | CHEMO, DETOX, MOVE, shedding relevance |
| 2 | Local nutrients | MOVE (stay incentive) |
| 3 | Local density (float) | CONSUME, DEFENSE, SOCIAL, MEDIATE, shedding relevance |
| 4–5 | Light gradient (y, x) | MOVE (phototaxis) |
| 6–7 | Toxic gradient (y, x) | MOVE (chemotaxis/avoidance) |
| 8 | Local decomposition | CONSUME (detritivory), shedding relevance |
| 9–10 | Density gradient (y, x) | MOVE (density seeking/avoiding), hijack override |
| 11–12 | Decomp scent gradient (y, x) | MOVE (detritivore navigation) |
| 13–14 | Nutrient gradient (y, x) | MOVE (FORAGE navigation) |
| 15–16 | Social field (producer, consumer) | SOCIAL compatibility |
| 17–20 | Social field gradients (prod y/x, cons y/x) | MOVE (social seeking), MEDIATE navigation |
| 21 | Local mediator field | Development progress |

---

## Movement

Five options: stay (0), up (1), down (2), right (3), left (4). Scores computed for each direction based on weighted combination of gradients:

- **Stay score**: stay_tendency + nutrient_attraction − density_avoidance + decomp_attraction (consumers)
- **Directional scores**: Sum of light_weight × light_gradient − toxic_weight × toxic_gradient + consume_density_weight × density_gradient + consume_scent_weight × scent_gradient + forage_nutrient_weight × nutrient_gradient + detox_toxic_weight × toxic_gradient + social_compat × social_gradient + mediate_dens × social_gradient
- **Behavioral hijack override**: Infected organisms have movement blended with virus goals (seek high-density, avoid existing viral zones). At heavy hijack (>0.7), random noise added.
- **Exploration**: Random noise scaled by random_wt parameter.
- **Sessile**: Organisms without MOVE get −999 for all movement directions (forced stay).

Movement costs 0.2 energy per step plus a complexity surcharge: each module beyond 2 adds 0.05 to MOVE cost (heavier organisms move more expensively).

---

## Output

Every 100 steps, a JSON snapshot is saved containing up to 500 randomly sampled organisms with their full state, plus aggregate statistics. The run summary includes full config and complete stats history.

### Stats Tracked

**Core**: population, average energy, max generation, toxic/decomp/nutrient means, average modules, module counts by type, role distribution, kills.

**Phase 3**: immune experience, relationship scores, mediator field mean, lysogenic fraction, manipulated births, hijack fraction, total mergers, composite organism count.

**Phase 4**: dormant modules (present but inactive), average module usage, average genomic stress, cascade organism count, max cascade phase, mature fraction, compromised count, collapsed zone count, average ecosystem integrity, fungal density mean/max.

---

## Key Design Principles

**No gradient descent**: Evolution is entirely through random mutation, horizontal gene transfer, viral integration, and endosymbiotic merger. There is no fitness function — survival is emergent from energy balance.

**Asymmetric costs**: Generalists pay metabolic interference taxes. Specialists are rewarded but fragile. This drives role differentiation.

**Hysteresis everywhere**: Collapse/recovery thresholds differ (0.28 vs 0.42). Module dormancy is reversible but genome loss isn't. Developmental compromise is permanent. These asymmetries create history-dependence.

**Compounding failures**: No single failure mode is necessarily lethal. But overlapping failures — toxic stress + genomic cascade + ecosystem collapse — multiply costs exponentially and create rapid population crashes.

**Death feeds life**: The decomposition → fungal network → nutrient redistribution → HGT transport pipeline ensures that mass death events create the conditions for novel recovery. What grows back carries the genomic history of what it replaced.

**All parameters are in Config**: Every threshold, rate, cost, and bonus is a named class attribute. The simulation can be fully reconfigured without touching any method code.
