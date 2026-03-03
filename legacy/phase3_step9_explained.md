# The Shimmering Field
## Phase 3 Step 9: Endosymbiosis — Technical Architecture Document

*An artificial life simulation where organisms evolve modules, build relationships, contract parasites, hijack each other's behavior, and merge into composite organisms through endosymbiosis.*

---

## 1. Overview

Phase 3 Step 9 is the capstone of the Shimmering Field simulation. It contains **2,130 lines of Python** implementing an artificial ecosystem where digital organisms evolve, interact, and transform on a 128×128 grid. Each organism carries a **48-parameter genome** controlling 11 possible functional modules. The simulation runs for 10,000 timesteps, during which organisms photosynthesize, hunt, scavenge, form social bonds, contract viruses, get behaviorally hijacked, and—new in this step—merge with compatible partners through endosymbiosis.

The codebase is structured as a single Python file with no external dependencies beyond NumPy and SciPy. All organism data is stored as parallel NumPy arrays (Structure-of-Arrays layout) for vectorized computation. The simulation loops through environment updates, organism sensing, energy acquisition, movement, social interactions, predation, viral dynamics, reproduction, and death—all in a specific order that matters for emergent behavior.

### 1.1 What This Step Adds

Step 9 introduces **endosymbiosis**—the merger of two organisms into a single composite entity—alongside **behavioral hijacking** from Step 8 (which was not documented in a previous visualizer). Specifically:

- **Endosymbiosis:** Two co-located organisms with sustained social bonds and metabolic complementarity can merge, combining their module inventories and blending their genomes.
- **Behavioral hijacking:** Lytic viruses override host movement, steering infected organisms toward dense crowds to spread infection (Toxoplasma/cordyceps analogy).
- **Sessile producers:** 40% of the initial population starts without MOVE, receiving a 50% photosynthesis bonus in exchange for immobility.
- **Movement complexity penalty:** Each module beyond 2 adds 0.05 to MOVE cost per step. This pressures composites to shed modules or become sessile.
- **merger_count tracking:** Tracks endosymbiotic lineage depth (0 = never merged, 1+ = composite). Inherited by offspring.

---

## 2. The Module System

Every organism carries a boolean vector of 11 modules—capabilities it either has or doesn't. Each module (except TOXPROD) contributes 4 evolvable weights to the genome, plus MOVE contributes 8. The genome is a flat array of `TOTAL_WEIGHT_PARAMS = 48` floats (44 module weights + 4 standalone parameters). Modules can be gained through mutation during reproduction (0.5% chance), lost (0.3%), or acquired through horizontal gene transfer from geological strata.

| Module | Wts | Function | What the Weights Control |
|--------|-----|----------|--------------------------|
| PHOTO | 4 | Photosynthesis | Light sensitivity, toxic tolerance, storage rate, efficiency |
| CHEMO | 4 | Chemosynthesis | Toxic energy extraction via Michaelis-Menten kinetics |
| CONSUME | 4 | Consumption | Prey selectivity, handling efficiency, decomp preference, aggression |
| MOVE | 8 | Locomotion | Light-seek, density-avoid, nutrient-stay, toxic response, stay tendency, etc. |
| FORAGE | 4 | Enhanced gathering | Extraction efficiency, resource discrimination, storage cap, cooperative signal |
| DEFENSE | 4 | Anti-predation | Shell (block hits), camouflage (avoid detection), size investment, counter-attack |
| DETOX | 4 | Toxin metabolism | Detox efficiency, tolerance, conversion rate, selective uptake |
| TOXPROD | 0 | Toxic waste | Always on. Universal pollution. No evolvable weights. |
| VRESIST | 4 | Viral immunity | Recognition specificity, suppression strength, resistance breadth, immune memory |
| SOCIAL | 4 | Social signaling | Identity signal, compatibility assessment, approach/avoidance, relationship strength |
| MEDIATE | 4 | Pollination | Pollination drive, route memory, network coordination, reward sensitivity |

Modules impose maintenance and expression costs (energy/step). An organism with PHOTO+MOVE+TOXPROD pays ~0.54 energy/step in maintenance. Adding FORAGE, DEFENSE, SOCIAL, VRESIST, etc. stacks costs rapidly. A fully-loaded 8-module composite from endosymbiosis might pay 2+ energy/step, requiring it to be in a resource-rich environment to survive.

### 2.1 Ecological Roles

Organisms aren't assigned roles—roles *emerge* from module combinations. The simulation classifies organisms for stats display based on what modules they have and what their CONSUME weights favor:

- **Producer:** Has PHOTO or CHEMO, no CONSUME. The base of the food web.
- **Herbivore:** Has CONSUME, no producer modules, low prey_selectivity (<0.5). Targets producers.
- **Carnivore:** Has CONSUME, no producer modules, high prey_selectivity (≥0.5). Targets consumers.
- **Detritivore:** Has CONSUME, no producer modules, high decomp_preference (≥0.5). Scavenges dead matter.
- **Omnivore:** Has both producer and consumer modules. Penalized at both tasks (30% hunt efficiency, 85% production).

---

## 3. Environment

The world is a 128×128 grid with seven overlapping environmental layers, each updated every timestep:

- **Light:** Static north-south gradient (1.0 at top, 0.05 at bottom). Drives photosynthesis and movement.
- **Toxicity:** Dynamic. Produced by TOXPROD organisms, modulated by 8×8 zone map, spreads via diffusion, decays slowly. Three damage tiers.
- **Nutrients:** Background food layer. Regenerates passively, enriched by decomposition and nutrient cycling (DETOX/CONSUME/FORAGE side effects).
- **Decomposition:** Organic matter from dead organisms. Detritivores consume it directly. Produces a Gaussian-blurred "scent" layer for navigation.
- **Viral particles:** Free-floating virus on the grid. Diffuses and decays. Deposited by lytic bursts and spontaneous shedding.
- **Social field:** Two channels (producer signal, consumer signal). Organisms sense complementary signals for partner-finding.
- **Mediator field:** Pollination/dispersal service density. MEDIATE organisms deposit signal; nearby organisms get reduced reproduction thresholds.

Additionally, a **stratified geological layer system** stores genomes of dead organisms in three layers (recent, intermediate, ancient) that sediment over time. Living organisms absorb DNA fragments from these layers through horizontal gene transfer, with deeper layers accessible only in toxic environments.

---

## 4. Sensing and Movement

Each organism reads **22 sensor channels** from its immediate environment every timestep. These include local values (light, toxic, nutrients, density, decomp, social signals, mediator field) and *gradients* (differences between the cell above/below and left/right). Gradients tell organisms which direction has more of something.

Movement uses these readings plus the MOVE module's 8 weights to score 5 options: stay, up, down, right, left. Different modules contribute different gradient preferences—PHOTO organisms follow light, CONSUME organisms follow density (carnivores) or decomp scent (detritivores, weighted 8×), SOCIAL organisms seek complementary-type signals, MEDIATE organisms seek populated areas, and DETOX organisms seek toxic zones. The highest-scoring option wins.

Sessile organisms (no MOVE module) are forced to stay. Their movement scores for directions 1–4 are set to −999.

---

## 5. Parasitic and Mutualistic Systems

The simulation layers five distinct biological interaction systems, each introduced in a different Phase 3 step. Together they create a rich web of parasitism, mutualism, and manipulation.

### 5.1 Viral Dynamics (Step 5)

Free-floating viral particles on the grid can infect organisms. 60% of infections go lytic (active replication: viral_load grows 0.1/step, drains 2.0 energy/step, bursts at load 1.0 spraying particles in a 3-cell radius). 40% go lysogenic (virus integrates silently into the genome as lysogenic_strength + lysogenic_genome). Under toxic stress, lysogenic virus can activate into lytic mode, blending the viral genome into the host's weights.

The **VRESIST module** provides much stronger resistance than the standalone fallback. It features a specificity-vs-breadth trade-off (can't max both) and immune memory that accumulates from surviving infections. VRESIST also suppresses lysogenic activation and partially resists behavioral hijacking.

### 5.2 Social Signaling (Step 5–6)

The SOCIAL module enables organisms to broadcast identity signals (producer or consumer type) onto the social field. Nearby organisms with SOCIAL sense complementary signals and accumulate **relationship_score** over time. This score decays at 0.2%/step but grows when compatible organisms remain proximate. Relationship score is the primary gate for endosymbiosis—both partners need ≥1.5 to be eligible.

### 5.3 Mediator Networks (Step 6)

MEDIATE organisms function as pollinators/dispersers. They deposit a service signal on the grid that lowers the reproduction threshold for nearby organisms. When reproduction events happen within their service radius, mediators receive energy rewards proportional to their reward_sensitivity weight. Multiple mediators reinforce each other through network_coordination.

### 5.4 Reproductive Manipulation (Step 7)

When a parent's lysogenic_strength exceeds 0.3, the dormant viral genome silently biases offspring through three mechanisms: (1) **trait bias** blends offspring weights toward the viral template, (2) **receptivity boost** makes offspring more susceptible to horizontal gene transfer, and (3) a **viability filter** penalizes offspring that diverge too far from the viral template. This Wolbachia-style manipulation is self-limiting—it weakens when >70% of the population carries the lysogenic material, preventing complete homogenization.

### 5.5 Behavioral Hijacking (Step 8)

Lytic infections don't just drain energy—they take over behavior. When viral_load falls between 0.05 and 1.0, the organism's hijack_intensity ramps from 0 to 1. At high intensity, the virus overrides movement (seeking dense crowds to spread infection), suppresses defense (up to 60%), suppresses energy acquisition (up to 70%), and adds erratic movement noise. Toxic stress amplifies hijack by 50%. VRESIST organisms partially resist through breadth and immune memory.

| Parameter | Value | Effect |
|-----------|-------|--------|
| `hijack_load_min` | 0.05 | Min viral load for any hijack effects |
| `hijack_load_heavy` | 0.5 | Full-intensity hijack threshold |
| `hijack_defense_suppress` | 0.6 | Max 60% defense suppression at full hijack |
| `hijack_energy_suppress` | 0.7 | Max 70% energy acquisition blocked |
| `hijack_density_seek` | 3.0 | Movement weight toward crowded areas |
| `hijack_stress_amplifier` | 1.5 | Toxic stress amplifies hijack by 50% |

---

## 6. Endosymbiosis: The Phase 3 Capstone

Endosymbiosis is the merger of two organisms into a single composite entity. It is the culmination of all the relational mechanics built in Steps 5–8: social signaling creates relationships, metabolic complementarity creates mutual benefit, environmental stress creates pressure, and the merger creates something genuinely new.

### 6.1 Prerequisites

All six conditions must be satisfied simultaneously. The check runs every 10 timesteps:

1. **SOCIAL module:** Both organisms must have SOCIAL active.
2. **Relationship:** Both must have relationship_score ≥ 1.5 (built up over many steps of compatible proximity).
3. **Energy:** Both must have energy ≥ 50 (can't merge while starving).
4. **No infection:** Neither can have active lytic infection (viral_load must be 0).
5. **Stress window:** Local toxicity must be between 0.05 and 1.0. Too peaceful means no evolutionary pressure to merge; too harsh means the organism can't survive the metabolic disruption.
6. **Complementarity:** The partners must differ in ≥ 2 modules. This ensures the merger creates something genuinely novel—a producer merging with a consumer creates a composite that can do both.

### 6.2 Merger Probability

When all prerequisites are met, the base merger probability is 15%. Two modifiers adjust this: if either partner has VRESIST, probability drops by 50% (the immune system treats the partner as foreign tissue, like transplant rejection). Higher average relationship_score boosts probability up to 2× the base rate. So a deeply bonded pair without immune systems merges at up to 30%, while an immune-active pair with minimum relationship merges at about 7.5%.

### 6.3 Merger Execution

When the roll succeeds, the simulation creates one new composite organism:

- **Modules:** Union of both parents' modules, capped at 8. When the union exceeds 8, lowest-priority modules are dropped. Priority: energy sources (10) > SOCIAL (9) > MOVE (8) > TOXPROD (7) > FORAGE/DEFENSE (6) > DETOX (5) > VRESIST (4) > MEDIATE (3).
- **Weights:** 50/50 blend of both parents' genomes. The composite inherits averaged traits.
- **Energy:** Sum of both parents' energy × 1.2 (synergy bonus), capped at 150.
- **Lineage:** merger_count = max(both parents) + 1. A first merger produces depth 1; two composites merging could produce depth 3+.
- **Cleanup:** Both original organisms are removed from the simulation. The composite starts at age 0 with fresh relationship_score but inherits immune experience and lysogenic material.

### 6.4 Configuration Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `endo_check_interval` | 10 | Steps between endosymbiosis checks |
| `endo_relationship_threshold` | 1.5 | Min relationship_score for both partners |
| `endo_energy_threshold` | 50.0 | Both partners need this much energy |
| `endo_toxic_min` / `max` | 0.05 / 1.0 | Environmental stress sweet spot |
| `endo_complementarity_min` | 2 | Min module differences between partners |
| `endo_probability` | 0.15 | Base 15% merger chance when all conditions met |
| `endo_vresist_penalty` | 0.5 | 50% probability reduction if either has VRESIST |
| `endo_weight_blend` | 0.5 | Equal 50/50 blend of both genomes |
| `endo_energy_bonus` | 1.2 | 20% energy bonus for merged organism |
| `endo_max_modules` | 8 | Max modules in composite (overflow drops lowest priority) |

### 6.5 Evolutionary Consequences

Composite organisms face a fundamental trade-off: they have more capabilities but higher costs. A 6-module composite with MOVE pays 0.20 extra per step in movement complexity alone, plus the combined maintenance of all modules. This creates three evolutionary pressures:

- **Become sessile:** Drop MOVE to save costs. The 50% photosynthesis bonus for sessile organisms makes this viable if the composite has PHOTO.
- **Shed modules:** Through reproduction mutation (0.3% loss rate), offspring gradually lose unnecessary modules over generations.
- **Specialize to rich niches:** Composites must inhabit resource-rich zones (high light + moderate toxicity for CHEMO + abundant decomp for scavenging) to sustain their metabolic load.

---

## 7. Update Loop Order

Each timestep executes 19 operations in a specific order that matters for emergent behavior. For example, hijack intensity is computed before energy acquisition so that energy suppression takes effect immediately, and endosymbiosis is checked after viral dynamics so that freshly cured organisms can merge:

1. Density recount
2. Sense environment (22 channels)
3. **Compute hijack intensity** (before energy, so suppression works)
4. Energy acquisition (with hijack suppression)
5. Toxic damage + DETOX metabolism
6. Pay module costs (with movement complexity penalty)
7. Decide and execute movement (with hijack override)
8. Age + 1
9. Social field update + interactions
10. Mediator field update
11. Nutrient cycling
12. Predation
13. Horizontal gene transfer (every 5 steps)
14. Viral dynamics (every 3 steps)
15. **Endosymbiosis check (every 10 steps)** ← NEW
16. Reproduce (with manipulation)
17. Kill and decompose
18. Update environment
19. Record stats

---

## 8. Code Structure

The file is organized into logical sections. Here is the approximate line map for orientation:

| Lines | Section |
|-------|---------|
| `1–113` | Module definitions, genome layout, weight offsets, costs |
| `119–304` | Config class (all parameters, organized by system) |
| `310–520` | World init, organism fields, helper methods |
| `523–630` | Environment updates (toxic, decomp, viral, strata) |
| `638–760` | Sensing (22 channels) + energy acquisition |
| `764–928` | Predation system (flat spatial index, combat, defense) |
| `931–1146` | Toxic damage, detox, social system, mediator, nutrient cycling |
| `1149–1186` | Behavioral hijacking (NEW in Step 8) |
| `1189–1332` | Endosymbiosis (NEW in Step 9) |
| `1336–1476` | Movement (with hijack override integration) |
| `1479–1644` | Horizontal gene transfer + viral dynamics |
| `1645–1809` | Reproduction (with Wolbachia-style manipulation) |
| `1810–1952` | Death, decomposition, viral burst, strata deposit |
| `1955–2130` | Stats, role classification, run loop |

---

## 9. Visualizer

The companion visualizer (`visualizer_p3s9.py`, 659 lines) renders the simulation in real-time using Pygame. It provides 7 toggleable environment layers and **10 organism color modes**:

- **Role:** Green producers, purple herbivores, red carnivores, brown detritivores, blue omnivores.
- **Energy:** Red (low) to green (high) energy level.
- **Modules:** Blue (few) to magenta (many) module count.
- **Generation / Defense / Social / Manipulation:** Carried from Step 7.
- **Hijack (NEW):** Teal → yellow → red as hijack intensity increases. VRESIST organisms tinted blue-green.
- **Endosymbiosis (NEW):** Dim blue (ordinary) → cyan (merger-ready) → gold (1st-gen composite) → white-gold (deep composite). Sessile organisms dimmed.

The stats panel shows dedicated sections for endosymbiosis (total mergers, living composites, max lineage depth, milestone banners), hijacking (fraction hijacked, cumulative steps, behavioral override warnings), and all existing social/manipulation/environment metrics.
