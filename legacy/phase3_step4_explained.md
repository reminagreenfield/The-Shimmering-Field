# The Shimmering Field — Phase 3 Step 4: Plain-English Walkthrough

## What Is This?

This is an artificial life simulation. You have a 128×128 grid (like a petri dish) filled with tiny digital organisms that eat, move, reproduce, evolve, get infected by viruses, and die. Over thousands of timesteps, natural selection shapes the population — organisms that gather energy efficiently survive and reproduce, while poorly adapted ones die off.

The twist: organisms are built from **modules** (like biological Lego bricks), and they can gain or lose modules over generations. So the simulation doesn't just evolve *how well* an organism does something — it evolves *what an organism can do at all*.

---

## The Grid World

The world is a 128×128 grid. Every cell has several invisible layers stacked on top of each other, like transparent sheets:

| Layer | What it is | Analogy |
|---|---|---|
| **light** | Brightness. Bright at the top, dim at the bottom. Never changes. | Sunlight hitting the ocean — bright at the surface, dark in the deep |
| **toxic** | Poison concentration. Organisms produce it, it spreads and decays. | Industrial pollution in a pond |
| **nutrients** | Background food. Slowly regenerates everywhere. | Minerals in soil |
| **decomposition** | Dead organism remains. Deposited when things die. | Rotting organic matter on a forest floor |
| **decomp_scent** | Blurred version of decomposition. Used for navigation. | The *smell* of rotting food — spreads farther than the food itself |
| **density** | How many organisms are in each cell. Recounted every step. | Crowding |
| **zone_map** | Fixed multiplier for toxic production. Some zones amplify, some absorb. | Different soil types — some areas accumulate pollution faster |
| **viral_particles** | Floating virus concentration. Spreads and decays. | Airborne virus particles |
| **strata pools** | DNA fragments from dead organisms, layered by age (recent/intermediate/ancient) | Fossil layers in rock — recent on top, ancient deep down |

---

## Modules: What Organisms Can Do

Every organism has a set of **modules** — think of them as body parts or organs. Each module gives the organism a capability, but also costs energy to maintain.

### The 9 Module Types

**PHOTO** — Photosynthesis. Converts light into energy. Works best in bright, non-toxic areas. The organism's primary "plant" module.

**CHEMO** — Chemosynthesis. Converts *toxicity* into energy. The opposite of PHOTO — thrives in polluted areas. Like deep-sea bacteria living on volcanic vents.

**CONSUME** — Eating. Two sub-modes controlled by the same module:
- *Carnivore mode* (low decomp_preference): Hunts and kills other organisms
- *Detritivore mode* (high decomp_preference): Eats dead organism remains

**MOVE** — Locomotion. Without this, an organism is rooted in place like a plant. With it, the organism can move one cell per step in any direction.

**FORAGE** — Enhanced gathering. Gives a bonus to all energy production, lets the organism store more energy, and provides a small boost when near other foragers (cooperation).

**DEFENSE** — Protection against predators. Four sub-capabilities:
- *Shell*: Reduces the chance a predator kills you
- *Camouflage*: Chance the predator doesn't even see you
- *Counter-attack*: Damages the predator if it misses
- *Size investment*: Makes you tougher but costs extra energy

**DETOX** — Toxin metabolism. Actively removes toxin from the environment and converts some of it into energy. Also raises the toxicity level at which you start taking damage.

**TOXPROD** — Toxic waste production. Always present. Every organism pollutes. (No evolvable genes — it's just on or off.)

**VRESIST** — Viral resistance. Reserved for future use. Currently does nothing.

### Module Costs

Every module has two costs:
- **Maintenance**: Energy per step just for *having* it (even if unused)
- **Expression**: Extra energy per step when it's *active*

A simple organism (PHOTO + MOVE + TOXPROD) pays about 0.54 energy/step. An organism with 7 modules pays about 1.50/step. This means generalists need to earn much more energy just to break even.

---

## Ecological Roles

Roles aren't assigned — they **emerge** from which modules an organism has:

| Role | Modules | What it does |
|---|---|---|
| **Producer** | PHOTO and/or CHEMO, no CONSUME | Makes energy from the environment. A "plant." |
| **Carnivore** | CONSUME (low decomp_preference), no PHOTO/CHEMO | Hunts and kills other organisms. A "wolf." |
| **Detritivore** | CONSUME (high decomp_preference), no PHOTO/CHEMO | Eats dead remains. A "vulture." |
| **Omnivore** | PHOTO/CHEMO + CONSUME | Does both, but worse at each. A "bear." |

---

## The Genome: 36 Numbers

Each organism carries 36 floating-point numbers — its genome. These are NOT module on/off flags (those are separate boolean arrays). These are the *tuning knobs* within each module:

```
Positions  0–3:   PHOTO weights   (efficiency, toxic tolerance, light sensitivity, storage)
Positions  4–7:   CHEMO weights   (efficiency, specificity, saturation, gradient follow)
Positions  8–11:  CONSUME weights (prey selectivity, handling, decomp preference, aggression)
Positions 12–19:  MOVE weights    (8 weights controlling navigation preferences)
Positions 20–23:  FORAGE weights  (extraction, discrimination, storage cap, cooperation)
Positions 24–27:  DEFENSE weights (shell, camouflage, size, counter-attack)
Positions 28–31:  DETOX weights   (efficiency, tolerance, conversion, selectivity)
Positions 32–35:  Standalone      (transfer receptivity, selectivity, viral resistance, lyso suppression)
```

When a module is absent, its weights still exist in the genome but do nothing — they're "junk DNA." If the organism later gains that module (through mutation or horizontal transfer), those dormant weights suddenly become active.

---

## Major Variables (What the Code Tracks)

### Per-Organism Arrays

All organisms are stored in parallel arrays — organism #5's data is at index 5 in every array:

| Variable | Type | What it is |
|---|---|---|
| `rows`, `cols` | int | Grid position |
| `energy` | float | Current energy. Dies at 0. |
| `age` | int | Steps alive. Dies at 200. |
| `generation` | int | How many ancestors. Gen 0 = original, Gen 50 = 50th-generation descendant. |
| `ids` | int | Unique ID (never reused) |
| `parent_ids` | int | Parent's ID (-1 for originals) |
| `weights` | float[36] | The genome — 36 evolvable parameters |
| `module_present` | bool[9] | Which modules this organism has in its body plan |
| `module_active` | bool[9] | Which modules are currently turned on (currently always = present) |
| `transfer_count` | int | How many times this organism absorbed DNA fragments |
| `viral_load` | float | Lytic virus replication level. 0 = uninfected. Dies at 1.0. |
| `lysogenic_strength` | float | How much dormant virus is integrated |
| `lysogenic_genome` | float[36] | The dormant virus's genome |

### Environment Grids

| Variable | Shape | What it is |
|---|---|---|
| `light` | 128×128 | Static brightness map (bright top, dim bottom) |
| `toxic` | 128×128 | Current toxin concentration |
| `nutrients` | 128×128 | Background nutrient level |
| `decomposition` | 128×128 | Dead organism remains |
| `decomp_scent` | 128×128 | Gaussian-blurred decomp (for detritivore navigation) |
| `density` | 128×128 | Organism count per cell |
| `zone_map` | 128×128 | Fixed toxic production multiplier |
| `viral_particles` | 128×128 | Floating virus concentration |
| `viral_genome_pool` | 128×128×36 | Average genome carried by viruses at each cell |
| `strata_pool` | 3 × 128×128×36 | DNA fragments in three age layers |
| `strata_weight` | 3 × 128×128 | How much material in each layer |
| `strata_modules` | 3 × 128×128×9 | Module presence in each layer's fragments |

---

## Major Functions: What Happens Each Step

### `update()` — The Main Loop

This runs once per timestep. Here's the order:

```
1.  Count density          — How crowded is each cell?
2.  Sense                  — Each organism reads 15 sensor values
3.  Acquire energy         — PHOTO, CHEMO, scavenging, FORAGE bonuses
4.  Toxic damage           — Toxicity hurts (reduced by CHEMO and DETOX)
5.  Detox metabolism       — DETOX organisms clean toxins and gain energy
6.  Pay module costs       — Deduct maintenance + expression costs
7.  Crowding penalty       — Sessile producers penalized in dense cells
8.  Move                   — Mobile organisms choose direction and move
9.  Age                    — Everyone gets one step older
10. Predation              — CONSUME organisms hunt nearby prey
11. Horizontal transfer    — Absorb DNA from dead organisms (every 5 steps)
12. Viral dynamics         — Infections, lytic damage, lysogenic activation (every 3 steps)
13. Reproduce              — Organisms with enough energy make offspring
14. Death + decomposition  — Remove dead, deposit remains
15. Environment update     — Diffuse toxins, decay decomp, sediment strata
```

### `_sense_local()` — What Organisms Can See

Each organism reads 15 numbers from its surroundings:

- The **value** at its cell (light, toxicity, nutrients, density, decomposition)
- The **gradient** (difference between the cell above and below, left and right) for light, toxicity, density, decomp scent, and nutrients

Gradients tell the organism *which direction has more* of something. A positive light gradient means "it's brighter above me." A positive scent gradient means "there's more rotting food above me."

### `_acquire_energy()` — How Organisms Eat

Four energy pathways, each gated by a module:

**Photosynthesis** (PHOTO): `base × efficiency × toxic_penalty × light_factor × storage × shade_penalty`
- More light = more energy
- Toxicity reduces output (unless you have good toxic tolerance genes)
- Crowding reduces output (shade from neighbors)

**Chemosynthesis** (CHEMO): `base × efficiency × toxin_factor × shade_penalty`
- More toxicity = more energy (opposite of PHOTO!)
- Uses Michaelis-Menten kinetics (diminishing returns at high toxicity)

**Scavenging** (CONSUME): `base × decomp_preference × handling × available_decomp`
- Extracts energy from the decomposition grid
- Dedicated detritivores (high decomp_preference) get up to 2× bonus
- Actually removes decomp from the grid (food is consumed)

**Forage bonuses** (FORAGE):
- Extraction efficiency: up to +25% to ALL energy production
- Cooperative signal: nearby foragers boost each other

Omnivores (organisms with both production AND consumption) pay penalties:
- Hunt at 55% effectiveness (was 30% before tuning)
- Produce at 90% effectiveness

### `_predation()` — The Combat System

Every step, organisms with CONSUME can hunt nearby organisms.

**How a hunt works:**
1. Predator looks within 1 cell (sessile) or 2 cells (mobile)
2. Camouflaged prey have a chance to be invisible (up to 35%)
3. Predator picks a target (prefers weaker prey — low energy = easier catch)
4. Success probability = base 14% + aggression bonus + specialist bonus - shell defense
5. On hit: predator gains 55% of prey's energy, prey dies
6. On miss: prey with counter-attack damages the predator
7. After a kill: 70% chance predator is "full" and stops hunting

**Balance mechanisms:**
- Max 1 kill per cell per step (prevents local extinction)
- Predators are shuffled randomly (no systematic advantage)
- Specialists (pure consumers) get +15% hit chance and +50% energy gain
- A flat spatial index makes neighbor lookup fast

### `_apply_toxic_damage()` — Toxicity Hurts

Two-tier damage system:
- Below 0.8 effective toxicity: no damage
- Above 0.8: moderate damage (1.5 per unit above threshold)
- Above 1.5: heavy damage (5.0 per unit above threshold)

Two defenses:
- CHEMO tolerance: reduces *effective* toxicity (you feel less of it)
- DETOX tolerance: raises the *threshold* (you can handle more before it hurts)

### `_apply_detox()` — Eating Poison

DETOX organisms actively metabolize toxins:
1. Calculate how much toxin they can process (up to 8% of local concentration)
2. Convert some to energy (up to 40% efficiency)
3. Remove some from the grid (50% of what's metabolized — actually cleans the environment)

Uses Michaelis-Menten kinetics for "selective uptake" — each organism has a preferred toxin concentration where it works best.

### `_decide_movement()` — Where To Go

Each organism scores 5 options: stay, up, down, right, left.

Scores are built from weighted combinations of environmental gradients:
- **Photosynthesizers** follow light gradients upward, avoid toxicity
- **Chemosynthesizers** follow toxicity gradients (it's their food)
- **Carnivores** follow density gradients (hunt where prey clusters)
- **Detritivores** follow scent gradients (navigate toward decomp — weighted 8× stronger than other signals)
- **Foragers** follow nutrient gradients
- **Detoxers** follow toxicity gradients (also their food)

Exploration noise is added (controlled by a MOVE gene), and sessile organisms are forced to pick "stay."

### `_horizontal_transfer()` — Absorbing Dead DNA

Every 5 steps, organisms can absorb DNA fragments from dead organisms buried in the strata:

**Three strata (geological layers):**
- **Recent**: Always accessible. 10% blend rate. Fresh DNA from recent deaths.
- **Intermediate**: Accessible when local toxicity ≥ 0.3. 18% blend rate.
- **Ancient**: Accessible when local toxicity ≥ 0.8. 30% blend rate. Very different DNA.

Toxicity acts as a "key" — crisis conditions unlock ancient genetic material.

**Module acquisition**: 1.5% chance per transfer to gain a module that was common in the fragment pool. This means modules can spread horizontally through the population, not just through parent-child inheritance.

### `_viral_dynamics()` — Infection Cycle

Every 3 steps, the viral system runs:

**New infections**: If viral particles are present at an organism's cell, there's a chance of infection (reduced by viral resistance gene). Two outcomes:
- **Lytic** (60%): Virus actively replicates. Viral load grows 0.1/step, draining energy. At load 1.0, the organism *bursts* — dies and sprays virus particles in a 3-cell radius.
- **Lysogenic** (40%): Virus quietly integrates into the genome. No immediate damage. 80% inherited by offspring.

**Lysogenic activation**: Under toxic stress (toxicity > 0.6), dormant viruses can "wake up" — they blend into the organism's genome and switch to lytic mode. The lysogenic suppression gene reduces this probability.

### `_reproduce()` — Making Babies

Requirements: enough energy + old enough (8 steps) + not lytic-infected.

**Density-dependent threshold**: Base threshold is 80 energy, but +5 per organism in the same cell. At 3 neighbors, you need 95 energy. This prevents population explosions.

**Module evolution** in offspring:
- 1% chance to **gain** a random new module
- 0.3% chance to **lose** a module (but never your last energy source or TOXPROD)

**Offspring placement**:
- Mobile parents place babies up to 5 cells away
- Sessile parents place babies 1 cell away
- Detritivore babies seek the strongest decomp scent within 8 cells (born near food)

**Genome mutation**: Offspring weights = parent weights + small random noise (σ = 0.08).

**Lysogenic inheritance**: 80% of parent's dormant virus passed to child.

### `_kill_and_decompose()` — Death

Three causes of death: viral burst (load ≥ 1.0), starvation (energy ≤ 0), old age (age ≥ 200).

When an organism dies:
1. **Decomposition deposit**: Energy × 0.4 + 3.5 base units of decomp placed at death cell
2. **Strata deposit**: Genome weights AND module presence blended into the "recent" stratum
3. **Viral burst** (if lytic): 8.0 viral particles sprayed in 3-cell radius, carrying the dead organism's genome
4. **Spontaneous shedding**: 15% of natural deaths release viral particles (prevents virus extinction)

### `_update_environment()` — World Maintenance

Runs every step after all organism actions:
- **Toxicity**: Diffuses to neighbors (6%), decays (1%), produced by TOXPROD organisms
- **Nutrients**: Tiny trickle of regeneration, very slow conversion from decomp
- **Decomposition**: Decays 0.2%/step, diffuses slightly every 3 steps
- **Decomp scent**: Gaussian blur of decomposition (σ=5), recomputed every 3 steps
- **Strata sedimentation**: Recent → intermediate → ancient, with decay and diffusion
- **Viral particles**: Decay 1%/step, diffuse 8%/step

---

## The Tuning Changes (What We Modified)

The original parameters punished generalism so severely that detritivores and omnivores went extinct every run. Here's what was changed and why:

| Parameter | Original | Tuned | Effect |
|---|---|---|---|
| `initial_detritivore_fraction` | 6% (12) | 12% (24) | More detritivores survive the early food shortage |
| `decomp_death_deposit` | 2.0 | 3.5 | Each death produces 75% more food for scavengers |
| Initial decomp patch | 20×20, low | 30×30, high | Bigger starting food supply covers the bootstrapping gap |
| `producer_consume_penalty` | 0.30 | 0.55 | Omnivores hunt at 55% (survivable) instead of 30% (death sentence) |
| `consume_producer_penalty` | 0.85 | 0.90 | Omnivores produce at 90% — smaller tax for versatility |
| `module_gain_rate` | 0.5% | 1% | New ecological roles appear twice as often |

The key insight: in the original, omnivory wasn't a tradeoff — it was strictly worse than specialization. The tuning makes it a genuine tradeoff: omnivores are worse at each individual task, but can survive in conditions where a specialist would starve.
