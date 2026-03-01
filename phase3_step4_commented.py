"""
The Shimmering Field — Phase 3 Step 4: FORAGE + DEFENSE + DETOX
============================================================

This is a MAJOR architectural leap. The simulation transforms from organisms
with a flat genome (16 numbers) into organisms with MODULAR BODIES — discrete
functional modules that can be gained, lost, and evolved independently.

═══════════════════════════════════════════════════════════════════════════════
THE BIG PICTURE: MODULAR ORGANISMS
═══════════════════════════════════════════════════════════════════════════════

In earlier phases, every organism was basically the same: it could photosynthesize,
move, and get infected by viruses. The only differences were in the gene values
(how WELL it did those things).

Now, organisms are built from MODULES — like biological Lego bricks:

  ┌──────────────────────────────────────────────────────────────────┐
  │  PHOTO    — Photosynthesis (convert light → energy)             │
  │  CHEMO    — Chemosynthesis (convert toxins → energy)            │
  │  CONSUME  — Predation/scavenging (eat other organisms or corpses)│
  │  MOVE     — Locomotion (ability to move across the grid)        │
  │  FORAGE   — Enhanced resource gathering + cooperation    (NEW)  │
  │  DEFENSE  — Shell/camouflage/counter-attack              (NEW)  │
  │  DETOX    — Active toxin metabolism                      (NEW)  │
  │  TOXPROD  — Toxic waste production (always on)                  │
  │  VRESIST  — Viral resistance (reserved for future)              │
  └──────────────────────────────────────────────────────────────────┘

Each module:
  - Can be PRESENT or ABSENT in an organism's body plan
  - Has its own EVOLVABLE WEIGHTS (genes specific to that module)
  - Costs energy to MAINTAIN (just having it) and EXPRESS (actively using it)
  - Can be GAINED through mutation during reproduction (0.5% chance)
  - Can be LOST through mutation (0.3% chance, with safety checks)
  - Can be TRANSFERRED via horizontal gene transfer from dead organisms

This creates ECOLOGICAL ROLES that emerge from module combinations:

  Producer:    PHOTO and/or CHEMO, no CONSUME
               → Plants. Make energy from environment.

  Carnivore:   CONSUME, no PHOTO/CHEMO, low decomp_preference
               → Predators. Hunt and kill other organisms.

  Detritivore: CONSUME, no PHOTO/CHEMO, high decomp_preference
               → Scavengers. Eat dead organism remains.

  Omnivore:    PHOTO/CHEMO + CONSUME
               → Jacks of all trades, masters of none (penalties apply).

═══════════════════════════════════════════════════════════════════════════════
NEW MODULES IN THIS STEP
═══════════════════════════════════════════════════════════════════════════════

FORAGE MODULE (4 weights):
  Enhances resource gathering and enables proto-cooperation.
  - extraction_eff:     Bonus to all energy production (+25% max)
  - resource_discrim:   Navigate toward nutrient-rich areas
  - storage_cap:        Extra energy storage capacity (+30 max energy)
  - cooperative_signal: Nearby foragers boost each other's energy gain

DEFENSE MODULE (4 weights):
  Protection against predation. Creates predator-prey arms race.
  - shell:         Reduces predator's success probability (up to -55%)
  - camouflage:    Chance predator doesn't even notice you (up to 35%)
  - size_invest:   Makes you harder to kill but costs extra energy
  - counter_attack: Damages predator on a failed hunt (up to 5 energy)

DETOX MODULE (4 weights):
  Active toxin metabolism — turns pollution into an energy source.
  - detox_eff:        Rate of toxin removal from environment
  - toxin_tolerance:  Raises the damage threshold (can survive higher toxicity)
  - conversion_rate:  Fraction of metabolized toxin converted to energy
  - selective_uptake: Michaelis-Menten style preference for toxin concentrations

Key interactions:
  - DEFENSE vs CONSUME: arms race — predators need more aggression to overcome defense
  - DETOX + CHEMO: synergy in toxic environments (CHEMO gains energy, DETOX removes damage)
  - DETOX + PHOTO: opens toxic zones to photosynthesizers (they can now survive there)
  - FORAGE + producers: better resource extraction, cooperative nutrient boost
  - DEFENSE cost: shell/size maintenance makes defended organisms slower to reproduce

Built on Phase 3 Step 3 (CONSUME module).
"""

# ─── IMPORTS ───────────────────────────────────────────────────────────────────
import numpy as np
import json
import os
import time
from scipy.ndimage import uniform_filter, gaussian_filter


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
# These constants define the modular organism architecture.
# Every organism has a boolean array (module_present) indicating which modules
# it has, and a float array (weights) containing all evolvable parameters.

# ── Module IDs ─────────────────────────────────────────────────────────────────
# Integer constants for indexing into module arrays.
# Example: self.module_present[organism_i, M_PHOTO] → True/False

M_PHOTO   = 0    # Photosynthesis: convert light into energy
M_CHEMO   = 1    # Chemosynthesis: convert toxic chemicals into energy
M_CONSUME = 2    # Consumption: hunt prey or scavenge decomposing matter
M_MOVE    = 3    # Movement: ability to change grid position
M_FORAGE  = 4    # Foraging: enhanced resource extraction + cooperation (NEW)
M_DEFENSE = 5    # Defense: protection against predation (NEW)
M_DETOX   = 6    # Detoxification: metabolize environmental toxins (NEW)
M_TOXPROD = 7    # Toxic production: produce waste (always present, no evolvable weights)
M_VRESIST = 8    # Viral resistance: (reserved for future implementation)

N_MODULES = 9    # Total number of module types

MODULE_NAMES = [
    "PHOTO", "CHEMO", "CONSUME", "MOVE", "FORAGE",
    "DEFENSE", "DETOX", "TOXPROD", "VRESIST"
]

# ── Module Weight Sizes ────────────────────────────────────────────────────────
# How many evolvable parameters (weights) each module has.
# These weights are the "genes" that control how well each module functions.
# Some modules (TOXPROD, VRESIST) have 0 weights — they're on/off only.

MODULE_WEIGHT_SIZES = np.array([
    4,  # PHOTO:   [efficiency, toxic_tolerance, light_sensitivity, storage_rate]
    4,  # CHEMO:   [efficiency, specificity, saturation_threshold, gradient_follow]
    4,  # CONSUME: [prey_selectivity, handling_efficiency, decomp_preference, aggression]
    8,  # MOVE:    [light_seek, density_avoid, nutrient_stay, chemo_toxic_seek,
        #           random_wt, stay_tend, light_str, toxic_response]
    4,  # FORAGE:  [extraction_eff, resource_discrim, storage_cap, cooperative_signal]
    4,  # DEFENSE: [shell, camouflage, size_invest, counter_attack]
    4,  # DETOX:   [detox_eff, toxin_tolerance, conversion_rate, selective_uptake]
    0,  # TOXPROD: (no evolvable weights — either you pollute or you don't)
    0,  # VRESIST: (future — no weights yet)
], dtype=np.int32)

# ── Weight Offsets ─────────────────────────────────────────────────────────────
# Each module's weights are stored in a FLAT array (one per organism).
# These offsets tell us WHERE in that flat array each module's weights begin.
#
# Layout of the weight array for one organism (36 total):
#   [0:4]   → PHOTO weights
#   [4:8]   → CHEMO weights
#   [8:12]  → CONSUME weights
#   [12:20] → MOVE weights (8 weights — the biggest module)
#   [20:24] → FORAGE weights
#   [24:28] → DEFENSE weights
#   [28:32] → DETOX weights
#   [32:36] → Standalone params (transfer + viral, not tied to any module)

MODULE_WEIGHT_OFFSETS = np.zeros(N_MODULES, dtype=np.int32)
_off = 0
for _m in range(N_MODULES):
    MODULE_WEIGHT_OFFSETS[_m] = _off
    _off += MODULE_WEIGHT_SIZES[_m]
TOTAL_MODULE_WEIGHTS = _off  # 32 — total weights used by all modules

# ── Standalone Parameters ──────────────────────────────────────────────────────
# 4 extra parameters that aren't tied to any module — they're always available.
# These control horizontal gene transfer and viral resistance (from Phase 2).

N_STANDALONE_PARAMS = 4
STANDALONE_OFFSET = TOTAL_MODULE_WEIGHTS  # = 32 (starts after module weights)
TOTAL_WEIGHT_PARAMS = TOTAL_MODULE_WEIGHTS + N_STANDALONE_PARAMS  # = 36

# Indices into the standalone params section (relative to STANDALONE_OFFSET)
SP_TRANSFER_RECEPTIVITY = 0   # Willingness to absorb DNA fragments
SP_TRANSFER_SELECTIVITY = 1   # How similar fragments must be
SP_VIRAL_RESISTANCE = 2       # Reduces viral infection probability
SP_LYSO_SUPPRESSION = 3       # Reduces lysogenic activation probability

# ── Module Costs ───────────────────────────────────────────────────────────────
# Every module has TWO costs:
#   MAINTENANCE: Energy cost per step just for HAVING the module (even if dormant)
#   EXPRESSION:  Additional cost per step when the module is ACTIVELY USED
#
# This creates a fundamental tradeoff: more modules = more capabilities but
# higher energy drain. A 6-module organism needs much more energy to survive
# than a 3-module specialist.

#                                      PH    CH    CO    MV    FO    DE    DT    TP    VR
MODULE_MAINTENANCE = np.array([       0.20, 0.25, 0.12, 0.08, 0.12, 0.18, 0.20, 0.03, 0.15])
MODULE_EXPRESSION  = np.array([       0.10, 0.12, 0.08, 0.06, 0.06, 0.10, 0.10, 0.02, 0.08])
BASE_MAINTENANCE = 0.05  # Every organism pays this regardless of modules

# Cost examples for common builds:
#   PHOTO + MOVE + TOXPROD                   = 0.05 + 0.30 + 0.14 + 0.20 + 0.05 = 0.54/step
#   + FORAGE (efficient producer)            = 0.72/step
#   + DEFENSE (armored producer)             = 1.00/step
#   + CONSUME (jack-of-all-trades)           = 1.20/step
#   + DETOX (toxic zone specialist)          = 1.50/step

# ── Gainable Modules ──────────────────────────────────────────────────────────
# Which modules can be gained through mutation or horizontal transfer.
# PHOTO and TOXPROD are always present from birth (not in this list).
# VRESIST is reserved for future use.
GAINABLE_MODULES = [M_CHEMO, M_CONSUME, M_MOVE, M_FORAGE, M_DEFENSE, M_DETOX]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Grid & Light (unchanged) ───────────────────────────────────────────
    grid_size = 128
    light_max = 1.0
    light_min = 0.05
    zone_count = 8

    # ── Toxicity (unchanged) ───────────────────────────────────────────────
    toxic_decay_rate = 0.01
    toxic_diffusion_rate = 0.06
    toxic_production_rate = 0.015
    toxic_threshold_low = 0.3
    toxic_threshold_medium = 0.8
    toxic_threshold_high = 1.5
    toxic_damage_medium = 1.5
    toxic_damage_high = 5.0
    toxic_photo_penalty = 1.0

    # ── Nutrients (unchanged) ──────────────────────────────────────────────
    nutrient_base_rate = 0.002
    nutrient_from_decomp = 0.4
    nutrient_max = 3.0

    # ── Population ─────────────────────────────────────────────────────────
    initial_population = 200          # Bigger starting pop (was 80 in Phase 2)
    initial_consumer_fraction = 0.10  # 10% start as carnivores (20 organisms)
    initial_detritivore_fraction = 0.06  # 6% start as detritivores (12 organisms)
    energy_initial = 40.0
    energy_max = 150.0
    energy_reproduction_threshold = 80.0
    energy_reproduction_cost = 40.0
    energy_movement_cost = 0.2
    max_age = 200
    min_reproduction_age = 8
    offspring_distance = 5

    # Density-dependent reproduction: it's harder to reproduce in crowds.
    # Effective threshold = base 80 + (neighbors × 5).
    # At 3 neighbors: need 95 energy. At 6 neighbors: need 110 energy.
    # This prevents population explosions in dense clusters.
    repro_density_penalty = 5.0

    # ── Energy Production ──────────────────────────────────────────────────
    photosynthesis_base = 3.0     # Base energy from photosynthesis
    chemosynthesis_base = 2.2     # Base energy from chemosynthesis (less than photo)
    scavenge_base = 2.5           # Base energy from scavenging decomposition

    # ── Metabolic Interference ─────────────────────────────────────────────
    # Omnivores (organisms with BOTH production AND consumption modules) pay
    # penalties — you can't be great at everything.
    #
    # Think of it like: a wolf that also photosynthesizes would be a bad wolf
    # AND a bad plant. Specialists outperform generalists.
    producer_consume_penalty = 0.30  # Omnivores hunt at only 30% effectiveness
    consume_producer_penalty = 0.85  # Omnivores produce at 85% effectiveness
    consumer_specialist_bonus = 0.5  # Pure consumers (no production) get +50% hunting

    # ── Predation ──────────────────────────────────────────────────────────
    predation_check_interval = 1      # Check every step (predation is constant pressure)
    predation_base_success = 0.14     # 14% base chance of a successful kill
    predation_energy_fraction = 0.55  # Predator gains 55% of prey's energy on kill
    predation_hunt_radius_base = 1    # Sessile predators: 1-cell radius (adjacent only)
    predation_hunt_radius_mobile = 2  # Mobile predators: 2-cell radius (can chase)
    predation_max_kills_per_cell = 1  # Max 1 kill per cell per step (prevents massacres)
    predator_satiation = 0.7          # 70% chance a fed predator skips the next hunt

    # ── Decomposition ─────────────────────────────────────────────────────
    decomp_death_deposit = 2.0        # Base decomp deposited when any organism dies
    decomp_decay_rate = 0.998         # Decomp decays 0.2% per step (slow — food persists)
    decomp_diffusion_rate = 0.008     # Decomp spreads slightly to neighboring cells
    decomp_diffusion_interval = 3     # Diffuse every 3 steps
    decomp_scent_sigma = 5            # Gaussian blur radius for "scent" navigation layer
    decomp_scent_interval = 3         # Recompute scent layer every 3 steps

    # ── FORAGE Module Parameters (NEW) ─────────────────────────────────────
    forage_extraction_bonus = 0.25    # Max +25% bonus to all resource extraction
    forage_storage_bonus = 30.0       # Max +30 extra energy capacity (150 → 180)
    forage_cooperative_radius = 1     # Check 1-cell radius for cooperating foragers
    forage_cooperative_bonus = 0.08   # +0.08 energy per nearby forager per step

    # ── DEFENSE Module Parameters (NEW) ────────────────────────────────────
    defense_shell_max = 0.55          # Shell can reduce hunt success by up to 55%
    defense_camouflage_max = 0.35     # Up to 35% chance predator misses you entirely
    defense_counter_damage = 5.0      # Counter-attack deals up to 5 energy damage on miss
    defense_size_cost_mult = 1.3      # Size investment increases DEFENSE module cost by 30%

    # ── DETOX Module Parameters (NEW) ──────────────────────────────────────
    detox_rate_max = 0.08             # Max 8% of local toxin removed per step
    detox_tolerance_bonus = 0.6       # Raises toxic damage threshold by up to 0.6
                                      # (medium threshold: 0.8 → 1.4 for a maxed detoxer)
    detox_energy_conversion = 0.4     # 40% of metabolized toxin becomes energy
    detox_environment_effect = 0.5    # 50% of detox actually cleans the grid cell

    # ── Evolution ──────────────────────────────────────────────────────────
    mutation_rate = 0.08              # Random noise on weights during reproduction
    module_gain_rate = 0.005          # 0.5% chance to gain a new module per birth
    module_lose_rate = 0.003          # 0.3% chance to lose a module per birth

    # ── Horizontal Gene Transfer (unchanged from Phase 2 Step 4) ───────────
    transfer_check_interval = 5
    transfer_blend_rate_recent = 0.10
    transfer_blend_rate_intermediate = 0.18
    transfer_blend_rate_ancient = 0.30
    decomp_fragment_decay = 0.005
    decomp_fragment_diffusion = 0.008
    module_transfer_rate = 0.015      # Chance of acquiring a module via HGT
    sedimentation_rate_recent = 0.005
    sedimentation_rate_intermediate = 0.002
    ancient_decay_rate = 0.001
    stratum_access_medium = 0.3
    stratum_access_high = 0.8

    # ── Viral System (unchanged from Phase 2 Step 4) ──────────────────────
    viral_decay_rate = 0.01
    viral_diffusion_rate = 0.08
    viral_infection_rate = 0.3
    viral_lytic_damage = 2.0
    viral_lytic_growth = 0.1
    viral_burst_threshold = 1.0
    viral_burst_amount = 8.0
    viral_burst_radius = 3
    lysogenic_probability = 0.4
    lysogenic_activation_toxic = 0.6
    lysogenic_blend_rate = 0.1
    lysogenic_inheritance = 0.8
    viral_check_interval = 3

    # ── Simulation ─────────────────────────────────────────────────────────
    total_timesteps = 10000
    snapshot_interval = 100
    output_dir = "output_p3s4"
    random_seed = 42


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD
# ═══════════════════════════════════════════════════════════════════════════════

class World:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.rng = np.random.default_rng(c.random_seed)
        self.timestep = 0
        self.next_id = 0
        N = c.grid_size

        # ══════════════════════════════════════════════════════════════════════
        # ENVIRONMENT GRIDS (mostly unchanged)
        # ══════════════════════════════════════════════════════════════════════
        self.light = np.linspace(c.light_max, c.light_min, N)[:, None] * np.ones((1, N))
        self.toxic = self.rng.uniform(0.0, 0.005, (N, N))
        self.nutrients = self.rng.uniform(0.02, 0.08, (N, N))
        self.decomposition = np.zeros((N, N))
        self.density = np.zeros((N, N), dtype=np.int32)

        # NEW: Decomposition "scent" layer — a blurred version of decomposition
        # that detritivores use for navigation. They follow the scent gradient
        # toward concentrations of dead matter, like vultures following the
        # smell of a carcass. Recomputed every few steps via gaussian blur.
        self.decomp_scent = np.zeros((N, N))

        # Zone map (unchanged — random toxicity multipliers per zone)
        zone_size = N // c.zone_count
        self.zone_map = np.ones((N, N))
        for zi in range(c.zone_count):
            for zj in range(c.zone_count):
                r0, r1 = zi * zone_size, (zi + 1) * zone_size
                c0, c1 = zj * zone_size, (zj + 1) * zone_size
                self.zone_map[r0:r1, c0:c1] = self.rng.choice(
                    [0.3, 0.5, 0.7, 1.0, 1.0, 1.2, 1.5, 2.0])
        self.zone_map = uniform_filter(self.zone_map, size=8)

        # ══════════════════════════════════════════════════════════════════════
        # STRATIFIED FRAGMENT POOLS (now stores MODULE info too)
        # ══════════════════════════════════════════════════════════════════════
        # Each stratum now has THREE layers:
        #   strata_pool:    Average WEIGHTS of dead organisms (36 params)
        #   strata_weight:  How much material is at each cell
        #   strata_modules: Average MODULE PRESENCE of dead organisms (9 modules)
        #                   This enables module transfer via HGT — living organisms
        #                   can acquire modules from dead organisms' DNA fragments!

        self.strata_pool = {s: np.zeros((N, N, TOTAL_WEIGHT_PARAMS))
                           for s in ("recent", "intermediate", "ancient")}
        self.strata_weight = {s: np.zeros((N, N))
                             for s in ("recent", "intermediate", "ancient")}
        self.strata_modules = {s: np.zeros((N, N, N_MODULES))
                              for s in ("recent", "intermediate", "ancient")}

        # ── Viral System (unchanged) ─────────────────────────────────────────
        self.viral_particles = np.zeros((N, N))
        for _ in range(8):
            sr, sc_ = self.rng.integers(0, N), self.rng.integers(0, N)
            r0, r1 = max(0, sr - 5), min(N, sr + 6)
            c0_, c1_ = max(0, sc_ - 5), min(N, sc_ + 6)
            self.viral_particles[r0:r1, c0_:c1_] += self.rng.uniform(0.5, 2.0)
        self.viral_genome_pool = np.zeros((N, N, TOTAL_WEIGHT_PARAMS))
        self.viral_genome_weight = np.zeros((N, N))
        seed_mask = self.viral_particles > 0.1
        self.viral_genome_pool[seed_mask] = self.rng.normal(
            0, 0.5, (int(seed_mask.sum()), TOTAL_WEIGHT_PARAMS))
        self.viral_genome_weight[seed_mask] = 1.0

        # ── Initialize Organisms ─────────────────────────────────────────────
        self._init_organisms()

        # ── Global Counters ──────────────────────────────────────────────────
        self.total_transfers = 0
        self.transfers_by_stratum = {"recent": 0, "intermediate": 0, "ancient": 0}
        self.total_lytic_deaths = 0
        self.total_lysogenic_integrations = 0
        self.total_lysogenic_activations = 0
        self.total_predation_kills = 0  # NEW: tracks total predation kills
        self.stats_history = []

    # ══════════════════════════════════════════════════════════════════════════
    # INITIAL POPULATION SEEDING
    # ══════════════════════════════════════════════════════════════════════════

    def _init_organisms(self):
        """
        Seed the initial population with three ecological roles:
        
        1. PRODUCERS (majority, ~84%): PHOTO + MOVE + TOXPROD
           Standard photosynthesizers. Spread randomly across the grid.
        
        2. CARNIVORES (~10%): CONSUME + MOVE + TOXPROD (no PHOTO)
           Predators that hunt other organisms. Seeded in the center of the
           grid, close together so they can find prey immediately.
           Given high aggression genes and extra starting energy.
        
        3. DETRITIVORES (~6%): CONSUME + MOVE + TOXPROD (no PHOTO)
           Scavengers that eat decomposing matter. Also seeded near center.
           Given high decomp_preference genes. The center also gets a patch
           of starting decomposition so they have food right away.
        """
        c = self.cfg
        N = c.grid_size
        pop = c.initial_population  # 200
        mid = N // 2                # 64 (center of grid)

        # Basic arrays
        self.rows = self.rng.integers(0, N, size=pop).astype(np.int64)
        self.cols = self.rng.integers(0, N, size=pop).astype(np.int64)
        self.energy = np.full(pop, c.energy_initial)
        self.age = np.zeros(pop, dtype=np.int32)
        self.generation = np.zeros(pop, dtype=np.int32)
        self.ids = np.arange(pop, dtype=np.int64)
        self.parent_ids = np.full(pop, -1, dtype=np.int64)
        self.next_id = pop

        # ── Default module loadout: PHOTO + MOVE + TOXPROD ───────────────────
        # module_present: which modules exist in the organism's body plan
        # module_active: which modules are currently turned on (starts = present)
        self.module_present = np.zeros((pop, N_MODULES), dtype=bool)
        self.module_present[:, M_PHOTO] = True     # Everyone photosynthesizes
        self.module_present[:, M_MOVE] = True       # Everyone can move
        self.module_present[:, M_TOXPROD] = True    # Everyone produces waste
        self.module_active = self.module_present.copy()

        # Weight array: 36 random parameters per organism
        self.weights = self.rng.normal(0, 0.5, (pop, TOTAL_WEIGHT_PARAMS))

        # ── Seed Carnivores ──────────────────────────────────────────────────
        # Remove PHOTO, add CONSUME. Place near center. Give high aggression.
        n_carn = int(pop * c.initial_consumer_fraction)  # 20
        carn_idx = self.rng.choice(pop, n_carn, replace=False)
        self.module_present[carn_idx, M_PHOTO] = False
        self.module_present[carn_idx, M_CONSUME] = True
        self.module_active[carn_idx] = self.module_present[carn_idx]
        # Place near center so they're close to prey
        self.rows[carn_idx] = self.rng.integers(mid - 20, mid + 20, n_carn).astype(np.int64)
        self.cols[carn_idx] = self.rng.integers(mid - 20, mid + 20, n_carn).astype(np.int64)
        self.energy[carn_idx] = c.energy_initial * 1.5  # Extra starting energy (60)
        # Set CONSUME weights: high aggression (gene 3), low decomp preference (gene 2)
        consume_off = int(MODULE_WEIGHT_OFFSETS[M_CONSUME])
        self.weights[carn_idx, consume_off + 3] = self.rng.uniform(1.0, 2.5, n_carn)    # aggression
        self.weights[carn_idx, consume_off + 2] = self.rng.uniform(-3.0, -1.5, n_carn)  # decomp_pref (negative = prefer live prey)

        # ── Seed Detritivores ────────────────────────────────────────────────
        # Same as carnivores but with opposite CONSUME gene emphasis.
        remaining = np.setdiff1d(np.arange(pop), carn_idx)
        n_detr = int(pop * c.initial_detritivore_fraction)  # 12
        detr_idx = self.rng.choice(remaining, n_detr, replace=False)
        self.module_present[detr_idx, M_PHOTO] = False
        self.module_present[detr_idx, M_CONSUME] = True
        self.module_active[detr_idx] = self.module_present[detr_idx]
        self.rows[detr_idx] = self.rng.integers(mid - 15, mid + 15, n_detr).astype(np.int64)
        self.cols[detr_idx] = self.rng.integers(mid - 15, mid + 15, n_detr).astype(np.int64)
        self.energy[detr_idx] = c.energy_initial * 1.5
        self.weights[detr_idx, consume_off + 2] = self.rng.uniform(1.5, 3.0, n_detr)    # HIGH decomp preference
        self.weights[detr_idx, consume_off + 3] = self.rng.uniform(-2.0, -0.5, n_detr)  # LOW aggression

        # ── Seed Starting Decomposition ──────────────────────────────────────
        # Place some dead matter near center so detritivores have initial food.
        self.decomposition[mid-10:mid+10, mid-10:mid+10] += self.rng.uniform(0.5, 2.0, (20, 20))

        # ── Viral/HGT State ──────────────────────────────────────────────────
        self.transfer_count = np.zeros(pop, dtype=np.int32)
        self.viral_load = np.zeros(pop, dtype=np.float64)
        self.lysogenic_strength = np.zeros(pop, dtype=np.float64)
        self.lysogenic_genome = np.zeros((pop, TOTAL_WEIGHT_PARAMS), dtype=np.float64)

    @property
    def pop(self):
        """Current living population count."""
        return len(self.rows)

    # ══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def _module_weights(self, module_id):
        """
        Extract the evolvable weights for a specific module across all organisms.
        Returns: (pop, weight_size) array, or None if module has 0 weights.
        
        Example: _module_weights(M_PHOTO) returns shape (pop, 4) — the 4
        photosynthesis genes for every organism.
        """
        off = int(MODULE_WEIGHT_OFFSETS[module_id])
        sz = int(MODULE_WEIGHT_SIZES[module_id])
        return self.weights[:, off:off+sz] if sz > 0 else None

    def _standalone_params(self):
        """
        Extract the 4 standalone parameters (transfer + viral defense)
        that aren't tied to any module. Returns: (pop, 4) array.
        """
        return self.weights[:, STANDALONE_OFFSET:]

    def _compute_module_costs(self):
        """
        Calculate total energy cost per step for each organism based on
        which modules they HAVE (maintenance) and which are ACTIVE (expression).
        
        Cost = BASE (0.05) + sum of maintenance for present modules
                            + sum of expression for active modules
        
        An organism with PHOTO+MOVE+TOXPROD pays ~0.54/step.
        An organism with all 7 modules pays ~1.50/step.
        """
        maint = self.module_present.astype(np.float64) @ MODULE_MAINTENANCE
        expr = self.module_active.astype(np.float64) @ MODULE_EXPRESSION
        return BASE_MAINTENANCE + maint + expr

    def _effective_energy_max(self):
        """
        FORAGE storage_cap: Organisms with FORAGE can store extra energy
        beyond the base 150 cap. The storage_cap weight (gene 2 of FORAGE)
        determines how much extra, up to +30.
        
        Returns: Array of effective energy maximums per organism.
        """
        c = self.cfg
        effective_max = np.full(self.pop, c.energy_max)  # Base: 150 for everyone
        has_forage = self.module_active[:, M_FORAGE]
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            # Sigmoid maps gene to 0-1, multiply by bonus (30)
            storage = c.forage_storage_bonus * (1.0 / (1.0 + np.exp(-fw[:, 2])))
            effective_max += storage * has_forage  # Only foragers get the bonus
        return effective_max

    def _compute_total_costs(self):
        """
        Total energy cost = module costs + DEFENSE size surcharge.
        
        The DEFENSE module's "size_invest" weight can increase the organism's
        size, making it harder to kill but more expensive to maintain.
        The surcharge is proportional to size_invest × 30% of DEFENSE cost.
        """
        c = self.cfg
        costs = self._compute_module_costs()
        has_defense = self.module_active[:, M_DEFENSE]
        if has_defense.any():
            dw = self._module_weights(M_DEFENSE)
            size_invest = np.maximum(0, np.tanh(dw[:, 2])) * has_defense  # 0-1 range
            costs += size_invest * (c.defense_size_cost_mult - 1.0) * (
                MODULE_MAINTENANCE[M_DEFENSE] + MODULE_EXPRESSION[M_DEFENSE])
        return costs

    # ══════════════════════════════════════════════════════════════════════════
    # ORGANISM ARRAY MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════
    # Simplified from Phase 2's registry system — now a flat list of field names.
    # The _append_organisms method requires ALL fields to be provided explicitly
    # (no defaults). This is simpler and less error-prone with many fields.

    _ORG_FIELDS = [
        "rows", "cols", "energy", "age", "generation", "ids", "parent_ids",
        "weights", "module_present", "module_active", "transfer_count",
        "viral_load", "lysogenic_strength", "lysogenic_genome",
    ]

    def _filter_organisms(self, mask):
        """Keep only organisms where mask is True."""
        for name in self._ORG_FIELDS:
            setattr(self, name, getattr(self, name)[mask])

    def _append_organisms(self, d):
        """Append new organisms. d must contain ALL fields in _ORG_FIELDS."""
        for name in self._ORG_FIELDS:
            setattr(self, name, np.concatenate([getattr(self, name), d[name]]))

    # ══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT UPDATE
    # ══════════════════════════════════════════════════════════════════════════

    def _update_environment(self):
        """
        Update all environment systems. Same as before PLUS:
          - Only organisms with TOXPROD module produce toxic waste
          - Decomposition now has its own diffusion and scent layer
          - Nutrient-from-decomp conversion is slower (0.1% vs 1.5% in Phase 2)
            because decomposition is now a direct food source for detritivores
        """
        c = self.cfg

        # ── Toxic diffusion + decay (unchanged) ─────────────────────────────
        p = np.pad(self.toxic, 1, mode='edge')
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        self.toxic += c.toxic_diffusion_rate * (nb - self.toxic)
        self.toxic *= (1.0 - c.toxic_decay_rate / np.maximum(0.3, self.zone_map))

        # ── Toxic production (now module-gated) ──────────────────────────────
        # Only organisms with TOXPROD active produce waste.
        # In practice, TOXPROD is always present, but the architecture supports
        # organisms that don't pollute (if TOXPROD were ever lost).
        if self.pop > 0:
            has_toxprod = self.module_active[:, M_TOXPROD]
            rates = np.where(has_toxprod, c.toxic_production_rate, 0.0)
            np.add.at(self.toxic, (self.rows, self.cols),
                      rates * self.zone_map[self.rows, self.cols])

        # ── Nutrients ────────────────────────────────────────────────────────
        self.nutrients += c.nutrient_base_rate  # Tiny trickle
        # Very slow decomp → nutrient conversion (0.1%, was 1.5% in Phase 2)
        # Decomp is now primarily food for detritivores, not a nutrient source
        xfer = self.decomposition * 0.001
        self.nutrients += xfer
        self.decomposition -= xfer

        # ── Decomposition decay + diffusion ──────────────────────────────────
        self.decomposition *= c.decomp_decay_rate  # 0.2% decay per step

        # Local diffusion: dead matter spreads slightly (creates a feeding zone)
        if self.timestep % c.decomp_diffusion_interval == 0:
            dp = np.pad(self.decomposition, 1, mode='edge')
            dnb = (dp[:-2, 1:-1] + dp[2:, 1:-1] + dp[1:-1, :-2] + dp[1:-1, 2:]) / 4.0
            self.decomposition += c.decomp_diffusion_rate * (dnb - self.decomposition)

        # ── Decomp scent layer (NEW) ─────────────────────────────────────────
        # Gaussian-blurred version of decomposition. Detritivores use the
        # GRADIENT of this layer to navigate toward food (like smelling carrion).
        # sigma=5 means the scent spreads about 15 cells from the source.
        if self.timestep % c.decomp_scent_interval == 0:
            self.decomp_scent = gaussian_filter(
                self.decomposition, sigma=c.decomp_scent_sigma, mode='constant')

        # ── Strata sedimentation + viral environment (unchanged) ─────────────
        self._update_strata()
        self._update_viral_environment()

        # ── Clamp ────────────────────────────────────────────────────────────
        np.clip(self.toxic, 0, 5.0, out=self.toxic)
        np.clip(self.nutrients, 0, c.nutrient_max, out=self.nutrients)
        np.clip(self.decomposition, 0, 30.0, out=self.decomposition)  # Higher cap (food source)
        np.clip(self.viral_particles, 0, 20.0, out=self.viral_particles)

    def _update_strata(self):
        """
        Sedimentation: recent → intermediate → ancient. Same as Phase 2 Step 4,
        but now also sediments MODULE PRESENCE alongside genome weights.
        """
        c = self.cfg

        # Recent → Intermediate
        xfer_w = self.strata_weight["recent"] * c.sedimentation_rate_recent
        self.strata_weight["recent"] -= xfer_w
        new_iw = self.strata_weight["intermediate"] + xfer_w
        blend = xfer_w / np.maximum(new_iw, 1e-8)
        # Blend both weight pools AND module pools
        for key in ("strata_pool", "strata_modules"):
            pool = getattr(self, key)
            pool["intermediate"] = (pool["intermediate"] * (1.0 - blend[:, :, None])
                                   + pool["recent"] * blend[:, :, None])
        self.strata_weight["intermediate"] = new_iw

        # Intermediate → Ancient (same pattern)
        xfer_w2 = self.strata_weight["intermediate"] * c.sedimentation_rate_intermediate
        self.strata_weight["intermediate"] -= xfer_w2
        new_aw = self.strata_weight["ancient"] + xfer_w2
        blend2 = xfer_w2 / np.maximum(new_aw, 1e-8)
        for key in ("strata_pool", "strata_modules"):
            pool = getattr(self, key)
            pool["ancient"] = (pool["ancient"] * (1.0 - blend2[:, :, None])
                              + pool["intermediate"] * blend2[:, :, None])
        self.strata_weight["ancient"] = new_aw

        # Decay (each stratum at different rates)
        self.strata_weight["recent"] *= (1.0 - c.decomp_fragment_decay)
        self.strata_weight["intermediate"] *= (1.0 - c.decomp_fragment_decay * 0.5)
        self.strata_weight["ancient"] *= (1.0 - c.ancient_decay_rate)

        # Fragment diffusion (every 10 steps)
        if self.timestep % 10 == 0:
            k = c.decomp_fragment_diffusion * 10
            for sname in ("recent", "intermediate", "ancient"):
                sw = self.strata_weight[sname]
                if sw.max() < 0.001:
                    continue
                pw = np.pad(sw, 1, mode='edge')
                wn = (pw[:-2, 1:-1] + pw[2:, 1:-1] + pw[1:-1, :-2] + pw[1:-1, 2:]) / 4.0
                self.strata_weight[sname] += k * (wn - sw)
                self.strata_weight[sname] = np.maximum(self.strata_weight[sname], 0.0)
                sp = self.strata_pool[sname]
                pp = np.pad(sp, ((1, 1), (1, 1), (0, 0)), mode='edge')
                gn = (pp[:-2, 1:-1, :] + pp[2:, 1:-1, :] + pp[1:-1, :-2, :] + pp[1:-1, 2:, :]) / 4.0
                self.strata_pool[sname] += k * (gn - sp)

    def _update_viral_environment(self):
        """Viral particle decay + diffusion (unchanged from Phase 2 Step 4)."""
        c = self.cfg
        self.viral_particles *= (1.0 - c.viral_decay_rate)
        pv = np.pad(self.viral_particles, 1, mode='edge')
        vn = (pv[:-2, 1:-1] + pv[2:, 1:-1] + pv[1:-1, :-2] + pv[1:-1, 2:]) / 4.0
        self.viral_particles += c.viral_diffusion_rate * (vn - self.viral_particles)
        self.viral_particles = np.maximum(self.viral_particles, 0.0)

        if self.timestep % 10 == 0 and self.viral_genome_weight.max() > 0.001:
            self.viral_genome_weight *= (1.0 - c.viral_decay_rate * 10)
            self.viral_genome_weight = np.maximum(self.viral_genome_weight, 0.0)
            pp = np.pad(self.viral_genome_pool, ((1, 1), (1, 1), (0, 0)), mode='edge')
            gn = (pp[:-2, 1:-1, :] + pp[2:, 1:-1, :] + pp[1:-1, :-2, :] + pp[1:-1, 2:, :]) / 4.0
            self.viral_genome_pool += c.viral_diffusion_rate * (gn - self.viral_genome_pool)

    def _update_density(self):
        """Count organisms per cell."""
        self.density[:] = 0
        if self.pop > 0:
            np.add.at(self.density, (self.rows, self.cols), 1)

    # ══════════════════════════════════════════════════════════════════════════
    # SENSING (expanded to 15 channels)
    # ══════════════════════════════════════════════════════════════════════════

    def _sense_local(self):
        """
        Each organism reads 15 sensor values (was 8 in Phase 2):

          [0]  local_light          — Brightness at this cell
          [1]  local_toxic          — Toxicity at this cell
          [2]  local_nutrients      — Nutrients at this cell
          [3]  local_density        — How many organisms here
          [4]  light_grad_y         — Light gradient vertical
          [5]  light_grad_x         — Light gradient horizontal
          [6]  toxic_grad_y         — Toxicity gradient vertical
          [7]  toxic_grad_x         — Toxicity gradient horizontal
          [8]  local_decomp         — Raw decomposition at this cell (NEW)
          [9]  density_grad_y       — Density gradient vertical (NEW)
          [10] density_grad_x       — Density gradient horizontal (NEW)
          [11] scent_grad_y         — Decomp scent gradient vertical (NEW)
          [12] scent_grad_x         — Decomp scent gradient horizontal (NEW)
          [13] nutrient_grad_y      — Nutrient gradient vertical (NEW)
          [14] nutrient_grad_x      — Nutrient gradient horizontal (NEW)

        The new channels enable:
          - Detritivores to follow decomp scent toward food
          - Carnivores to follow density gradients toward prey clusters
          - Foragers to follow nutrient gradients toward rich patches
          - Detoxers to navigate by toxicity gradients
        """
        rows, cols = self.rows, self.cols
        N = self.cfg.grid_size
        ru = np.clip(rows - 1, 0, N - 1)   # Row above
        rd = np.clip(rows + 1, 0, N - 1)   # Row below
        cl = np.clip(cols - 1, 0, N - 1)   # Column left
        cr = np.clip(cols + 1, 0, N - 1)   # Column right
        return np.column_stack([
            self.light[rows, cols],                                          # 0
            self.toxic[rows, cols],                                          # 1
            self.nutrients[rows, cols],                                      # 2
            self.density[rows, cols].astype(np.float64),                     # 3
            self.light[ru, cols] - self.light[rd, cols],                     # 4
            self.light[rows, cr] - self.light[rows, cl],                     # 5
            self.toxic[ru, cols] - self.toxic[rd, cols],                     # 6
            self.toxic[rows, cr] - self.toxic[rows, cl],                     # 7
            self.decomposition[rows, cols],                                  # 8  (NEW)
            self.density[ru, cols].astype(np.float64)
                - self.density[rd, cols].astype(np.float64),                 # 9  (NEW)
            self.density[rows, cr].astype(np.float64)
                - self.density[rows, cl].astype(np.float64),                 # 10 (NEW)
            self.decomp_scent[ru, cols] - self.decomp_scent[rd, cols],       # 11 (NEW)
            self.decomp_scent[rows, cr] - self.decomp_scent[rows, cl],       # 12 (NEW)
            self.nutrients[ru, cols] - self.nutrients[rd, cols],              # 13 (NEW)
            self.nutrients[rows, cr] - self.nutrients[rows, cl],              # 14 (NEW)
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ENERGY ACQUISITION (the big multi-module energy system)
    # ══════════════════════════════════════════════════════════════════════════

    def _acquire_energy(self, readings):
        """
        Calculate total energy gained by all organisms from all sources.
        
        This is now a complex multi-pathway system where different modules
        contribute to energy in different ways:
        
        1. PHOTOSYNTHESIS (PHOTO module): Convert light → energy
           Same 5-factor formula from Phase 1. Penalized if omnivore.
        
        2. CHEMOSYNTHESIS (CHEMO module): Convert toxicity → energy
           Uses toxin concentration with Michaelis-Menten saturation.
        
        3. SCAVENGING (CONSUME module): Eat decomposing matter
           Detritivores extract energy from the decomposition grid.
           The decomp_preference weight controls how good they are at this.
        
        4. FORAGE BONUSES (FORAGE module):
           a) Extraction efficiency: +25% to ALL energy production
           b) Cooperative signal: nearby foragers boost each other
        
        Note: PREDATION (hunting live prey) is handled separately in _predation()
        because it involves interactions between specific organisms.
        
        METABOLIC INTERFERENCE: Omnivores (organisms with both production AND
        consumption modules) are penalized at both activities:
          - Hunting at 30% effectiveness
          - Production at 85% effectiveness
        Obligate consumers (no production modules) get +50% hunting bonus.
        """
        c = self.cfg
        n = self.pop
        total_gain = np.zeros(n)

        # Shorthand for commonly used sensor readings
        local_light = readings[:, 0]
        local_toxic = readings[:, 1]
        local_density = readings[:, 3]
        local_decomp = readings[:, 8]

        # ── Metabolic interference masks ─────────────────────────────────────
        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]
        is_omnivore = has_producer & has_consume   # Has both → penalized at both
        is_specialist = has_consume & ~has_producer  # Pure consumer → bonus

        # Multipliers based on metabolic interference
        prod_mult = np.where(is_omnivore, c.consume_producer_penalty, 1.0)  # 0.85 for omnivores
        cons_mult = np.where(is_omnivore, c.producer_consume_penalty, 1.0)  # 0.30 for omnivores
        spec_mult = np.where(is_specialist, 1.0 + c.consumer_specialist_bonus, 1.0)  # 1.5 for specialists

        # ── FORAGE extraction bonus ──────────────────────────────────────────
        # Foragers get up to +25% bonus to ALL resource extraction.
        has_forage = self.module_active[:, M_FORAGE]
        forage_mult = np.ones(n)
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            extract_eff = c.forage_extraction_bonus * (1.0 / (1.0 + np.exp(-fw[:, 0])))
            forage_mult = 1.0 + extract_eff * has_forage  # 1.0 to 1.25

        # ── PHOTOSYNTHESIS ───────────────────────────────────────────────────
        # Same 5-factor formula as Phase 1. Now multiplied by prod_mult
        # (omnivore penalty) and forage_mult (forager bonus).
        has_photo = self.module_active[:, M_PHOTO]
        if has_photo.any():
            ph = self._module_weights(M_PHOTO)
            eff = c.photosynthesis_base * (1.0 + 0.3 * np.tanh(ph[:, 0]))
            tol = np.maximum(0.1, 1.0 + 0.5 * np.tanh(ph[:, 1]))
            tp = np.maximum(0.0, 1.0 - local_toxic * c.toxic_photo_penalty / tol)
            le = np.maximum(0.3, 1.0 - 0.3 * np.tanh(ph[:, 2]))
            lm = np.power(np.maximum(local_light, 0.01), le)
            st = 0.5 + 0.5 / (1.0 + np.exp(-ph[:, 3]))
            sh = 1.0 / np.maximum(1.0, local_density * 0.5)
            total_gain += (eff * tp * lm * st * sh * prod_mult * forage_mult) * has_photo

        # ── CHEMOSYNTHESIS ───────────────────────────────────────────────────
        # Converts toxicity into energy. Higher toxicity = more energy,
        # but with Michaelis-Menten saturation (diminishing returns).
        has_chemo = self.module_active[:, M_CHEMO]
        if has_chemo.any():
            ch = self._module_weights(M_CHEMO)
            eff = c.chemosynthesis_base * (1.0 + 0.3 * np.tanh(ch[:, 0]))
            sat = 1.0 + np.abs(ch[:, 2]) * 0.5  # Saturation threshold
            toxic_factor = local_toxic / (local_toxic + sat)  # Michaelis-Menten
            sh = 1.0 / np.maximum(1.0, local_density * 0.4)
            total_gain += (eff * toxic_factor * sh * prod_mult * forage_mult) * has_chemo

        # ── SCAVENGING (CONSUME module eating decomposition) ─────────────────
        # CONSUME organisms can eat dead matter from the decomposition grid.
        # decomp_preference (gene 2) controls how well they scavenge:
        #   High decomp_pref → detritivore (great at scavenging)
        #   Low decomp_pref → carnivore (scavenging is secondary)
        if has_consume.any():
            uw = self._module_weights(M_CONSUME)
            decomp_pref = 1.0 / (1.0 + np.exp(-uw[:, 2]))  # 0-1 preference
            detr_bonus = 1.0 + 1.0 * decomp_pref  # Up to 2x for dedicated detritivores
            eff = c.scavenge_base * (0.3 + 0.7 * decomp_pref) * cons_mult * spec_mult * detr_bonus
            handling = 1.0 + 0.3 * np.tanh(uw[:, 1])  # Handling efficiency
            extract_rate = 0.10 + 0.10 * decomp_pref  # How much decomp per step (up to 20%)
            intake_cap = 1.5 + 1.5 * decomp_pref      # Max decomp intake per step
            available = np.minimum(local_decomp * extract_rate, intake_cap)
            scavenge_gain = eff * handling * available * forage_mult
            total_gain += scavenge_gain * has_consume

            # Remove consumed decomposition from the grid
            consumed = np.minimum(local_decomp * 0.15, scavenge_gain * 0.05) * has_consume
            np.add.at(self.decomposition, (self.rows, self.cols), -consumed)
            np.clip(self.decomposition, 0, 30.0, out=self.decomposition)

        # ── FORAGE cooperative signal ────────────────────────────────────────
        # Nearby foragers boost each other's energy gain. The cooperative_signal
        # weight controls how strongly each organism contributes to the group.
        # This creates a mild incentive for foragers to cluster.
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            coop_strength = 1.0 / (1.0 + np.exp(-fw[:, 3]))  # 0-1 cooperation willingness
            # Count foragers per cell
            forage_density = np.zeros_like(self.density)
            fidx = np.where(has_forage)[0]
            if len(fidx) > 0:
                np.add.at(forage_density, (self.rows[fidx], self.cols[fidx]), 1)
            local_foragers = forage_density[self.rows, self.cols].astype(np.float64)
            # Bonus = (neighbors - self) × bonus_rate × cooperation_strength
            coop_bonus = (np.maximum(0, local_foragers - 1) * c.forage_cooperative_bonus
                         * coop_strength * has_forage)
            total_gain += coop_bonus

        return total_gain

    # ══════════════════════════════════════════════════════════════════════════
    # PREDATION (the combat system)
    # ══════════════════════════════════════════════════════════════════════════

    def _predation(self):
        """
        Organisms with CONSUME module can hunt and kill other organisms.
        
        This is a complex spatial interaction with many balancing mechanisms:
        
        HUNT PROCESS (for each predator):
          1. Check hunt radius (1 cell if sessile, 2 if mobile)
          2. Find candidates in nearby cells (skip camouflaged prey)
          3. Select target (prefer weaker prey — lower energy = easier)
          4. Roll for success (base 14% + skill bonuses - defense penalties)
          5. On success: gain 55% of prey's energy, prey dies
          6. On failure: take counter-attack damage from defended prey
          7. If successful: 70% chance to become satiated (skip next hunt)
        
        DEFENSE INTEGRATION:
          - Shell: Directly reduces hunt success probability (up to -55%)
          - Camouflage: Chance prey is invisible to predator (up to 35%)
          - Counter-attack: Damages predator on failed hunt (up to 5 energy)
          - Size: Indirectly helps via higher maintenance cost (no direct combat effect)
        
        BALANCE MECHANISMS:
          - Per-cell kill cap (1 per step) prevents local extinction
          - Predator satiation reduces hunting pressure after a meal
          - Specialists are better hunters (+15% base + +50% energy gain)
          - Omnivores are bad hunters (30% effectiveness)
          - Weaker prey targeted preferentially (energy-proportional selection)
        
        OPTIMIZATION:
          Uses a flat spatial index (cell → sorted organism indices) for fast
          neighbor lookup instead of nested loops over the entire grid.
        """
        c = self.cfg
        n = self.pop
        if n < 2:
            return 0

        N = c.grid_size
        has_consume = self.module_active[:, M_CONSUME]
        has_move = self.module_active[:, M_MOVE]
        has_defense = self.module_active[:, M_DEFENSE]
        predator_mask = has_consume.copy()
        if not predator_mask.any():
            return 0

        # ── Precompute predator stats ────────────────────────────────────────
        uw = self._module_weights(M_CONSUME)
        aggression = 1.0 / (1.0 + np.exp(-uw[:, 3]))       # 0-1 aggressiveness
        decomp_pref = 1.0 / (1.0 + np.exp(-uw[:, 2]))     # 0-1 (high = detritivore)
        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        is_specialist = has_consume & ~has_producer  # Pure consumers
        is_omnivore = has_consume & has_producer     # Mixed

        # ── Precompute defense stats ─────────────────────────────────────────
        shell_val = np.zeros(n)    # Reduces hunt success probability
        camo_val = np.zeros(n)     # Chance to be invisible
        counter_val = np.zeros(n)  # Damage dealt to predator on miss
        if has_defense.any():
            dw = self._module_weights(M_DEFENSE)
            shell_val = c.defense_shell_max * (1.0 / (1.0 + np.exp(-dw[:, 0]))) * has_defense
            camo_val = c.defense_camouflage_max * (1.0 / (1.0 + np.exp(-dw[:, 1]))) * has_defense
            counter_val = c.defense_counter_damage * np.tanh(np.maximum(0, dw[:, 3])) * has_defense

        # Pre-roll camouflage outcomes for all organisms this step
        camo_rolls = self.rng.random(n)

        # ── Pre-compute hunt success base for all predators ──────────────────
        # Base 14% + aggression bonus, minus detritivore penalty
        hunt_skill = aggression * (1.0 - 0.5 * decomp_pref)  # Detritivores hunt worse
        base_prob = c.predation_base_success + hunt_skill * 0.3
        base_prob += np.where(has_move, 0.08, 0.0)        # Mobile predators: +8%
        base_prob += np.where(is_specialist, 0.15, 0.0)   # Specialists: +15%
        omni_mult = np.where(is_omnivore, c.producer_consume_penalty, 1.0)  # Omnivores: ×0.30

        # Energy fraction gained on kill (specialists get more)
        frac_base = np.where(is_specialist,
                             c.predation_energy_fraction * (1.0 + c.consumer_specialist_bonus),
                             c.predation_energy_fraction)

        # ── Build spatial index for fast neighbor lookup ──────────────────────
        # Instead of searching all organisms, we sort them by cell and use
        # a lookup table: cell_key → (start_index, count) in the sorted array.
        cell_key = self.rows * N + self.cols
        sort_idx = np.argsort(cell_key)
        sorted_keys = cell_key[sort_idx]
        unique_cells, cell_starts, cell_counts = np.unique(
            sorted_keys, return_index=True, return_counts=True)
        cell_start_map = np.full(N * N, -1, dtype=np.int32)
        cell_count_map = np.zeros(N * N, dtype=np.int32)
        cell_start_map[unique_cells] = cell_starts.astype(np.int32)
        cell_count_map[unique_cells] = cell_counts.astype(np.int32)

        # ── Hunt loop ────────────────────────────────────────────────────────
        kill_mask = np.zeros(n, dtype=bool)        # Track dead prey
        cell_kill_count = np.zeros(N * N, dtype=np.int32)  # Per-cell kill cap
        satiated = np.zeros(n, dtype=bool)         # Full predators skip hunting
        kills = 0

        predator_idx = np.where(predator_mask)[0]
        self.rng.shuffle(predator_idx)  # Random order prevents systematic bias

        # Pre-roll hunt and satiation outcomes for all predators
        hunt_rolls = self.rng.random(len(predator_idx))
        sat_rolls = self.rng.random(len(predator_idx))

        for pi_idx, pi in enumerate(predator_idx):
            if kill_mask[pi] or satiated[pi]:
                continue  # Dead or full — skip

            pr, pc = int(self.rows[pi]), int(self.cols[pi])
            R = c.predation_hunt_radius_mobile if has_move[pi] else c.predation_hunt_radius_base

            # ── Find candidates in nearby cells ──────────────────────────────
            candidates = []
            r_lo, r_hi = max(0, pr - R), min(N - 1, pr + R)
            c_lo, c_hi = max(0, pc - R), min(N - 1, pc + R)
            for cr in range(r_lo, r_hi + 1):
                for cc_iter in range(c_lo, c_hi + 1):
                    ck = cr * N + cc_iter  # Cell key
                    if cell_kill_count[ck] >= c.predation_max_kills_per_cell:
                        continue  # Kill cap reached for this cell
                    cs = cell_start_map[ck]
                    if cs < 0:
                        continue  # No organisms in this cell
                    cnt = cell_count_map[ck]
                    for si in range(cs, cs + cnt):
                        j = sort_idx[si]
                        if j != pi and not kill_mask[j]:
                            # Camouflage check: roll was pre-computed
                            if camo_val[j] > 0 and camo_rolls[j] < camo_val[j]:
                                continue  # Prey is camouflaged — invisible
                            candidates.append(j)

            if not candidates:
                continue

            # ── Target selection: prefer weaker prey ─────────────────────────
            # Lower energy = higher selection probability (easier to catch)
            cands = np.array(candidates)
            target_e = np.maximum(self.energy[cands], 1.0)
            inv_e = 1.0 / target_e
            inv_e /= inv_e.sum()  # Normalize to probability distribution
            ti = cands[self.rng.choice(len(cands), p=inv_e)]  # Weighted random pick

            # ── Calculate success probability ────────────────────────────────
            target_difficulty = np.clip(self.energy[ti] / c.energy_max, 0.1, 1.0)
            prob = (base_prob[pi] - target_difficulty * 0.08) * omni_mult[pi]
            prob -= shell_val[ti]  # Shell defense reduces probability
            prob = np.clip(prob, 0.01, 0.60)  # Floor 1%, ceiling 60%

            # ── Roll for kill ────────────────────────────────────────────────
            if hunt_rolls[pi_idx] < prob:
                # SUCCESS: predator gains energy, prey dies
                gained = max(0.0, self.energy[ti]) * frac_base[pi]
                self.energy[pi] = min(self.energy[pi] + gained, c.energy_max)
                self.energy[ti] = -1.0  # Mark for death
                kill_mask[ti] = True
                cell_kill_count[int(self.rows[ti]) * N + int(self.cols[ti])] += 1
                kills += 1
                # Satiation check: 70% chance to stop hunting after a meal
                if sat_rolls[pi_idx] < c.predator_satiation:
                    satiated[pi] = True
            else:
                # FAILURE: if prey has counter-attack, predator takes damage
                if counter_val[ti] > 0:
                    self.energy[pi] -= counter_val[ti]

        self.total_predation_kills += kills
        return kills

    # ══════════════════════════════════════════════════════════════════════════
    # TOXIC DAMAGE (now with DETOX tolerance bonus)
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_toxic_damage(self, readings):
        """
        Energy damage from toxicity. Same two-tier system as Phase 2, but now
        with TWO sources of damage reduction:
        
        1. CHEMO tolerance (existing): Chemosynthesizers take less toxic damage.
           They're adapted to toxic environments.
        
        2. DETOX tolerance (NEW): DETOX module raises the damage THRESHOLD.
           Instead of taking damage above 0.8, a maxed detoxer takes damage
           above 1.4. This means DETOX organisms can survive in zones that
           would kill anyone else — they literally metabolize the poison.
        """
        c = self.cfg
        lt = readings[:, 1]  # Local toxicity
        dmg = np.zeros(self.pop)

        # ── CHEMO tolerance: reduces effective toxicity ──────────────────────
        has_chemo = self.module_active[:, M_CHEMO]
        chemo_reduction = np.ones(self.pop)
        if has_chemo.any():
            ch = self._module_weights(M_CHEMO)
            tolerance = 0.3 + 0.4 / (1.0 + np.exp(-ch[:, 1]))  # 0.3 to 0.7
            chemo_reduction[has_chemo] = 1.0 - tolerance[has_chemo]

        # ── DETOX tolerance: raises damage thresholds (NEW) ──────────────────
        has_detox = self.module_active[:, M_DETOX]
        threshold_boost = np.zeros(self.pop)
        if has_detox.any():
            dtw = self._module_weights(M_DETOX)
            threshold_boost = (c.detox_tolerance_bonus
                              * (1.0 / (1.0 + np.exp(-dtw[:, 1])))
                              * has_detox)

        # Apply both: reduced effective toxicity + raised thresholds
        effective = lt * chemo_reduction
        eff_thresh_med = c.toxic_threshold_medium + threshold_boost   # 0.8 + up to 0.6
        eff_thresh_high = c.toxic_threshold_high + threshold_boost    # 1.5 + up to 0.6

        m = effective > eff_thresh_med
        if m.any():
            dmg[m] += (effective[m] - eff_thresh_med[m]) * c.toxic_damage_medium
        h = effective > eff_thresh_high
        if h.any():
            dmg[h] += (effective[h] - eff_thresh_high[h]) * c.toxic_damage_high
        self.energy -= dmg

    # ══════════════════════════════════════════════════════════════════════════
    # DETOX MODULE: Active toxin metabolism (NEW)
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_detox(self):
        """
        DETOX organisms actively metabolize environmental toxins.
        
        Three effects:
          1. ENERGY GAIN: Convert metabolized toxin into energy
          2. ENVIRONMENT CLEANING: Remove toxin from the grid cell
          3. SELECTIVE UPTAKE: Michaelis-Menten kinetics — performs best
             at certain toxin concentrations (not too low, not too high)
        
        This creates a new ecological niche: organisms that THRIVE in toxic
        zones. Combined with CHEMO (which also benefits from toxicity), this
        enables "toxic zone specialists" that live where nobody else can.
        """
        c = self.cfg
        has_detox = self.module_active[:, M_DETOX]
        if not has_detox.any():
            return

        dtw = self._module_weights(M_DETOX)

        # Detox efficiency: how much toxin can be processed per step (up to 8%)
        detox_eff = c.detox_rate_max * (1.0 / (1.0 + np.exp(-dtw[:, 0])))

        # Selective uptake: Michaelis-Menten style — has a preferred concentration
        # Higher selectivity = peaks at lower concentrations, drops off at high ones
        selectivity = 1.0 + np.abs(dtw[:, 3]) * 0.5
        local_toxic = self.toxic[self.rows, self.cols]
        uptake_factor = local_toxic / (local_toxic + selectivity)

        # Amount of toxin metabolized this step
        metabolized = detox_eff * uptake_factor * has_detox

        # Energy gained from metabolized toxin (up to 40% conversion efficiency)
        conversion = c.detox_energy_conversion * (1.0 / (1.0 + np.exp(-dtw[:, 2])))
        energy_gained = metabolized * conversion * local_toxic
        self.energy += energy_gained * has_detox

        # Remove toxin from the environment (50% of what was metabolized)
        toxin_removed = metabolized * c.detox_environment_effect * local_toxic
        np.add.at(self.toxic, (self.rows, self.cols), -toxin_removed * has_detox)
        np.clip(self.toxic, 0, 5.0, out=self.toxic)

    # ══════════════════════════════════════════════════════════════════════════
    # MOVEMENT (expanded for new modules)
    # ══════════════════════════════════════════════════════════════════════════

    def _decide_movement(self, readings):
        """
        Movement decision now incorporates signals from ALL modules:
        
        Base signals (MOVE weights 0-7):
          - Light-seeking (photosynthesizers go toward bright areas)
          - Density avoidance (avoid crowded cells)
          - Nutrient attraction (stay in nutrient-rich spots)
          - Toxic response (chemosynthesizers seek toxicity)
          - Exploration noise
          - Stay tendency
        
        CONSUME signals:
          - Carnivores follow DENSITY gradients (hunt where prey clusters)
          - Detritivores follow SCENT gradients (navigate toward carrion)
          - The decomp_preference gene controls the blend between these
        
        FORAGE signals:
          - Follow NUTRIENT gradients (seek resource-rich patches)
        
        DETOX signals:
          - Follow TOXICITY gradients (seek toxic areas as food source)
        
        Sessile organisms (no MOVE module) are forced to stay in place
        by setting all directional scores to -999.
        """
        n = self.pop
        if n == 0:
            return np.array([], dtype=np.int32)

        has_move = self.module_active[:, M_MOVE]
        has_photo = self.module_active[:, M_PHOTO]
        has_chemo = self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]
        mp = self._module_weights(M_MOVE)

        sc = np.zeros((n, 5))  # Scores for stay/up/down/right/left

        # ── Stay score: nutrients + decomp attraction ────────────────────────
        sc[:, 0] = (mp[:, 5] + mp[:, 2] * readings[:, 2]
                    - mp[:, 1] * np.minimum(readings[:, 3], 10) * 0.1)
        if has_consume.any():
            sc[:, 0] += has_consume * readings[:, 8] * 0.3  # Stay near decomp

        # ── Gradient weights (per-module) ────────────────────────────────────
        # Photosynthesizers follow light, avoid toxicity.
        # Chemosynthesizers follow toxicity (it's their food).
        light_w = mp[:, 6] * has_photo.astype(np.float64)
        net_toxic = mp[:, 7] * has_photo.astype(np.float64) - mp[:, 3] * has_chemo.astype(np.float64)

        # Consumers: carnivores follow density, detritivores follow scent
        consume_dens_w = np.zeros(n)
        consume_scent_w = np.zeros(n)
        if has_consume.any():
            uw = self._module_weights(M_CONSUME)
            dp = 1.0 / (1.0 + np.exp(-uw[:, 2]))  # decomp_preference
            consume_dens_w = has_consume * (1.0 - dp) * 0.5   # Carnivores follow density
            consume_scent_w = has_consume * dp * 8.0           # Detritivores follow scent (strong!)

        # FORAGE: seek nutrient-rich areas
        has_forage = self.module_active[:, M_FORAGE]
        forage_nutr_w = np.zeros(n)
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            resource_discrim = 1.0 / (1.0 + np.exp(-fw[:, 1]))
            forage_nutr_w = has_forage * resource_discrim * 2.0

        # DETOX: seek toxic areas (they're food!)
        has_detox = self.module_active[:, M_DETOX]
        detox_toxic_w = np.zeros(n)
        if has_detox.any():
            dtw = self._module_weights(M_DETOX)
            detox_seek = 1.0 / (1.0 + np.exp(-dtw[:, 0]))
            detox_toxic_w = has_detox * detox_seek * 1.5

        # ── Compute directional scores ───────────────────────────────────────
        # Combine all gradient signals into single Y and X components
        lg_y, lg_x = readings[:, 4], readings[:, 5]    # Light gradients
        tg_y, tg_x = readings[:, 6], readings[:, 7]    # Toxic gradients
        dg_y, dg_x = readings[:, 9], readings[:, 10]   # Density gradients
        sg_y, sg_x = readings[:, 11], readings[:, 12]  # Scent gradients
        ng_y, ng_x = readings[:, 13], readings[:, 14]  # Nutrient gradients

        gy = (light_w * lg_y - net_toxic * tg_y + consume_dens_w * dg_y
              + consume_scent_w * sg_y + forage_nutr_w * ng_y + detox_toxic_w * tg_y)
        gx = (light_w * lg_x - net_toxic * tg_x + consume_dens_w * dg_x
              + consume_scent_w * sg_x + forage_nutr_w * ng_x + detox_toxic_w * tg_x)

        sc[:, 1] = gy;  sc[:, 2] = -gy   # Up / Down
        sc[:, 3] = gx;  sc[:, 4] = -gx   # Right / Left

        # Exploration noise (MOVE gene 4 controls amplitude)
        sc += mp[:, 4:5] * self.rng.normal(0, 0.3, (n, 5))

        # Sessile organisms (no MOVE module) are locked in place
        sc[~has_move, 1:] = -999.0  # Only "stay" option is viable

        return np.argmax(sc, axis=1)

    def _execute_movement(self, actions):
        """Move organisms. Clamp to grid. Deduct movement cost."""
        N = self.cfg.grid_size
        m = actions == 1; self.rows[m] = np.maximum(0, self.rows[m] - 1)
        m = actions == 2; self.rows[m] = np.minimum(N - 1, self.rows[m] + 1)
        m = actions == 3; self.cols[m] = np.minimum(N - 1, self.cols[m] + 1)
        m = actions == 4; self.cols[m] = np.maximum(0, self.cols[m] - 1)
        self.energy[actions > 0] -= self.cfg.energy_movement_cost

    # ══════════════════════════════════════════════════════════════════════════
    # HORIZONTAL GENE TRANSFER (now can transfer modules!)
    # ══════════════════════════════════════════════════════════════════════════

    def _horizontal_transfer(self):
        """
        Stratified HGT from Phase 2, with one major addition:
        
        MODULE ACQUISITION VIA HGT: When absorbing DNA from a stratum,
        there's a 1.5% chance per transfer that the organism gains a new
        module from the fragment. The strata_modules layer tracks which
        modules are common in the dead organisms at each cell.
        
        For example, if many organisms with DEFENSE died at a cell, the
        strata_modules for that cell will have high DEFENSE signal. A living
        organism absorbing fragments from there has a chance to gain DEFENSE.
        
        This means modules can spread horizontally across the population —
        not just through parent-child inheritance, but through "archaeological
        gene transfer" from dead organisms.
        """
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        sp = self._standalone_params()
        receptivity = 1.0 / (1.0 + np.exp(-sp[:, SP_TRANSFER_RECEPTIVITY]))
        selectivity = np.abs(sp[:, SP_TRANSFER_SELECTIVITY])
        local_toxic = self.toxic[self.rows, self.cols]

        # Stratum access gated by toxicity (same as Phase 2)
        can_recent = np.ones(n, dtype=bool)
        can_intermediate = local_toxic >= c.stratum_access_medium
        can_ancient = local_toxic >= c.stratum_access_high

        for sname, access_mask, blend_rate in [
            ("recent",       can_recent,       c.transfer_blend_rate_recent),
            ("intermediate", can_intermediate,  c.transfer_blend_rate_intermediate),
            ("ancient",      can_ancient,       c.transfer_blend_rate_ancient),
        ]:
            sw = self.strata_weight[sname]
            local_sw = sw[self.rows, self.cols]
            eligible = access_mask & (local_sw > 0.1) & (self.rng.random(n) < receptivity)
            eidx = np.where(eligible)[0]
            if len(eidx) == 0:
                continue

            # Selectivity filter (with depth-based relaxation)
            local_frags = self.strata_pool[sname][self.rows[eidx], self.cols[eidx]]
            dists = np.sqrt(np.mean((self.weights[eidx] - local_frags) ** 2, axis=1))
            depth_factor = {"recent": 1.0, "intermediate": 1.5, "ancient": 2.5}[sname]
            thresh = (2.0 * depth_factor) / (1.0 + selectivity[eidx])
            tidx = eidx[dists < thresh]
            if len(tidx) == 0:
                continue

            # Blend weights (same as Phase 2)
            frags = self.strata_pool[sname][self.rows[tidx], self.cols[tidx]]
            self.weights[tidx] = (1.0 - blend_rate) * self.weights[tidx] + blend_rate * frags
            self.transfer_count[tidx] += 1
            self.total_transfers += len(tidx)
            self.transfers_by_stratum[sname] += len(tidx)

            # ── MODULE ACQUISITION VIA HGT (NEW) ────────────────────────────
            # 50% of transfers check for module acquisition (performance savings).
            # For each transferring organism, 1.5% chance to gain a module
            # that was common in the fragment pool (signal > 0.3).
            if self.rng.random() < 0.5:
                frag_mods = self.strata_modules[sname][self.rows[tidx], self.cols[tidx]]
                for i, ti in enumerate(tidx):
                    if self.rng.random() < c.module_transfer_rate:
                        # Find modules present in fragment but absent in organism
                        available = [gm for gm in GAINABLE_MODULES
                                    if frag_mods[i, gm] > 0.3 and not self.module_present[ti, gm]]
                        if available:
                            self.module_present[ti, self.rng.choice(available)] = True
                            self.module_active[ti] = self.module_present[ti]

    # ══════════════════════════════════════════════════════════════════════════
    # VIRAL DYNAMICS (unchanged from Phase 2 Step 4)
    # ══════════════════════════════════════════════════════════════════════════

    def _viral_dynamics(self):
        """
        Lytic/lysogenic viral cycle. Same as Phase 2 Step 4.
        Uses standalone params for resistance/suppression genes.
        Now blends into 'weights' array instead of 'genomes'.
        """
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        sp = self._standalone_params()
        viral_resistance = 1.0 / (1.0 + np.exp(-sp[:, SP_VIRAL_RESISTANCE]))
        lysogenic_suppression = 1.0 / (1.0 + np.exp(-sp[:, SP_LYSO_SUPPRESSION]))
        local_viral = self.viral_particles[self.rows, self.cols]
        local_toxic = self.toxic[self.rows, self.cols]

        # ── New infections ───────────────────────────────────────────────────
        candidates = (self.viral_load == 0) & (local_viral > 0.05)
        if candidates.any():
            cidx = np.where(candidates)[0]
            inf_prob = (c.viral_infection_rate * np.minimum(local_viral[cidx], 5.0) / 5.0
                       * (1.0 - viral_resistance[cidx] * 0.8))
            infected = cidx[self.rng.random(len(cidx)) < inf_prob]

            if len(infected) > 0:
                lyso_rolls = self.rng.random(len(infected))
                lytic = infected[lyso_rolls >= c.lysogenic_probability]
                lysogenic = infected[lyso_rolls < c.lysogenic_probability]

                if len(lytic) > 0:
                    self.viral_load[lytic] = 0.1
                if len(lysogenic) > 0:
                    vgw = self.viral_genome_weight[self.rows[lysogenic], self.cols[lysogenic]]
                    has_mat = vgw > 0.01
                    lwm = lysogenic[has_mat]
                    if len(lwm) > 0:
                        vg = self.viral_genome_pool[self.rows[lwm], self.cols[lwm]]
                        new_str = self.lysogenic_strength[lwm] + 0.3
                        blend = 0.3 / np.maximum(new_str, 0.01)
                        self.lysogenic_genome[lwm] = (
                            self.lysogenic_genome[lwm] * (1.0 - blend[:, None])
                            + vg * blend[:, None])
                        self.lysogenic_strength[lwm] = new_str
                        self.total_lysogenic_integrations += len(lwm)

        # ── Lytic progression ────────────────────────────────────────────────
        lytic_mask = self.viral_load > 0
        if lytic_mask.any():
            lidx = np.where(lytic_mask)[0]
            self.viral_load[lidx] += c.viral_lytic_growth
            self.energy[lidx] -= c.viral_lytic_damage * self.viral_load[lidx]

        # ── Lysogenic activation under toxic stress ──────────────────────────
        act_cand = ((self.lysogenic_strength > 0.01) &
                    (local_toxic > c.lysogenic_activation_toxic) &
                    (self.viral_load == 0))
        if act_cand.any():
            acidx = np.where(act_cand)[0]
            stress = (local_toxic[acidx] - c.lysogenic_activation_toxic) / c.lysogenic_activation_toxic
            act_prob = np.minimum(stress * 0.3, 0.5) * (1.0 - lysogenic_suppression[acidx] * 0.7)
            activated = acidx[self.rng.random(len(acidx)) < act_prob]
            if len(activated) > 0:
                blend = np.minimum(c.lysogenic_blend_rate * self.lysogenic_strength[activated], 0.4)
                self.weights[activated] = (
                    self.weights[activated] * (1.0 - blend[:, None])
                    + self.lysogenic_genome[activated] * blend[:, None])
                self.viral_load[activated] = 0.2
                self.lysogenic_strength[activated] *= 0.3
                self.total_lysogenic_activations += len(activated)

    # ══════════════════════════════════════════════════════════════════════════
    # REPRODUCTION (with module evolution)
    # ══════════════════════════════════════════════════════════════════════════

    def _reproduce(self):
        """
        Reproduction with three major additions:
        
        1. DENSITY-DEPENDENT THRESHOLD: Harder to reproduce in crowds.
           Threshold = base 80 + (local_density × 5).
           This prevents explosive population growth in clusters.
        
        2. MODULE EVOLUTION: Children can gain or lose modules:
           - 0.5% chance to GAIN a random module from GAINABLE_MODULES
           - 0.3% chance to LOSE a module (with safety checks):
             • Can't lose TOXPROD
             • Can't lose your LAST energy module (PHOTO/CHEMO/CONSUME)
        
        3. DETRITIVORE OFFSPRING SEEKING: Babies from detritivore parents
           are placed near the strongest decomp scent signal rather than
           randomly near the parent. This ensures detritivore babies start
           near food — critical for survival.
        
        Also: mobile parents place offspring farther away (5 cells) than
        sessile parents (1 cell).
        """
        c = self.cfg
        N = c.grid_size

        # ── Density-dependent reproduction threshold ─────────────────────────
        local_dens = self.density[self.rows, self.cols].astype(np.float64)
        effective_threshold = c.energy_reproduction_threshold + local_dens * c.repro_density_penalty

        # Requirements: enough energy + old enough + not lytic-infected
        can = ((self.energy >= effective_threshold) &
               (self.age >= c.min_reproduction_age) &
               (self.viral_load == 0))
        pidx = np.where(can)[0]
        nb = len(pidx)
        if nb == 0:
            return

        self.energy[pidx] -= c.energy_reproduction_cost
        child_ids = np.arange(self.next_id, self.next_id + nb, dtype=np.int64)
        self.next_id += nb

        # ── Module evolution ─────────────────────────────────────────────────
        child_modules = self.module_present[pidx].copy()

        # Module GAIN: 0.5% chance per birth
        gain_rolls = self.rng.random(nb)
        for i in np.where(gain_rolls < c.module_gain_rate)[0]:
            absent = [gm for gm in GAINABLE_MODULES if not child_modules[i, gm]]
            if absent:
                child_modules[i, self.rng.choice(absent)] = True

        # Module LOSS: 0.3% chance per birth, with safety checks
        lose_rolls = self.rng.random(nb)
        for i in np.where(lose_rolls < c.module_lose_rate)[0]:
            present = np.where(child_modules[i])[0]
            droppable = []
            for m in present:
                if m == M_TOXPROD:
                    continue  # Never lose TOXPROD
                # If this is an energy module (PHOTO/CHEMO/CONSUME), only lose it
                # if another energy module exists (prevent energy-less organisms)
                if m in (M_PHOTO, M_CHEMO, M_CONSUME):
                    if not any(child_modules[i, e] for e in (M_PHOTO, M_CHEMO, M_CONSUME) if e != m):
                        continue  # Would leave organism with no energy source
                droppable.append(m)
            if droppable:
                child_modules[i, self.rng.choice(droppable)] = False

        # ── Offspring placement ──────────────────────────────────────────────
        # Mobile parents: offspring up to 5 cells away
        # Sessile parents: offspring only 1 cell away
        has_move_p = self.module_active[pidx, M_MOVE]
        off_dist = np.where(has_move_p, c.offspring_distance, 1)
        row_off = np.round(self.rng.uniform(-1, 1, nb) * off_dist).astype(np.int64)
        col_off = np.round(self.rng.uniform(-1, 1, nb) * off_dist).astype(np.int64)

        # ── Detritivore offspring seeking (NEW) ──────────────────────────────
        # Detritivore children seek the strongest decomp scent within 8 cells
        # of the parent. This places babies near food sources.
        has_consume_p = self.module_active[pidx, M_CONSUME]
        if has_consume_p.any():
            uw = self._module_weights(M_CONSUME)
            dp_pref = 1.0 / (1.0 + np.exp(-uw[pidx, 2]))  # Parent's decomp preference
            seekers = np.where(has_consume_p & (dp_pref > 0.4))[0]  # Only strong detritivores
            if len(seekers) > 0:
                search_r = 8  # Search radius
                for i in seekers:
                    pr, pc = int(self.rows[pidx[i]]), int(self.cols[pidx[i]])
                    r_lo = max(0, pr - search_r)
                    r_hi = min(N - 1, pr + search_r)
                    c_lo = max(0, pc - search_r)
                    c_hi = min(N - 1, pc + search_r)
                    # Sample scent in search area (step by 2 for performance)
                    patch = self.decomp_scent[r_lo:r_hi+1:2, c_lo:c_hi+1:2]
                    if patch.size > 0 and patch.max() > 0.01:
                        best = np.unravel_index(patch.argmax(), patch.shape)
                        row_off[i] = r_lo + best[0] * 2 - pr
                        col_off[i] = c_lo + best[1] * 2 - pc

        # ── Create offspring ─────────────────────────────────────────────────
        self._append_organisms({
            "rows": np.clip(self.rows[pidx] + row_off, 0, N - 1),
            "cols": np.clip(self.cols[pidx] + col_off, 0, N - 1),
            "energy": np.full(nb, c.energy_initial),
            "age": np.zeros(nb, dtype=np.int32),
            "generation": self.generation[pidx] + 1,
            "ids": child_ids,
            "parent_ids": self.ids[pidx],
            "weights": self.weights[pidx] + self.rng.normal(0, c.mutation_rate, (nb, TOTAL_WEIGHT_PARAMS)),
            "module_present": child_modules,
            "module_active": child_modules.copy(),
            "transfer_count": np.zeros(nb, dtype=np.int32),
            "viral_load": np.zeros(nb, dtype=np.float64),
            "lysogenic_strength": self.lysogenic_strength[pidx] * c.lysogenic_inheritance,
            "lysogenic_genome": self.lysogenic_genome[pidx] * c.lysogenic_inheritance,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # DEATH & DECOMPOSITION (now deposits modules into strata)
    # ══════════════════════════════════════════════════════════════════════════

    def _kill_and_decompose(self):
        """
        Death + decomposition. Three causes: viral burst, starvation, old age.
        
        Key addition: dead organisms now deposit their MODULE PRESENCE into
        the strata alongside their weights. This means the fragment pool
        "remembers" which modules were common in the dead organisms at each
        cell, enabling module transfer via HGT.
        
        Also: larger decomposition deposit (2.0 base, was 0.5) because
        decomposition is now a primary food source for detritivores.
        """
        c = self.cfg
        N = c.grid_size

        bursting = self.viral_load >= c.viral_burst_threshold
        natural_death = (self.energy <= 0) | (self.age >= c.max_age)
        dead = bursting | natural_death

        if dead.any():
            dr, dc = self.rows[dead], self.cols[dead]
            de = np.maximum(0, self.energy[dead])
            dw = self.weights[dead]          # Dead organisms' weights
            dm = self.module_present[dead]   # Dead organisms' modules

            # ── Decomposition deposit ────────────────────────────────────────
            # Larger base deposit (2.0) — this is food for detritivores
            np.add.at(self.decomposition, (dr, dc), de * c.nutrient_from_decomp + c.decomp_death_deposit)

            # ── Strata deposit (weights + modules) ───────────────────────────
            cell_ids = dr * N + dc
            unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
            nc = len(unique_cells)
            weight_sums = np.zeros((nc, TOTAL_WEIGHT_PARAMS))
            module_sums = np.zeros((nc, N_MODULES))
            counts = np.zeros(nc)
            np.add.at(weight_sums, inverse, dw)
            np.add.at(module_sums, inverse, dm.astype(np.float64))
            np.add.at(counts, inverse, 1.0)
            ur, uc = unique_cells // N, unique_cells % N

            wt = self.strata_weight["recent"][ur, uc] + counts
            blend = counts / wt
            avg_w = weight_sums / counts[:, None]
            avg_m = module_sums / counts[:, None]
            # Blend weights into recent stratum
            self.strata_pool["recent"][ur, uc] = (
                self.strata_pool["recent"][ur, uc] * (1.0 - blend[:, None])
                + avg_w * blend[:, None])
            # Blend modules into recent stratum (NEW)
            self.strata_modules["recent"][ur, uc] = (
                self.strata_modules["recent"][ur, uc] * (1.0 - blend[:, None])
                + avg_m * blend[:, None])
            self.strata_weight["recent"][ur, uc] = wt

            # ── Viral burst (unchanged) ──────────────────────────────────────
            burst_mask = self.viral_load[dead] >= c.viral_burst_threshold
            if burst_mask.any():
                br, bc, bg = dr[burst_mask], dc[burst_mask], dw[burst_mask]
                self.total_lytic_deaths += len(br)
                for i in range(len(br)):
                    r0 = max(0, br[i] - c.viral_burst_radius)
                    r1 = min(N, br[i] + c.viral_burst_radius + 1)
                    c0 = max(0, bc[i] - c.viral_burst_radius)
                    c1 = min(N, bc[i] + c.viral_burst_radius + 1)
                    area = (r1 - r0) * (c1 - c0)
                    self.viral_particles[r0:r1, c0:c1] += c.viral_burst_amount / area
                    w_old = self.viral_genome_weight[r0:r1, c0:c1]
                    w_add = c.viral_burst_amount / area
                    w_new = w_old + w_add
                    blend_v = w_add / np.maximum(w_new, 1e-8)
                    self.viral_genome_pool[r0:r1, c0:c1] = (
                        self.viral_genome_pool[r0:r1, c0:c1] * (1.0 - blend_v[:, :, None])
                        + bg[i][None, None, :] * blend_v[:, :, None])
                    self.viral_genome_weight[r0:r1, c0:c1] = w_new

            # ── Spontaneous viral shedding from natural deaths ───────────────
            nat_dead = ~burst_mask
            if nat_dead.any():
                spont = self.rng.random(int(nat_dead.sum())) < 0.15
                if spont.any():
                    nat_idx = np.where(nat_dead)[0][spont]
                    np.add.at(self.viral_particles, (dr[nat_idx], dc[nat_idx]), 1.0)
                    np.add.at(self.viral_genome_weight, (dr[nat_idx], dc[nat_idx]), 1.0)

        self._filter_organisms(~dead)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN SIMULATION LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def update(self):
        """
        One timestep. Order of operations:
        
          1.  Density count
          2.  Sense (15 channels now)
          3.  Energy acquisition (PHOTO + CHEMO + scavenging + FORAGE bonuses)
          4.  Toxic damage (with CHEMO + DETOX tolerance)
          5.  DETOX active metabolism (clean toxins, gain energy)
          6.  Module costs + DEFENSE surcharge
          7.  Sessile crowding penalty (producers only)
          8.  Movement (with module-specific navigation signals)
          9.  Aging
          10. Predation (CONSUME vs DEFENSE arms race)
          11. Horizontal transfer (every 5 steps, with module acquisition)
          12. Viral dynamics (every 3 steps)
          13. Reproduction (with module evolution + density penalty)
          14. Death + decomposition (deposits modules into strata)
          15. Environment update
        """
        if self.pop == 0:
            self._update_environment()
            self.timestep += 1
            self._record_stats(0)
            return

        self._update_density()                                                  # 1
        readings = self._sense_local()                                          # 2

        effective_max = self._effective_energy_max()                             # FORAGE storage
        self.energy = np.minimum(                                               # 3
            self.energy + self._acquire_energy(readings), effective_max)
        self._apply_toxic_damage(readings)                                      # 4
        self._apply_detox()                                                     # 5
        self.energy -= self._compute_total_costs()                              # 6

        # 7. Sessile crowding penalty: producers without MOVE get penalized
        #    for being in crowded cells. Consumers thrive in crowds (more prey).
        no_move = ~self.module_active[:, M_MOVE]
        sessile_prod = no_move & ~self.module_active[:, M_CONSUME]
        if sessile_prod.any():
            crowding = np.maximum(0, readings[:, 3] - 2) * 0.15
            self.energy[sessile_prod] -= crowding[sessile_prod]

        self._execute_movement(self._decide_movement(readings))                 # 8
        self.age += 1                                                           # 9

        kills = 0
        if self.timestep % self.cfg.predation_check_interval == 0:              # 10
            kills = self._predation()
        if self.timestep % self.cfg.transfer_check_interval == 0:               # 11
            self._horizontal_transfer()
        if self.timestep % self.cfg.viral_check_interval == 0:                  # 12
            self._viral_dynamics()

        self._reproduce()                                                       # 13
        self._kill_and_decompose()                                              # 14
        self._update_environment()                                              # 15
        self._record_stats(kills)
        self.timestep += 1

    # ══════════════════════════════════════════════════════════════════════════
    # STATS & SNAPSHOTS
    # ══════════════════════════════════════════════════════════════════════════

    def _classify_roles(self):
        """
        Classify all living organisms into ecological roles:
          - Producer:    Has PHOTO or CHEMO, no CONSUME
          - Omnivore:    Has production AND consumption modules
          - Carnivore:   CONSUME only, low decomp_preference (<0.5)
          - Detritivore: CONSUME only, high decomp_preference (>=0.5)
        """
        n = self.pop
        if n == 0:
            return {"producer": 0, "carnivore": 0, "detritivore": 0, "omnivore": 0}

        has_prod = self.module_present[:, M_PHOTO] | self.module_present[:, M_CHEMO]
        has_cons = self.module_present[:, M_CONSUME]
        producer = has_prod & ~has_cons
        omnivore = has_prod & has_cons
        obligate = has_cons & ~has_prod  # Pure consumers

        carnivore = detritivore = np.zeros(n, dtype=bool)
        if obligate.any():
            uw = self._module_weights(M_CONSUME)
            dp = 1.0 / (1.0 + np.exp(-uw[:, 2]))  # decomp_preference
            carnivore = obligate & (dp < 0.5)       # Prefer live prey
            detritivore = obligate & (dp >= 0.5)     # Prefer dead matter

        return {
            "producer": int(producer.sum()),
            "carnivore": int(carnivore.sum()),
            "detritivore": int(detritivore.sum()),
            "omnivore": int(omnivore.sum()),
        }

    def _record_stats(self, kills=0):
        """
        Record per-timestep metrics. Now includes:
          - Per-module counts (how many organisms have each module)
          - Ecological role breakdown (producer/carnivore/detritivore/omnivore)
          - Average modules per organism
          - Predation kills this step + cumulative
        """
        p = self.pop
        if p > 0:
            roles = self._classify_roles()
            mod_counts = {MODULE_NAMES[m]: int(self.module_present[:, m].sum())
                         for m in range(N_MODULES)}
            self.stats_history.append({
                "t": self.timestep, "pop": p,
                "avg_energy": round(float(self.energy.mean()), 1),
                "max_gen": int(self.generation.max()),
                "toxic_mean": round(float(self.toxic.mean()), 3),
                "decomp_mean": round(float(self.decomposition.mean()), 2),
                "avg_modules": round(float(self.module_present.sum(axis=1).mean()), 2),
                "module_counts": mod_counts,
                "roles": roles,
                "kills": kills,
                "total_kills": self.total_predation_kills,
            })
        else:
            self.stats_history.append({
                "t": self.timestep, "pop": 0, "avg_energy": 0, "max_gen": 0,
                "toxic_mean": round(float(self.toxic.mean()), 3),
                "decomp_mean": round(float(self.decomposition.mean()), 2),
                "avg_modules": 0,
                "module_counts": {MODULE_NAMES[m]: 0 for m in range(N_MODULES)},
                "roles": {"producer": 0, "carnivore": 0, "detritivore": 0, "omnivore": 0},
                "kills": 0, "total_kills": self.total_predation_kills,
            })

    def save_snapshot(self, output_dir):
        """Save JSON snapshot with organism data including module lists."""
        os.makedirs(output_dir, exist_ok=True)
        p = self.pop
        idx = self.rng.choice(p, min(p, 500), replace=False) if p > 500 else np.arange(p)
        orgs = [{
            "id": int(self.ids[i]), "row": int(self.rows[i]), "col": int(self.cols[i]),
            "energy": round(float(self.energy[i]), 2), "age": int(self.age[i]),
            "generation": int(self.generation[i]),
            "modules": [MODULE_NAMES[m] for m in range(N_MODULES) if self.module_present[i, m]],
        } for i in idx]
        s = self.stats_history[-1] if self.stats_history else {}
        with open(os.path.join(output_dir, f"snapshot_{self.timestep:06d}.json"), 'w') as f:
            json.dump({"timestep": self.timestep, "population": p, "organisms": orgs, "stats": s}, f)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(cfg=None):
    """
    Run the full simulation. Progress output shows:
      - Population, energy, generation, toxicity, decomposition
      - Ecological roles: producers, carnivores, detritivores, omnivores
      - New module counts: FORAGE, DEFENSE, DETOX
      - Predation kills per snapshot
      - Average modules per organism
    """
    cfg = cfg or Config()
    world = World(cfg)

    print(f"The Shimmering Field — Phase 3 Step 4: FORAGE + DEFENSE + DETOX")
    print(f"Grid: {cfg.grid_size}×{cfg.grid_size}  |  Pop: {cfg.initial_population}  |  Weights: {TOTAL_WEIGHT_PARAMS}")
    print(f"Module costs — PH:{MODULE_MAINTENANCE[M_PHOTO]+MODULE_EXPRESSION[M_PHOTO]:.2f}  "
          f"CH:{MODULE_MAINTENANCE[M_CHEMO]+MODULE_EXPRESSION[M_CHEMO]:.2f}  "
          f"CO:{MODULE_MAINTENANCE[M_CONSUME]+MODULE_EXPRESSION[M_CONSUME]:.2f}  "
          f"MV:{MODULE_MAINTENANCE[M_MOVE]+MODULE_EXPRESSION[M_MOVE]:.2f}  "
          f"FO:{MODULE_MAINTENANCE[M_FORAGE]+MODULE_EXPRESSION[M_FORAGE]:.2f}  "
          f"DE:{MODULE_MAINTENANCE[M_DEFENSE]+MODULE_EXPRESSION[M_DEFENSE]:.2f}  "
          f"DT:{MODULE_MAINTENANCE[M_DETOX]+MODULE_EXPRESSION[M_DETOX]:.2f}  "
          f"TP:{MODULE_MAINTENANCE[M_TOXPROD]+MODULE_EXPRESSION[M_TOXPROD]:.2f}")
    print(f"{'─' * 140}")

    start = time.time()
    for t in range(cfg.total_timesteps):
        world.update()

        if world.timestep % cfg.snapshot_interval == 0:
            world.save_snapshot(cfg.output_dir)
            s = world.stats_history[-1]
            el = time.time() - start
            r = s["roles"]
            mc = s["module_counts"]
            print(
                f"  t={s['t']:5d}  |  pop={s['pop']:5d}  |  e={s['avg_energy']:5.1f}  |  "
                f"gen={s['max_gen']:4d}  |  tox={s['toxic_mean']:.3f} dcp={s['decomp_mean']:.2f}  |  "
                f"prod={r['producer']:4d} carn={r['carnivore']:3d} detr={r['detritivore']:3d} omni={r['omnivore']:4d}  |  "
                f"FO={mc['FORAGE']:4d} DE={mc['DEFENSE']:4d} DT={mc['DETOX']:4d}  |  "
                f"kill={s['kills']:3d}  |  mod={s['avg_modules']:.2f}  |  {el:.1f}s"
            )

        if world.pop == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            break

    el = time.time() - start
    print(f"{'─' * 140}")
    print(f"Done in {el:.1f}s  |  Pop: {world.pop}  |  Total predation kills: {world.total_predation_kills}")

    if world.pop > 0:
        r = world.stats_history[-1]["roles"]
        mc = world.stats_history[-1]["module_counts"]
        print(f"Roles — Producers: {r['producer']}  Carnivores: {r['carnivore']}  "
              f"Detritivores: {r['detritivore']}  Omnivores: {r['omnivore']}")
        print(f"Modules — PH:{mc['PHOTO']} CH:{mc['CHEMO']} CO:{mc['CONSUME']} "
              f"MV:{mc['MOVE']} FO:{mc['FORAGE']} DE:{mc['DEFENSE']} DT:{mc['DETOX']} TP:{mc['TOXPROD']}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_summary.json"), 'w') as f:
        json.dump({"config": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                   "stats_history": world.stats_history}, f, indent=2)
    return world


if __name__ == "__main__":
    run_simulation()
