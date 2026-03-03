"""
The Shimmering Field — Phase 3 Step 8: Behavioral Hijacking
============================================================
Parasitic viral material overrides host behavior (Toxoplasma/cordyceps-style).

Not a new module — operates through active viral infection (lytic phase).
When an organism carries viral_load in the hijacking range (0.05-0.7),
the local viral genome encodes "hijack instructions" that override behavior:

  Light hijack  (load 0.05-0.2): subtle movement bias toward dense uninfected areas
  Medium hijack (load 0.2-0.5):  defense/immune suppression, forced movement
  Heavy hijack  (load 0.5-0.7):  host stops feeding, erratic movement, accelerated burst

Key mechanics:
  - Hijacked hosts move toward high-density, low-viral-particle areas (optimal for spread)
  - Host DEFENSE and VRESIST effectiveness suppressed proportional to hijack intensity
  - At heavy hijack, energy acquisition disabled → host becomes a viral delivery vehicle
  - Hijack intensity scales with local toxic stress (stressed hosts easier to control)
  - VRESIST module resists hijacking (immune_memory provides partial protection)

Failure mode setup:
  - Hijacking + reproductive manipulation = dual viral strategy
  - Population looks active but behavior is increasingly viral-directed
  - Collapse when hijacked organisms cluster and mass-burst simultaneously

Built on Phase 3 Step 7 (Reproductive Manipulation).
"""

import numpy as np
import json
import os
import time
from scipy.ndimage import uniform_filter, gaussian_filter


# ─────────────────────────────────────────────────────
# Module Definitions
# ─────────────────────────────────────────────────────

M_PHOTO   = 0
M_CHEMO   = 1
M_CONSUME = 2
M_MOVE    = 3
M_FORAGE  = 4
M_DEFENSE = 5
M_DETOX   = 6
M_TOXPROD = 7
M_VRESIST = 8
M_SOCIAL  = 9
M_MEDIATE = 10

N_MODULES = 11

MODULE_NAMES = [
    "PHOTO", "CHEMO", "CONSUME", "MOVE", "FORAGE",
    "DEFENSE", "DETOX", "TOXPROD", "VRESIST", "SOCIAL", "MEDIATE"
]

MODULE_WEIGHT_SIZES = np.array([
    4,  # PHOTO:   [efficiency, toxic_tolerance, light_sensitivity, storage_rate]
    4,  # CHEMO:   [efficiency, specificity, saturation_threshold, gradient_follow]
    4,  # CONSUME: [prey_selectivity, handling_efficiency, decomp_preference, aggression]
    8,  # MOVE:    [light_seek, density_avoid, nutrient_stay, chemo_toxic_seek,
        #           random_wt, stay_tend, light_str, toxic_response]
    4,  # FORAGE:  [extraction_eff, resource_discrim, storage_cap, cooperative_signal]
    4,  # DEFENSE: [shell, camouflage, size_invest, counter_attack]
    4,  # DETOX:   [detox_eff, toxin_tolerance, conversion_rate, selective_uptake]
    0,  # TOXPROD: (no evolvable weights)
    4,  # VRESIST: [recognition_specificity, suppression_strength, resistance_breadth, immune_memory]
    4,  # SOCIAL:  [identity_signal, compatibility_assessment, approach_avoidance, relationship_strength]
    4,  # MEDIATE: [pollination_drive, route_memory, network_coordination, reward_sensitivity]
], dtype=np.int32)

MODULE_WEIGHT_OFFSETS = np.zeros(N_MODULES, dtype=np.int32)
_off = 0
for _m in range(N_MODULES):
    MODULE_WEIGHT_OFFSETS[_m] = _off
    _off += MODULE_WEIGHT_SIZES[_m]
TOTAL_MODULE_WEIGHTS = _off  # 32

N_STANDALONE_PARAMS = 4
STANDALONE_OFFSET = TOTAL_MODULE_WEIGHTS
TOTAL_WEIGHT_PARAMS = TOTAL_MODULE_WEIGHTS + N_STANDALONE_PARAMS  # 36

SP_TRANSFER_RECEPTIVITY = 0
SP_TRANSFER_SELECTIVITY = 1
SP_VIRAL_RESISTANCE = 2
SP_LYSO_SUPPRESSION = 3

# Module costs: [maintenance, expression]
#                PH    CH    CO    MV    FO    DE    DT    TP    VR    SO    ME
MODULE_MAINTENANCE = np.array([0.20, 0.25, 0.12, 0.08, 0.12, 0.18, 0.20, 0.03, 0.15, 0.10, 0.12])
MODULE_EXPRESSION  = np.array([0.10, 0.12, 0.08, 0.06, 0.06, 0.10, 0.10, 0.02, 0.08, 0.05, 0.08])
BASE_MAINTENANCE = 0.05
# Cost examples:
#   PHOTO+MOVE+TOXPROD                    = 0.54
#   + FORAGE (efficient producer)         = 0.72
#   + DEFENSE (armored producer)          = 1.00
#   + CONSUME (all-in generalist)         = 1.20
#   + DETOX (toxic zone specialist)       = 1.50
#   + VRESIST (immune system)             = 1.73
#   + SOCIAL (relational)                 = 1.88
#   + MEDIATE (pollinator)                = 2.08

GAINABLE_MODULES = [M_CHEMO, M_CONSUME, M_MOVE, M_FORAGE, M_DEFENSE, M_DETOX, M_VRESIST, M_SOCIAL, M_MEDIATE]


# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────

class Config:
    grid_size = 128
    light_max = 1.0
    light_min = 0.05
    zone_count = 8

    # Toxicity
    toxic_decay_rate = 0.01
    toxic_diffusion_rate = 0.06
    toxic_production_rate = 0.015
    toxic_threshold_low = 0.3
    toxic_threshold_medium = 0.8
    toxic_threshold_high = 1.5
    toxic_damage_medium = 1.5
    toxic_damage_high = 5.0
    toxic_photo_penalty = 1.0

    # Nutrients
    nutrient_base_rate = 0.002
    nutrient_from_decomp = 0.4
    nutrient_max = 3.0

    # Population
    initial_population = 200
    initial_consumer_fraction = 0.10
    initial_detritivore_fraction = 0.06
    energy_initial = 40.0
    energy_max = 150.0
    energy_reproduction_threshold = 80.0
    energy_reproduction_cost = 40.0
    energy_movement_cost = 0.2
    max_age = 200
    min_reproduction_age = 8
    offspring_distance = 5
    # Density-dependent reproduction: threshold scales up with local crowding
    repro_density_penalty = 5.0  # extra energy needed per neighbor

    # Energy production
    photosynthesis_base = 3.0
    chemosynthesis_base = 2.2
    scavenge_base = 2.5

    # Metabolic interference (asymmetric)
    producer_consume_penalty = 0.30  # omnivores hunt at 30% effectiveness
    consume_producer_penalty = 0.85  # omnivores produce at 85% effectiveness
    consumer_specialist_bonus = 0.5  # +50% for obligate consumers

    # Predation
    predation_check_interval = 1
    predation_base_success = 0.14
    predation_energy_fraction = 0.55
    predation_hunt_radius_base = 1
    predation_hunt_radius_mobile = 2
    predation_max_kills_per_cell = 1  # prevents local extinction cascades
    predator_satiation = 0.7  # probability a fed predator skips next hunt
    # Herbivore/carnivore gradient (via prey_selectivity weight)
    herbivore_producer_bonus = 0.10   # success bonus when herbivore targets producer
    carnivore_consumer_bonus = 0.15   # success bonus when carnivore targets consumer (harder prey)
    herbivore_energy_mult = 1.20      # herbivores extract MORE (abundant easy prey, eat often)
    carnivore_energy_mult = 0.90      # carnivores extract less per kill (prey already depleted energy fleeing)

    # Decomposition
    decomp_death_deposit = 2.0
    decomp_decay_rate = 0.998
    decomp_diffusion_rate = 0.008  # light local diffusion for feeding zone
    decomp_diffusion_interval = 3
    decomp_scent_sigma = 5  # gaussian blur for navigation scent
    decomp_scent_interval = 3  # recompute scent every N steps

    # FORAGE module
    forage_extraction_bonus = 0.25     # max +25% resource extraction
    forage_storage_bonus = 30.0        # max extra energy capacity
    forage_cooperative_radius = 1      # cells to check for cooperators
    forage_cooperative_bonus = 0.08    # energy bonus per cooperating neighbor

    # DEFENSE module
    defense_shell_max = 0.55           # max predation probability reduction
    defense_camouflage_max = 0.35      # max chance predator skips target
    defense_counter_damage = 5.0       # energy damage to predator on failed hunt
    defense_size_cost_mult = 1.3       # size investment increases module cost

    # DETOX module
    detox_rate_max = 0.08              # max fraction of local toxin removed per step
    detox_tolerance_bonus = 0.6        # max addition to toxic damage threshold
    detox_energy_conversion = 0.4      # fraction of metabolized toxin → energy
    detox_environment_effect = 0.5     # fraction of detox that cleans the environment

    # VRESIST module
    vresist_base_resistance = 0.6      # base infection resistance with VRESIST (vs ~0.3 standalone)
    vresist_specificity_bonus = 0.25   # extra resistance from specificity (familiar strains)
    vresist_breadth_bonus = 0.15       # extra resistance from breadth (all strains)
    vresist_suppression_max = 0.85     # max lysogenic suppression probability
    vresist_memory_boost = 0.3         # resistance boost after surviving infection
    vresist_memory_decay = 0.995       # how fast immune memory decays per step

    # SOCIAL module
    social_signal_radius = 1           # cells for social signal deposit (0 = own cell only)
    social_compatibility_bonus = 0.15  # energy bonus per compatible neighbor
    social_incompatibility_penalty = 0.05  # energy penalty per incompatible neighbor
    social_relationship_growth = 0.02  # relationship score growth per step near compatible
    social_relationship_decay = 0.998  # relationship score decay per step
    social_update_interval = 2         # how often social field updates

    # MEDIATE module (pollination/dispersal)
    mediate_repro_bonus = 0.25         # reproduction threshold reduction per mediator nearby
    mediate_radius = 2                 # cells within which mediators provide bonus
    mediate_energy_reward = 0.5        # energy reward to mediator per facilitated reproduction
    mediate_network_decay = 0.97       # mediator field decay per step
    mediate_update_interval = 2        # how often mediator field updates

    # Nutrient cycling (emergent from module interactions)
    nutrient_detox_deposit = 0.3       # fraction of detox byproduct that becomes nutrients
    nutrient_consume_deposit = 0.15    # nutrient release from consume processing
    nutrient_death_per_module = 0.3    # nutrients deposited per module on death
    nutrient_forage_coop_boost = 0.1   # local nutrient regen boost per FORAGE cooperator

    # Reproductive manipulation (Wolbachia-style, via lysogenic genome)
    repro_manip_threshold = 0.3        # min lysogenic_strength for manipulation to activate
    repro_manip_trait_bias = 0.15      # max weight blend toward lysogenic genome in offspring
    repro_manip_receptivity_boost = 0.4  # boost to offspring's transfer_receptivity param
    repro_manip_viability_cost = 3.0   # energy penalty to divergent offspring
    repro_manip_divergence_thresh = 0.5  # weight distance above which viability penalty applies
    repro_manip_saturation = 0.7       # population lysogenic fraction where self-limiting kicks in

    # Behavioral hijacking (Toxoplasma/cordyceps-style, via lytic viral load)
    hijack_load_min = 0.05             # minimum viral_load for hijack effects
    hijack_load_heavy = 0.5            # viral_load above which heavy hijack kicks in
    hijack_defense_suppress = 0.6      # max defense suppression at full hijack
    hijack_vresist_suppress = 0.4      # max VRESIST suppression at full hijack
    hijack_energy_suppress = 0.7       # fraction of energy acquisition blocked at heavy hijack
    hijack_density_seek = 3.0          # movement weight toward high-density areas (spread virus)
    hijack_stress_amplifier = 1.5      # toxic stress multiplier on hijack intensity

    # Evolution
    mutation_rate = 0.08
    module_gain_rate = 0.005       # slightly higher — more modules available now
    module_lose_rate = 0.003

    # Horizontal transfer
    transfer_check_interval = 5
    transfer_blend_rate_recent = 0.10
    transfer_blend_rate_intermediate = 0.18
    transfer_blend_rate_ancient = 0.30
    decomp_fragment_decay = 0.005
    decomp_fragment_diffusion = 0.008
    module_transfer_rate = 0.015
    sedimentation_rate_recent = 0.005
    sedimentation_rate_intermediate = 0.002
    ancient_decay_rate = 0.001
    stratum_access_medium = 0.3
    stratum_access_high = 0.8

    # Viral system
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

    # Simulation
    total_timesteps = 10000
    snapshot_interval = 100
    output_dir = "output_p3s8"
    random_seed = 42


# ─────────────────────────────────────────────────────
# World
# ─────────────────────────────────────────────────────

class World:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.rng = np.random.default_rng(c.random_seed)
        self.timestep = 0
        self.next_id = 0
        N = c.grid_size

        # ── Environment layers ──
        self.light = np.linspace(c.light_max, c.light_min, N)[:, None] * np.ones((1, N))
        self.toxic = self.rng.uniform(0.0, 0.005, (N, N))
        self.nutrients = self.rng.uniform(0.02, 0.08, (N, N))
        self.decomposition = np.zeros((N, N))
        self.decomp_scent = np.zeros((N, N))
        self.density = np.zeros((N, N), dtype=np.int32)

        # Zone map: heterogeneous toxic production rates
        zone_size = N // c.zone_count
        self.zone_map = np.ones((N, N))
        for zi in range(c.zone_count):
            for zj in range(c.zone_count):
                r0, r1 = zi * zone_size, (zi + 1) * zone_size
                c0, c1 = zj * zone_size, (zj + 1) * zone_size
                self.zone_map[r0:r1, c0:c1] = self.rng.choice(
                    [0.3, 0.5, 0.7, 1.0, 1.0, 1.2, 1.5, 2.0])
        self.zone_map = uniform_filter(self.zone_map, size=8)

        # Stratified fragment pools (horizontal gene transfer)
        self.strata_pool = {s: np.zeros((N, N, TOTAL_WEIGHT_PARAMS))
                           for s in ("recent", "intermediate", "ancient")}
        self.strata_weight = {s: np.zeros((N, N))
                             for s in ("recent", "intermediate", "ancient")}
        self.strata_modules = {s: np.zeros((N, N, N_MODULES))
                              for s in ("recent", "intermediate", "ancient")}

        # Viral system
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

        # Social signal field: two channels [producer_signal, consumer_signal]
        self.social_field = np.zeros((N, N, 2))

        # Mediator field: pollination/dispersal service availability
        self.mediator_field = np.zeros((N, N))

        # ── Organisms ──
        self._init_organisms()

        # Stats
        self.total_transfers = 0
        self.transfers_by_stratum = {"recent": 0, "intermediate": 0, "ancient": 0}
        self.total_lytic_deaths = 0
        self.total_lysogenic_integrations = 0
        self.total_lysogenic_activations = 0
        self.total_predation_kills = 0
        self.stats_history = []
        self.total_manipulated_births = 0
        self.total_hijacked_steps = 0  # cumulative organism-steps under hijack

    def _init_organisms(self):
        """Seed initial population: producers, carnivores, detritivores."""
        c = self.cfg
        N = c.grid_size
        pop = c.initial_population
        mid = N // 2

        self.rows = self.rng.integers(0, N, size=pop).astype(np.int64)
        self.cols = self.rng.integers(0, N, size=pop).astype(np.int64)
        self.energy = np.full(pop, c.energy_initial)
        self.age = np.zeros(pop, dtype=np.int32)
        self.generation = np.zeros(pop, dtype=np.int32)
        self.ids = np.arange(pop, dtype=np.int64)
        self.parent_ids = np.full(pop, -1, dtype=np.int64)
        self.next_id = pop

        # Default genome: PHOTO + MOVE + TOXPROD
        self.module_present = np.zeros((pop, N_MODULES), dtype=bool)
        self.module_present[:, M_PHOTO] = True
        self.module_present[:, M_MOVE] = True
        self.module_present[:, M_TOXPROD] = True
        self.module_active = self.module_present.copy()

        self.weights = self.rng.normal(0, 0.5, (pop, TOTAL_WEIGHT_PARAMS))

        # Seed carnivores: CONSUME+MOVE+TOXPROD, no PHOTO, high aggression
        n_carn = int(pop * c.initial_consumer_fraction)
        carn_idx = self.rng.choice(pop, n_carn, replace=False)
        self.module_present[carn_idx, M_PHOTO] = False
        self.module_present[carn_idx, M_CONSUME] = True
        self.module_active[carn_idx] = self.module_present[carn_idx]
        self.rows[carn_idx] = self.rng.integers(mid - 20, mid + 20, n_carn).astype(np.int64)
        self.cols[carn_idx] = self.rng.integers(mid - 20, mid + 20, n_carn).astype(np.int64)
        self.energy[carn_idx] = c.energy_initial * 1.5
        # Seed CONSUME weights: high aggression, low decomp_pref
        consume_off = int(MODULE_WEIGHT_OFFSETS[M_CONSUME])
        self.weights[carn_idx, consume_off + 3] = self.rng.uniform(1.0, 2.5, n_carn)   # aggression
        self.weights[carn_idx, consume_off + 2] = self.rng.uniform(-3.0, -1.5, n_carn)  # decomp_pref

        # Seed detritivores: CONSUME+MOVE+TOXPROD, no PHOTO, high decomp_pref
        remaining = np.setdiff1d(np.arange(pop), carn_idx)
        n_detr = int(pop * c.initial_detritivore_fraction)
        detr_idx = self.rng.choice(remaining, n_detr, replace=False)
        self.module_present[detr_idx, M_PHOTO] = False
        self.module_present[detr_idx, M_CONSUME] = True
        self.module_active[detr_idx] = self.module_present[detr_idx]
        self.rows[detr_idx] = self.rng.integers(mid - 15, mid + 15, n_detr).astype(np.int64)
        self.cols[detr_idx] = self.rng.integers(mid - 15, mid + 15, n_detr).astype(np.int64)
        self.energy[detr_idx] = c.energy_initial * 1.5
        self.weights[detr_idx, consume_off + 2] = self.rng.uniform(1.5, 3.0, n_detr)   # decomp_pref
        self.weights[detr_idx, consume_off + 3] = self.rng.uniform(-2.0, -0.5, n_detr)  # aggression

        # Seed decomp near center so detritivores have initial food
        self.decomposition[mid-10:mid+10, mid-10:mid+10] += self.rng.uniform(0.5, 2.0, (20, 20))

        # Viral/HGT state
        self.transfer_count = np.zeros(pop, dtype=np.int32)
        self.viral_load = np.zeros(pop, dtype=np.float64)
        self.lysogenic_strength = np.zeros(pop, dtype=np.float64)
        self.lysogenic_genome = np.zeros((pop, TOTAL_WEIGHT_PARAMS), dtype=np.float64)

        # VRESIST: immune memory from surviving infections
        self.immune_experience = np.zeros(pop, dtype=np.float64)
        # SOCIAL: relationship accumulation with compatible neighbors
        self.relationship_score = np.zeros(pop, dtype=np.float64)

    @property
    def pop(self):
        return len(self.rows)

    # ── Helpers ──

    def _module_weights(self, module_id):
        off = int(MODULE_WEIGHT_OFFSETS[module_id])
        sz = int(MODULE_WEIGHT_SIZES[module_id])
        return self.weights[:, off:off+sz] if sz > 0 else None

    def _standalone_params(self):
        return self.weights[:, STANDALONE_OFFSET:]

    def _compute_module_costs(self):
        maint = self.module_present.astype(np.float64) @ MODULE_MAINTENANCE
        expr = self.module_active.astype(np.float64) @ MODULE_EXPRESSION
        return BASE_MAINTENANCE + maint + expr

    def _effective_energy_max(self):
        """FORAGE storage_cap: foragers can store extra energy."""
        c = self.cfg
        effective_max = np.full(self.pop, c.energy_max)
        has_forage = self.module_active[:, M_FORAGE]
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            storage = c.forage_storage_bonus * (1.0 / (1.0 + np.exp(-fw[:, 2])))
            effective_max += storage * has_forage
        return effective_max

    def _compute_total_costs(self):
        """Module costs + DEFENSE size investment surcharge."""
        c = self.cfg
        costs = self._compute_module_costs()
        has_defense = self.module_active[:, M_DEFENSE]
        if has_defense.any():
            dw = self._module_weights(M_DEFENSE)
            size_invest = np.maximum(0, np.tanh(dw[:, 2])) * has_defense
            costs += size_invest * (c.defense_size_cost_mult - 1.0) * (
                MODULE_MAINTENANCE[M_DEFENSE] + MODULE_EXPRESSION[M_DEFENSE])
        return costs

    _ORG_FIELDS = [
        "rows", "cols", "energy", "age", "generation", "ids", "parent_ids",
        "weights", "module_present", "module_active", "transfer_count",
        "viral_load", "lysogenic_strength", "lysogenic_genome",
        "immune_experience", "relationship_score",
    ]

    def _filter_organisms(self, mask):
        for name in self._ORG_FIELDS:
            setattr(self, name, getattr(self, name)[mask])

    def _append_organisms(self, d):
        for name in self._ORG_FIELDS:
            setattr(self, name, np.concatenate([getattr(self, name), d[name]]))

    # ── Environment ──

    def _update_environment(self):
        c = self.cfg

        # Toxic diffusion + decay
        p = np.pad(self.toxic, 1, mode='edge')
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        self.toxic += c.toxic_diffusion_rate * (nb - self.toxic)
        self.toxic *= (1.0 - c.toxic_decay_rate / np.maximum(0.3, self.zone_map))

        # Toxic production by organisms
        if self.pop > 0:
            has_toxprod = self.module_active[:, M_TOXPROD]
            rates = np.where(has_toxprod, c.toxic_production_rate, 0.0)
            np.add.at(self.toxic, (self.rows, self.cols),
                      rates * self.zone_map[self.rows, self.cols])

        # Nutrients: slow regeneration + very slow decomp→nutrient conversion
        self.nutrients += c.nutrient_base_rate
        xfer = self.decomposition * 0.001
        self.nutrients += xfer
        self.decomposition -= xfer

        # Decomp decay
        self.decomposition *= c.decomp_decay_rate

        # Decomp local diffusion: spreads food 2-3 cells from death sites
        if self.timestep % c.decomp_diffusion_interval == 0:
            dp = np.pad(self.decomposition, 1, mode='edge')
            dnb = (dp[:-2, 1:-1] + dp[2:, 1:-1] + dp[1:-1, :-2] + dp[1:-1, 2:]) / 4.0
            self.decomposition += c.decomp_diffusion_rate * (dnb - self.decomposition)

        # Scent layer: gaussian blur for detritivore navigation (cached)
        if self.timestep % c.decomp_scent_interval == 0:
            self.decomp_scent = gaussian_filter(
                self.decomposition, sigma=c.decomp_scent_sigma, mode='constant')

        # Stratified sedimentation
        self._update_strata()

        # Viral diffusion + decay
        self._update_viral_environment()

        # Clamp
        np.clip(self.toxic, 0, 5.0, out=self.toxic)
        np.clip(self.nutrients, 0, c.nutrient_max, out=self.nutrients)
        np.clip(self.decomposition, 0, 30.0, out=self.decomposition)
        np.clip(self.viral_particles, 0, 20.0, out=self.viral_particles)

    def _update_strata(self):
        c = self.cfg
        # Recent → Intermediate
        xfer_w = self.strata_weight["recent"] * c.sedimentation_rate_recent
        self.strata_weight["recent"] -= xfer_w
        new_iw = self.strata_weight["intermediate"] + xfer_w
        blend = xfer_w / np.maximum(new_iw, 1e-8)
        for key in ("strata_pool", "strata_modules"):
            pool = getattr(self, key)
            pool["intermediate"] = (pool["intermediate"] * (1.0 - blend[:, :, None])
                                   + pool["recent"] * blend[:, :, None])
        self.strata_weight["intermediate"] = new_iw

        # Intermediate → Ancient
        xfer_w2 = self.strata_weight["intermediate"] * c.sedimentation_rate_intermediate
        self.strata_weight["intermediate"] -= xfer_w2
        new_aw = self.strata_weight["ancient"] + xfer_w2
        blend2 = xfer_w2 / np.maximum(new_aw, 1e-8)
        for key in ("strata_pool", "strata_modules"):
            pool = getattr(self, key)
            pool["ancient"] = (pool["ancient"] * (1.0 - blend2[:, :, None])
                              + pool["intermediate"] * blend2[:, :, None])
        self.strata_weight["ancient"] = new_aw

        # Decay
        self.strata_weight["recent"] *= (1.0 - c.decomp_fragment_decay)
        self.strata_weight["intermediate"] *= (1.0 - c.decomp_fragment_decay * 0.5)
        self.strata_weight["ancient"] *= (1.0 - c.ancient_decay_rate)

        # Fragment diffusion (periodic)
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
        self.density[:] = 0
        if self.pop > 0:
            np.add.at(self.density, (self.rows, self.cols), 1)

    # ── Sensing ──

    def _sense_local(self):
        rows, cols = self.rows, self.cols
        N = self.cfg.grid_size
        ru = np.clip(rows - 1, 0, N - 1)
        rd = np.clip(rows + 1, 0, N - 1)
        cl = np.clip(cols - 1, 0, N - 1)
        cr = np.clip(cols + 1, 0, N - 1)
        return np.column_stack([
            self.light[rows, cols],                                          # 0: local_light
            self.toxic[rows, cols],                                          # 1: local_toxic
            self.nutrients[rows, cols],                                      # 2: local_nutrients
            self.density[rows, cols].astype(np.float64),                     # 3: local_density
            self.light[ru, cols] - self.light[rd, cols],                     # 4: light_grad_y
            self.light[rows, cr] - self.light[rows, cl],                     # 5: light_grad_x
            self.toxic[ru, cols] - self.toxic[rd, cols],                     # 6: toxic_grad_y
            self.toxic[rows, cr] - self.toxic[rows, cl],                     # 7: toxic_grad_x
            self.decomposition[rows, cols],                                  # 8: local_decomp (raw)
            self.density[ru, cols].astype(np.float64)
                - self.density[rd, cols].astype(np.float64),                 # 9: density_grad_y
            self.density[rows, cr].astype(np.float64)
                - self.density[rows, cl].astype(np.float64),                 # 10: density_grad_x
            self.decomp_scent[ru, cols] - self.decomp_scent[rd, cols],       # 11: scent_grad_y
            self.decomp_scent[rows, cr] - self.decomp_scent[rows, cl],       # 12: scent_grad_x
            self.nutrients[ru, cols] - self.nutrients[rd, cols],              # 13: nutrient_grad_y
            self.nutrients[rows, cr] - self.nutrients[rows, cl],              # 14: nutrient_grad_x
            self.social_field[rows, cols, 0],                                 # 15: social_prod_signal
            self.social_field[rows, cols, 1],                                 # 16: social_cons_signal
            self.social_field[ru, cols, 0] - self.social_field[rd, cols, 0],  # 17: social_prod_grad_y
            self.social_field[rows, cr, 0] - self.social_field[rows, cl, 0],  # 18: social_prod_grad_x
            self.social_field[ru, cols, 1] - self.social_field[rd, cols, 1],  # 19: social_cons_grad_y
            self.social_field[rows, cr, 1] - self.social_field[rows, cl, 1],  # 20: social_cons_grad_x
            self.mediator_field[rows, cols],                                   # 21: local_mediator
        ])

    # ── Energy Acquisition ──

    def _acquire_energy(self, readings):
        c = self.cfg
        n = self.pop
        total_gain = np.zeros(n)

        local_light = readings[:, 0]
        local_toxic = readings[:, 1]
        local_density = readings[:, 3]
        local_decomp = readings[:, 8]

        # Metabolic interference masks
        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]
        is_omnivore = has_producer & has_consume
        is_specialist = has_consume & ~has_producer

        prod_mult = np.where(is_omnivore, c.consume_producer_penalty, 1.0)
        cons_mult = np.where(is_omnivore, c.producer_consume_penalty, 1.0)
        spec_mult = np.where(is_specialist, 1.0 + c.consumer_specialist_bonus, 1.0)

        # ── FORAGE bonus: extraction efficiency ──
        has_forage = self.module_active[:, M_FORAGE]
        forage_mult = np.ones(n)
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            extract_eff = c.forage_extraction_bonus * (1.0 / (1.0 + np.exp(-fw[:, 0])))
            forage_mult = 1.0 + extract_eff * has_forage

        # PHOTO
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

        # CHEMO
        has_chemo = self.module_active[:, M_CHEMO]
        if has_chemo.any():
            ch = self._module_weights(M_CHEMO)
            eff = c.chemosynthesis_base * (1.0 + 0.3 * np.tanh(ch[:, 0]))
            sat = 1.0 + np.abs(ch[:, 2]) * 0.5
            toxic_factor = local_toxic / (local_toxic + sat)
            sh = 1.0 / np.maximum(1.0, local_density * 0.4)
            total_gain += (eff * toxic_factor * sh * prod_mult * forage_mult) * has_chemo

        # CONSUME (scavenging component)
        if has_consume.any():
            uw = self._module_weights(M_CONSUME)
            decomp_pref = 1.0 / (1.0 + np.exp(-uw[:, 2]))
            detr_bonus = 1.0 + 1.0 * decomp_pref
            eff = c.scavenge_base * (0.3 + 0.7 * decomp_pref) * cons_mult * spec_mult * detr_bonus
            handling = 1.0 + 0.3 * np.tanh(uw[:, 1])
            extract_rate = 0.10 + 0.10 * decomp_pref
            intake_cap = 1.5 + 1.5 * decomp_pref
            available = np.minimum(local_decomp * extract_rate, intake_cap)
            scavenge_gain = eff * handling * available * forage_mult
            total_gain += scavenge_gain * has_consume

            consumed = np.minimum(local_decomp * 0.15, scavenge_gain * 0.05) * has_consume
            np.add.at(self.decomposition, (self.rows, self.cols), -consumed)
            np.clip(self.decomposition, 0, 30.0, out=self.decomposition)

        # ── FORAGE cooperative signal: nearby foragers boost each other ──
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            coop_strength = 1.0 / (1.0 + np.exp(-fw[:, 3]))
            forage_density = np.zeros_like(self.density)
            fidx = np.where(has_forage)[0]
            if len(fidx) > 0:
                np.add.at(forage_density, (self.rows[fidx], self.cols[fidx]), 1)
            local_foragers = forage_density[self.rows, self.cols].astype(np.float64)
            coop_bonus = (np.maximum(0, local_foragers - 1) * c.forage_cooperative_bonus
                         * coop_strength * has_forage)
            total_gain += coop_bonus

        return total_gain

    # ── Predation (optimized) ──

    def _predation(self):
        """Predation with DEFENSE integration, per-cell kill cap, satiation.
        Optimized: pre-rolled randomness, flat spatial index."""
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

        uw = self._module_weights(M_CONSUME)
        aggression = 1.0 / (1.0 + np.exp(-uw[:, 3]))
        decomp_pref = 1.0 / (1.0 + np.exp(-uw[:, 2]))
        prey_selectivity = 1.0 / (1.0 + np.exp(-uw[:, 0]))  # 0=herbivore, 1=carnivore
        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        is_specialist = has_consume & ~has_producer
        is_omnivore = has_consume & has_producer

        # Precompute DEFENSE stats
        shell_val = np.zeros(n)
        camo_val = np.zeros(n)
        counter_val = np.zeros(n)
        if has_defense.any():
            dw = self._module_weights(M_DEFENSE)
            shell_val = c.defense_shell_max * (1.0 / (1.0 + np.exp(-dw[:, 0]))) * has_defense
            camo_val = c.defense_camouflage_max * (1.0 / (1.0 + np.exp(-dw[:, 1]))) * has_defense
            counter_val = c.defense_counter_damage * np.tanh(np.maximum(0, dw[:, 3])) * has_defense

        # Behavioral hijacking suppresses defense (hijacked hosts can't defend)
        if self.hijack_intensity.any():
            suppress = self.hijack_intensity * c.hijack_defense_suppress
            shell_val *= (1.0 - suppress)
            camo_val *= (1.0 - suppress)
            counter_val *= (1.0 - suppress)

        # Pre-roll camouflage outcomes: each organism's camo roll for this step
        camo_rolls = self.rng.random(n)  # compare against camo_val[j] — hidden if roll < camo_val

        # Pre-compute hunt success base for all predators
        hunt_skill = aggression * (1.0 - 0.5 * decomp_pref)
        base_prob = c.predation_base_success + hunt_skill * 0.3
        base_prob += np.where(has_move, 0.08, 0.0)
        base_prob += np.where(is_specialist, 0.15, 0.0)
        omni_mult = np.where(is_omnivore, c.producer_consume_penalty, 1.0)

        # Energy fraction for specialists
        frac_base = np.where(is_specialist,
                             c.predation_energy_fraction * (1.0 + c.consumer_specialist_bonus),
                             c.predation_energy_fraction)

        # Flat spatial index: cell_key → sorted organism indices
        cell_key = self.rows * N + self.cols
        sort_idx = np.argsort(cell_key)
        sorted_keys = cell_key[sort_idx]
        unique_cells, cell_starts, cell_counts = np.unique(
            sorted_keys, return_index=True, return_counts=True)
        # Fast lookup: cell → (start, count) in sort_idx
        cell_start_map = np.full(N * N, -1, dtype=np.int32)
        cell_count_map = np.zeros(N * N, dtype=np.int32)
        cell_start_map[unique_cells] = cell_starts.astype(np.int32)
        cell_count_map[unique_cells] = cell_counts.astype(np.int32)

        kill_mask = np.zeros(n, dtype=bool)
        cell_kill_count = np.zeros(N * N, dtype=np.int32)
        satiated = np.zeros(n, dtype=bool)
        kills = 0

        predator_idx = np.where(predator_mask)[0]
        self.rng.shuffle(predator_idx)

        # Pre-roll hunt and satiation outcomes
        hunt_rolls = self.rng.random(len(predator_idx))
        sat_rolls = self.rng.random(len(predator_idx))

        for pi_idx, pi in enumerate(predator_idx):
            if kill_mask[pi] or satiated[pi]:
                continue

            pr, pc = int(self.rows[pi]), int(self.cols[pi])
            R = c.predation_hunt_radius_mobile if has_move[pi] else c.predation_hunt_radius_base

            # Gather candidates from nearby cells using flat lookup
            candidates = []
            r_lo, r_hi = max(0, pr - R), min(N - 1, pr + R)
            c_lo, c_hi = max(0, pc - R), min(N - 1, pc + R)
            for cr in range(r_lo, r_hi + 1):
                for cc_iter in range(c_lo, c_hi + 1):
                    ck = cr * N + cc_iter
                    if cell_kill_count[ck] >= c.predation_max_kills_per_cell:
                        continue
                    cs = cell_start_map[ck]
                    if cs < 0:
                        continue
                    cnt = cell_count_map[ck]
                    for si in range(cs, cs + cnt):
                        j = sort_idx[si]
                        if j != pi and not kill_mask[j]:
                            # Camouflage: pre-rolled
                            if camo_val[j] > 0 and camo_rolls[j] < camo_val[j]:
                                continue
                            candidates.append(j)

            if not candidates:
                continue

            # Target selection: weight by energy AND prey type preference
            cands = np.array(candidates)
            target_e = np.maximum(self.energy[cands], 1.0)
            inv_e = 1.0 / target_e  # prefer weaker prey

            # Prey type preference: herbivores prefer producers, carnivores prefer consumers
            target_is_producer = has_producer[cands].astype(np.float64)
            # prey_selectivity[pi]: 0=herbivore, 1=carnivore
            # herbivore (low selectivity) → weight producers higher
            # carnivore (high selectivity) → weight consumers higher
            ps = prey_selectivity[pi]
            type_weight = np.where(target_is_producer,
                                   1.0 + (1.0 - ps) * 2.0,  # herbivore bonus for producers
                                   1.0 + ps * 2.0)            # carnivore bonus for consumers
            sel_weight = inv_e * type_weight
            sel_weight /= sel_weight.sum()
            ti = cands[self.rng.choice(len(cands), p=sel_weight)]

            # Success probability — bonus for matching prey type
            target_is_prod = has_producer[ti]
            type_match_bonus = 0.0
            if target_is_prod and ps < 0.5:
                type_match_bonus = c.herbivore_producer_bonus * (1.0 - ps)
            elif not target_is_prod and ps > 0.5:
                type_match_bonus = c.carnivore_consumer_bonus * ps

            target_difficulty = np.clip(self.energy[ti] / c.energy_max, 0.1, 1.0)
            prob = (base_prob[pi] - target_difficulty * 0.08) * omni_mult[pi] + type_match_bonus
            prob -= shell_val[ti]
            prob = np.clip(prob, 0.01, 0.60)

            if hunt_rolls[pi_idx] < prob:
                # Energy gain modulated by predator type
                energy_mult = 1.0
                if ps < 0.5:
                    energy_mult = c.herbivore_energy_mult  # herbivores extract more (abundant easy prey)
                elif ps >= 0.5:
                    energy_mult = c.carnivore_energy_mult  # carnivores extract less (prey depleted fleeing)
                gained = max(0.0, self.energy[ti]) * frac_base[pi] * energy_mult
                self.energy[pi] = min(self.energy[pi] + gained, c.energy_max)
                self.energy[ti] = -1.0
                kill_mask[ti] = True
                cell_kill_count[int(self.rows[ti]) * N + int(self.cols[ti])] += 1
                kills += 1
                if sat_rolls[pi_idx] < c.predator_satiation:
                    satiated[pi] = True
            else:
                # Counter-attack on failed hunt
                if counter_val[ti] > 0:
                    self.energy[pi] -= counter_val[ti]

        self.total_predation_kills += kills
        return kills

    # ── Toxic Damage ──

    def _apply_toxic_damage(self, readings):
        c = self.cfg
        lt = readings[:, 1]
        dmg = np.zeros(self.pop)

        # CHEMO tolerance (existing)
        has_chemo = self.module_active[:, M_CHEMO]
        chemo_reduction = np.ones(self.pop)
        if has_chemo.any():
            ch = self._module_weights(M_CHEMO)
            tolerance = 0.3 + 0.4 / (1.0 + np.exp(-ch[:, 1]))
            chemo_reduction[has_chemo] = 1.0 - tolerance[has_chemo]

        # DETOX tolerance (new — raises damage thresholds)
        has_detox = self.module_active[:, M_DETOX]
        threshold_boost = np.zeros(self.pop)
        if has_detox.any():
            dtw = self._module_weights(M_DETOX)
            threshold_boost = (c.detox_tolerance_bonus
                              * (1.0 / (1.0 + np.exp(-dtw[:, 1])))
                              * has_detox)

        effective = lt * chemo_reduction
        eff_thresh_med = c.toxic_threshold_medium + threshold_boost
        eff_thresh_high = c.toxic_threshold_high + threshold_boost

        m = effective > eff_thresh_med
        if m.any():
            dmg[m] += (effective[m] - eff_thresh_med[m]) * c.toxic_damage_medium
        h = effective > eff_thresh_high
        if h.any():
            dmg[h] += (effective[h] - eff_thresh_high[h]) * c.toxic_damage_high
        self.energy -= dmg

    def _apply_detox(self):
        """DETOX module: metabolize environmental toxins for energy."""
        c = self.cfg
        has_detox = self.module_active[:, M_DETOX]
        if not has_detox.any():
            return

        dtw = self._module_weights(M_DETOX)
        detox_eff = c.detox_rate_max * (1.0 / (1.0 + np.exp(-dtw[:, 0])))
        # selective_uptake: prefer certain toxin concentrations (Michaelis-Menten style)
        selectivity = 1.0 + np.abs(dtw[:, 3]) * 0.5
        local_toxic = self.toxic[self.rows, self.cols]
        uptake_factor = local_toxic / (local_toxic + selectivity)

        # Amount of toxin metabolized
        metabolized = detox_eff * uptake_factor * has_detox

        # Energy from toxin conversion
        conversion = c.detox_energy_conversion * (1.0 / (1.0 + np.exp(-dtw[:, 2])))
        energy_gained = metabolized * conversion * local_toxic
        self.energy += energy_gained * has_detox

        # Environmental detox: remove toxin from the grid
        toxin_removed = metabolized * c.detox_environment_effect * local_toxic
        np.add.at(self.toxic, (self.rows, self.cols), -toxin_removed * has_detox)
        np.clip(self.toxic, 0, 5.0, out=self.toxic)

    # ── Social System ──

    def _update_social_field(self):
        """Deposit social identity signals on the grid.
        Channel 0: producer-type signal (PHOTO/CHEMO organisms)
        Channel 1: consumer-type signal (CONSUME organisms)
        Signal strength modulated by SOCIAL module if present."""
        c = self.cfg
        n = self.pop
        self.social_field *= 0.5  # decay old signals

        if n == 0:
            return

        has_social = self.module_active[:, M_SOCIAL]
        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]

        # Base signal: all organisms emit weak identity signal
        base_strength = np.ones(n) * 0.3

        # SOCIAL module amplifies signal
        if has_social.any():
            sw = self._module_weights(M_SOCIAL)
            signal_strength = 1.0 / (1.0 + np.exp(-sw[:, 0]))  # identity_signal weight
            base_strength = np.where(has_social, 0.3 + 0.7 * signal_strength, base_strength)

        # Deposit producer signal
        prod_signal = base_strength * has_producer
        if prod_signal.any():
            np.add.at(self.social_field[:, :, 0], (self.rows, self.cols), prod_signal)

        # Deposit consumer signal
        cons_signal = base_strength * has_consume
        if cons_signal.any():
            np.add.at(self.social_field[:, :, 1], (self.rows, self.cols), cons_signal)

        np.clip(self.social_field, 0, 10.0, out=self.social_field)

    def _apply_social_interactions(self):
        """Compatible neighbors provide energy bonus; relationship scores accumulate.
        Compatibility: consumers benefit from producer proximity and vice versa
        (metabolic complementarity — future endosymbiosis prerequisite)."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        has_social = self.module_active[:, M_SOCIAL]
        if not has_social.any():
            # Relationship scores still decay for all organisms
            self.relationship_score *= c.social_relationship_decay
            return

        sw = self._module_weights(M_SOCIAL)
        compat_skill = 1.0 / (1.0 + np.exp(-sw[:, 1]))       # compatibility_assessment
        approach = np.tanh(sw[:, 2])                           # approach_avoidance (-1 to 1)
        rel_strength = 1.0 / (1.0 + np.exp(-sw[:, 3]))        # relationship_strength

        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]

        # Local social signals at each organism's cell
        local_prod = self.social_field[self.rows, self.cols, 0]
        local_cons = self.social_field[self.rows, self.cols, 1]

        # Compatibility: producers benefit from nearby consumers, consumers from producers
        # This is metabolic complementarity — the foundation for endosymbiosis
        compatible_signal = np.zeros(n)
        compatible_signal += has_producer * local_cons  # producers near consumers
        compatible_signal += has_consume * local_prod   # consumers near producers
        # Same-type organisms also provide mild cooperative benefit
        compatible_signal += has_producer * local_prod * 0.3
        compatible_signal += has_consume * local_cons * 0.2

        # Energy bonus from compatible proximity (only for SOCIAL module holders)
        bonus = (c.social_compatibility_bonus * compat_skill
                * np.minimum(compatible_signal, 3.0) * has_social)
        self.energy += bonus

        # Relationship score: grows near compatible organisms, decays otherwise
        growth = (c.social_relationship_growth * rel_strength
                 * np.minimum(compatible_signal, 5.0) * has_social)
        self.relationship_score *= c.social_relationship_decay
        self.relationship_score += growth
        np.clip(self.relationship_score, 0, 10.0, out=self.relationship_score)

    # ── Mediator System (Pollination/Dispersal) ──

    def _update_mediator_field(self):
        """Mediators deposit pollination service signal on the grid.
        Higher pollination_drive = stronger signal = more reproduction facilitation."""
        c = self.cfg
        n = self.pop
        self.mediator_field *= c.mediate_network_decay

        if n == 0:
            return

        has_mediate = self.module_active[:, M_MEDIATE]
        if not has_mediate.any():
            return

        mw = self._module_weights(M_MEDIATE)
        poll_drive = 1.0 / (1.0 + np.exp(-mw[:, 0]))  # pollination_drive
        network_coord = 1.0 / (1.0 + np.exp(-mw[:, 2]))  # network_coordination

        # Signal strength: base + coordination bonus when multiple mediators nearby
        local_med = self.mediator_field[self.rows, self.cols]
        signal = poll_drive * (1.0 + network_coord * np.minimum(local_med, 3.0) * 0.3)
        signal *= has_mediate

        np.add.at(self.mediator_field, (self.rows, self.cols), signal)
        np.clip(self.mediator_field, 0, 10.0, out=self.mediator_field)

    # ── Nutrient Cycling ──

    def _apply_nutrient_cycling(self):
        """Emergent nutrient cycling from module interactions.
        Not a separate module — organisms with DETOX, CONSUME, and FORAGE
        contribute to local nutrient chemistry as a side effect."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        # DETOX byproducts → nutrients (detoxification enriches soil)
        has_detox = self.module_active[:, M_DETOX]
        if has_detox.any():
            dtw = self._module_weights(M_DETOX)
            detox_eff = 1.0 / (1.0 + np.exp(-dtw[:, 0]))
            conversion = 1.0 / (1.0 + np.exp(-dtw[:, 2]))
            local_toxic = self.toxic[self.rows, self.cols]
            deposit = (c.nutrient_detox_deposit * detox_eff * conversion
                      * np.minimum(local_toxic, 2.0) * has_detox)
            np.add.at(self.nutrients, (self.rows, self.cols), deposit)

        # CONSUME processing → nutrients (digestion releases nutrients)
        has_consume = self.module_active[:, M_CONSUME]
        if has_consume.any():
            uw = self._module_weights(M_CONSUME)
            handling = 1.0 / (1.0 + np.exp(-uw[:, 1]))
            deposit = c.nutrient_consume_deposit * handling * has_consume
            np.add.at(self.nutrients, (self.rows, self.cols), deposit)

        # FORAGE cooperative clusters boost local nutrient regeneration
        has_forage = self.module_active[:, M_FORAGE]
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            coop_signal = 1.0 / (1.0 + np.exp(-fw[:, 3]))
            boost = c.nutrient_forage_coop_boost * coop_signal * has_forage
            np.add.at(self.nutrients, (self.rows, self.cols), boost)

        np.clip(self.nutrients, 0, self.cfg.nutrient_max, out=self.nutrients)

    # ── Behavioral Hijacking ──

    def _compute_hijack_intensity(self):
        """Compute per-organism hijack intensity from viral load.
        Returns array of 0-1 values. Called once per step, stored on self."""
        c = self.cfg
        n = self.pop
        self.hijack_intensity = np.zeros(n)
        if n == 0:
            return

        # Hijack active when viral_load in range [min, burst_threshold)
        active = (self.viral_load >= c.hijack_load_min) & (self.viral_load < c.viral_burst_threshold)
        if not active.any():
            return

        aidx = np.where(active)[0]
        load = self.viral_load[aidx]

        # Intensity: ramps from 0 at hijack_load_min to 1 at hijack_load_heavy
        raw = (load - c.hijack_load_min) / max(c.hijack_load_heavy - c.hijack_load_min, 0.01)
        intensity = np.clip(raw, 0.0, 1.0)

        # Toxic stress amplifies hijack (stressed hosts easier to control)
        local_toxic = self.toxic[self.rows[aidx], self.cols[aidx]]
        stress_mult = 1.0 + local_toxic * c.hijack_stress_amplifier
        intensity = np.minimum(intensity * stress_mult, 1.0)

        # VRESIST partially resists hijacking
        has_vresist = self.module_active[aidx, M_VRESIST]
        if has_vresist.any():
            vw = self._module_weights(M_VRESIST)
            breadth = 1.0 / (1.0 + np.exp(-vw[aidx, 2]))
            memory = np.minimum(self.immune_experience[aidx], 3.0) / 3.0
            resistance = (breadth * 0.5 + memory * 0.5) * c.hijack_vresist_suppress
            intensity[has_vresist] *= (1.0 - resistance[has_vresist])

        self.hijack_intensity[aidx] = intensity
        self.total_hijacked_steps += int(active.sum())

    # ── Movement ──

    def _decide_movement(self, readings):
        n = self.pop
        if n == 0:
            return np.array([], dtype=np.int32)

        has_move = self.module_active[:, M_MOVE]
        has_photo = self.module_active[:, M_PHOTO]
        has_chemo = self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]
        mp = self._module_weights(M_MOVE)

        sc = np.zeros((n, 5))

        # Stay score: nutrient attraction + decomp attraction for consumers
        sc[:, 0] = (mp[:, 5] + mp[:, 2] * readings[:, 2]
                    - mp[:, 1] * np.minimum(readings[:, 3], 10) * 0.1)
        if has_consume.any():
            sc[:, 0] += has_consume * readings[:, 8] * 0.3

        # Gradient weights
        light_w = mp[:, 6] * has_photo.astype(np.float64)
        net_toxic = mp[:, 7] * has_photo.astype(np.float64) - mp[:, 3] * has_chemo.astype(np.float64)

        # Consumer-specific: carnivores→density, detritivores→scent
        consume_dens_w = np.zeros(n)
        consume_scent_w = np.zeros(n)
        if has_consume.any():
            uw = self._module_weights(M_CONSUME)
            dp = 1.0 / (1.0 + np.exp(-uw[:, 2]))
            consume_dens_w = has_consume * (1.0 - dp) * 0.5
            consume_scent_w = has_consume * dp * 8.0

        # FORAGE: seek nutrient-rich areas
        has_forage = self.module_active[:, M_FORAGE]
        forage_nutr_w = np.zeros(n)
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            resource_discrim = 1.0 / (1.0 + np.exp(-fw[:, 1]))
            forage_nutr_w = has_forage * resource_discrim * 2.0

        # DETOX: seek toxic areas (food source)
        has_detox = self.module_active[:, M_DETOX]
        detox_toxic_w = np.zeros(n)
        if has_detox.any():
            dtw = self._module_weights(M_DETOX)
            detox_seek = 1.0 / (1.0 + np.exp(-dtw[:, 0]))
            detox_toxic_w = has_detox * detox_seek * 1.5

        # SOCIAL: seek compatible organisms via social signal gradients
        has_social = self.module_active[:, M_SOCIAL]
        social_compat_gy = np.zeros(n)
        social_compat_gx = np.zeros(n)
        if has_social.any():
            sw = self._module_weights(M_SOCIAL)
            approach = np.tanh(sw[:, 2])  # approach_avoidance: positive = approach
            has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]

            # Producers follow consumer signal gradients (seek metabolic complement)
            # Consumers follow producer signal gradients
            # readings 17-20: social gradients
            spg_y, spg_x = readings[:, 17], readings[:, 18]  # producer signal gradients
            scg_y, scg_x = readings[:, 19], readings[:, 20]  # consumer signal gradients

            # Each organism seeks the complementary type
            compat_gy = np.where(has_producer, scg_y, spg_y) * has_social
            compat_gx = np.where(has_producer, scg_x, spg_x) * has_social
            # Also mild attraction to same-type (cooperative)
            same_gy = np.where(has_producer, spg_y, scg_y) * has_social * 0.3
            same_gx = np.where(has_producer, spg_x, scg_x) * has_social * 0.3

            social_compat_gy = approach * (compat_gy + same_gy) * 1.5
            social_compat_gx = approach * (compat_gx + same_gx) * 1.5

        # MEDIATE: seek organism-dense areas (go where pollination is needed)
        has_mediate = self.module_active[:, M_MEDIATE]
        mediate_dens_gy = np.zeros(n)
        mediate_dens_gx = np.zeros(n)
        if has_mediate.any():
            mew = self._module_weights(M_MEDIATE)
            route_mem = 1.0 / (1.0 + np.exp(-mew[:, 1]))  # route_memory
            # Mediators seek social signal gradients (where organisms are)
            # Combined producer+consumer social signal as target
            spg_y, spg_x = readings[:, 17], readings[:, 18]
            scg_y, scg_x = readings[:, 19], readings[:, 20]
            mediate_dens_gy = has_mediate * (spg_y + scg_y) * (0.5 + route_mem) * 2.0
            mediate_dens_gx = has_mediate * (spg_x + scg_x) * (0.5 + route_mem) * 2.0

        # Directional scores
        lg_y, lg_x = readings[:, 4], readings[:, 5]
        tg_y, tg_x = readings[:, 6], readings[:, 7]
        dg_y, dg_x = readings[:, 9], readings[:, 10]
        sg_y, sg_x = readings[:, 11], readings[:, 12]
        ng_y, ng_x = readings[:, 13], readings[:, 14]

        gy = (light_w * lg_y - net_toxic * tg_y + consume_dens_w * dg_y
              + consume_scent_w * sg_y + forage_nutr_w * ng_y + detox_toxic_w * tg_y
              + social_compat_gy + mediate_dens_gy)
        gx = (light_w * lg_x - net_toxic * tg_x + consume_dens_w * dg_x
              + consume_scent_w * sg_x + forage_nutr_w * ng_x + detox_toxic_w * tg_x
              + social_compat_gx + mediate_dens_gx)
        sc[:, 1] = gy;  sc[:, 2] = -gy
        sc[:, 3] = gx;  sc[:, 4] = -gx

        # Behavioral hijacking: override movement for infected organisms
        hijacked = self.hijack_intensity > 0.1
        if hijacked.any():
            hidx = np.where(hijacked)[0]
            hi = self.hijack_intensity[hidx]
            # Hijacked organisms seek high-density, low-viral areas (spread virus to new hosts)
            hijack_gy = (readings[hidx, 9] * self.cfg.hijack_density_seek   # toward density
                        - readings[hidx, 6] * 1.0)                    # away from existing viral zones
            hijack_gx = (readings[hidx, 10] * self.cfg.hijack_density_seek
                        - readings[hidx, 7] * 1.0)
            # Blend: hijack overrides host movement proportional to intensity
            sc[hidx, 0] *= (1.0 - hi * 0.8)  # suppress stay tendency
            sc[hidx, 1] = sc[hidx, 1] * (1.0 - hi) + hijack_gy * hi
            sc[hidx, 2] = sc[hidx, 2] * (1.0 - hi) - hijack_gy * hi
            sc[hidx, 3] = sc[hidx, 3] * (1.0 - hi) + hijack_gx * hi
            sc[hidx, 4] = sc[hidx, 4] * (1.0 - hi) - hijack_gx * hi
            # Heavy hijack: erratic movement noise
            heavy = hi > 0.7
            if heavy.any():
                hh = hidx[heavy]
                sc[hh, 1:] += self.rng.normal(0, 2.0, (len(hh), 4))

        # Exploration noise
        sc += mp[:, 4:5] * self.rng.normal(0, 0.3, (n, 5))

        # Sessile organisms stay
        sc[~has_move, 1:] = -999.0

        return np.argmax(sc, axis=1)

    def _execute_movement(self, actions):
        N = self.cfg.grid_size
        m = actions == 1; self.rows[m] = np.maximum(0, self.rows[m] - 1)
        m = actions == 2; self.rows[m] = np.minimum(N - 1, self.rows[m] + 1)
        m = actions == 3; self.cols[m] = np.minimum(N - 1, self.cols[m] + 1)
        m = actions == 4; self.cols[m] = np.maximum(0, self.cols[m] - 1)
        self.energy[actions > 0] -= self.cfg.energy_movement_cost

    # ── Horizontal Transfer ──

    def _horizontal_transfer(self):
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        sp = self._standalone_params()
        receptivity = 1.0 / (1.0 + np.exp(-sp[:, SP_TRANSFER_RECEPTIVITY]))
        selectivity = np.abs(sp[:, SP_TRANSFER_SELECTIVITY])
        local_toxic = self.toxic[self.rows, self.cols]

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

            local_frags = self.strata_pool[sname][self.rows[eidx], self.cols[eidx]]
            dists = np.sqrt(np.mean((self.weights[eidx] - local_frags) ** 2, axis=1))
            depth_factor = {"recent": 1.0, "intermediate": 1.5, "ancient": 2.5}[sname]
            thresh = (2.0 * depth_factor) / (1.0 + selectivity[eidx])
            tidx = eidx[dists < thresh]
            if len(tidx) == 0:
                continue

            frags = self.strata_pool[sname][self.rows[tidx], self.cols[tidx]]
            self.weights[tidx] = (1.0 - blend_rate) * self.weights[tidx] + blend_rate * frags
            self.transfer_count[tidx] += 1
            self.total_transfers += len(tidx)
            self.transfers_by_stratum[sname] += len(tidx)

            # Module acquisition via HGT
            if self.rng.random() < 0.5:
                frag_mods = self.strata_modules[sname][self.rows[tidx], self.cols[tidx]]
                for i, ti in enumerate(tidx):
                    if self.rng.random() < c.module_transfer_rate:
                        available = [gm for gm in GAINABLE_MODULES
                                    if frag_mods[i, gm] > 0.3 and not self.module_present[ti, gm]]
                        if available:
                            self.module_present[ti, self.rng.choice(available)] = True
                            self.module_active[ti] = self.module_present[ti]

    # ── Viral Dynamics (VRESIST-enhanced) ──

    def _viral_dynamics(self):
        """Viral infection, lytic damage, lysogenic activation.
        VRESIST module provides much stronger resistance with specificity/breadth trade-off.
        Organisms without VRESIST use weaker standalone params as fallback."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        sp = self._standalone_params()
        has_vresist = self.module_active[:, M_VRESIST]
        local_viral = self.viral_particles[self.rows, self.cols]
        local_toxic = self.toxic[self.rows, self.cols]

        # Compute viral resistance: VRESIST module vs standalone fallback
        # Standalone: weak baseline resistance (~0.3 effective)
        base_resistance = 0.3 * (1.0 / (1.0 + np.exp(-sp[:, SP_VIRAL_RESISTANCE])))

        if has_vresist.any():
            vw = self._module_weights(M_VRESIST)
            specificity = 1.0 / (1.0 + np.exp(-vw[:, 0]))  # targeted resistance
            breadth = 1.0 / (1.0 + np.exp(-vw[:, 2]))      # broad resistance
            # Specificity/breadth trade-off: can't max both
            # Specificity good in stable environments; breadth good against novel strains
            vresist_resist = (c.vresist_base_resistance
                            + c.vresist_specificity_bonus * specificity
                            + c.vresist_breadth_bonus * breadth)
            # Immune memory from past infections boosts resistance
            memory_boost = (c.vresist_memory_boost
                          * (1.0 / (1.0 + np.exp(-vw[:, 3])))  # immune_memory weight
                          * np.minimum(self.immune_experience, 3.0))
            vresist_resist = np.minimum(vresist_resist + memory_boost, 0.95)
            # Merge: VRESIST organisms use module resistance, others use standalone
            viral_resistance = np.where(has_vresist, vresist_resist, base_resistance)
        else:
            viral_resistance = base_resistance

        # Compute lysogenic suppression: VRESIST enhances
        base_suppression = 0.3 * (1.0 / (1.0 + np.exp(-sp[:, SP_LYSO_SUPPRESSION])))
        if has_vresist.any():
            vw = self._module_weights(M_VRESIST)
            vresist_supp = c.vresist_suppression_max * (1.0 / (1.0 + np.exp(-vw[:, 1])))
            lysogenic_suppression = np.where(has_vresist, vresist_supp, base_suppression)
        else:
            lysogenic_suppression = base_suppression

        # Immune memory decay
        self.immune_experience *= c.vresist_memory_decay

        # New infections
        candidates = (self.viral_load == 0) & (local_viral > 0.05)
        if candidates.any():
            cidx = np.where(candidates)[0]
            inf_prob = (c.viral_infection_rate * np.minimum(local_viral[cidx], 5.0) / 5.0
                       * (1.0 - viral_resistance[cidx]))
            inf_prob = np.maximum(inf_prob, 0.01)  # minimum infection chance
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

        # Lytic damage + immune experience from surviving
        lytic_mask = self.viral_load > 0
        if lytic_mask.any():
            lidx = np.where(lytic_mask)[0]
            self.viral_load[lidx] += c.viral_lytic_growth
            self.energy[lidx] -= c.viral_lytic_damage * self.viral_load[lidx]

            # Organisms that survive lytic infection gain immune experience
            # (viral_load will be cleared by death; survivors keep accumulating)
            survivors = lidx[self.energy[lidx] > 0]
            if len(survivors) > 0:
                self.immune_experience[survivors] += 0.05

        # Lysogenic activation under stress (suppression from VRESIST)
        act_cand = ((self.lysogenic_strength > 0.01) &
                    (local_toxic > c.lysogenic_activation_toxic) &
                    (self.viral_load == 0))
        if act_cand.any():
            acidx = np.where(act_cand)[0]
            stress = (local_toxic[acidx] - c.lysogenic_activation_toxic) / c.lysogenic_activation_toxic
            act_prob = np.minimum(stress * 0.3, 0.5) * (1.0 - lysogenic_suppression[acidx])
            activated = acidx[self.rng.random(len(acidx)) < act_prob]
            if len(activated) > 0:
                blend = np.minimum(c.lysogenic_blend_rate * self.lysogenic_strength[activated], 0.4)
                self.weights[activated] = (
                    self.weights[activated] * (1.0 - blend[:, None])
                    + self.lysogenic_genome[activated] * blend[:, None])
                self.viral_load[activated] = 0.2
                self.lysogenic_strength[activated] *= 0.3
                self.total_lysogenic_activations += len(activated)

    # ── Reproduction ──

    def _reproduce(self):
        c = self.cfg
        N = c.grid_size

        # Density-dependent threshold: harder to reproduce in crowded areas
        local_dens = self.density[self.rows, self.cols].astype(np.float64)
        effective_threshold = c.energy_reproduction_threshold + local_dens * c.repro_density_penalty

        # Mediator bonus: pollination service lowers reproduction threshold
        local_med = self.mediator_field[self.rows, self.cols]
        mediator_bonus = np.minimum(local_med, 3.0) * c.mediate_repro_bonus
        effective_threshold -= mediator_bonus

        can = ((self.energy >= effective_threshold) &
               (self.age >= c.min_reproduction_age) &
               (self.viral_load == 0))
        pidx = np.where(can)[0]
        nb = len(pidx)
        if nb == 0:
            return

        self.energy[pidx] -= c.energy_reproduction_cost

        # Reward mediators near reproduction events
        if nb > 0:
            has_mediate = self.module_active[:, M_MEDIATE]
            if has_mediate.any():
                mw = self._module_weights(M_MEDIATE)
                reward_sens = 1.0 / (1.0 + np.exp(-mw[:, 3]))  # reward_sensitivity
                local_repro_signal = np.zeros(self.pop)
                # Count reproductions at each mediator's location
                repro_cells = set(zip(self.rows[pidx].tolist(), self.cols[pidx].tolist()))
                for mi in np.where(has_mediate)[0]:
                    r, col = self.rows[mi], self.cols[mi]
                    # Check within mediate_radius
                    nearby_repros = sum(1 for rr, cc in repro_cells
                                       if abs(rr - r) <= c.mediate_radius
                                       and abs(cc - col) <= c.mediate_radius)
                    if nearby_repros > 0:
                        self.energy[mi] += (c.mediate_energy_reward * reward_sens[mi]
                                          * min(nearby_repros, 5))
        child_ids = np.arange(self.next_id, self.next_id + nb, dtype=np.int64)
        self.next_id += nb

        child_modules = self.module_present[pidx].copy()

        # Module gain
        gain_rolls = self.rng.random(nb)
        for i in np.where(gain_rolls < c.module_gain_rate)[0]:
            absent = [gm for gm in GAINABLE_MODULES if not child_modules[i, gm]]
            if absent:
                child_modules[i, self.rng.choice(absent)] = True

        # Module loss (protect last energy module)
        lose_rolls = self.rng.random(nb)
        for i in np.where(lose_rolls < c.module_lose_rate)[0]:
            present = np.where(child_modules[i])[0]
            droppable = []
            for m in present:
                if m == M_TOXPROD:
                    continue
                if m in (M_PHOTO, M_CHEMO, M_CONSUME):
                    if not any(child_modules[i, e] for e in (M_PHOTO, M_CHEMO, M_CONSUME) if e != m):
                        continue
                droppable.append(m)
            if droppable:
                child_modules[i, self.rng.choice(droppable)] = False

        # Offspring placement
        has_move_p = self.module_active[pidx, M_MOVE]
        off_dist = np.where(has_move_p, c.offspring_distance, 1)
        row_off = np.round(self.rng.uniform(-1, 1, nb) * off_dist).astype(np.int64)
        col_off = np.round(self.rng.uniform(-1, 1, nb) * off_dist).astype(np.int64)

        # Decomp-seeking offspring: detritivore children seek carrion via scent
        has_consume_p = self.module_active[pidx, M_CONSUME]
        if has_consume_p.any():
            uw = self._module_weights(M_CONSUME)
            dp_pref = 1.0 / (1.0 + np.exp(-uw[pidx, 2]))
            seekers = np.where(has_consume_p & (dp_pref > 0.4))[0]
            if len(seekers) > 0:
                search_r = 8
                for i in seekers:
                    pr, pc = int(self.rows[pidx[i]]), int(self.cols[pidx[i]])
                    r_lo = max(0, pr - search_r)
                    r_hi = min(N - 1, pr + search_r)
                    c_lo = max(0, pc - search_r)
                    c_hi = min(N - 1, pc + search_r)
                    # Sample scent in search area (step by 2 for speed)
                    patch = self.decomp_scent[r_lo:r_hi+1:2, c_lo:c_hi+1:2]
                    if patch.size > 0 and patch.max() > 0.01:
                        best = np.unravel_index(patch.argmax(), patch.shape)
                        row_off[i] = r_lo + best[0] * 2 - pr
                        col_off[i] = c_lo + best[1] * 2 - pc

        child_weights = self.weights[pidx] + self.rng.normal(0, c.mutation_rate, (nb, TOTAL_WEIGHT_PARAMS))

        # ── Reproductive Manipulation (Wolbachia-style) ──
        # Lysogenic viral material in parents silently biases offspring
        parent_lyso = self.lysogenic_strength[pidx]
        manipulated = parent_lyso > c.repro_manip_threshold
        n_manipulated = int(manipulated.sum())

        if n_manipulated > 0:
            midx = np.where(manipulated)[0]
            parent_lyso_genome = self.lysogenic_genome[pidx[midx]]
            manip_strength = np.minimum(parent_lyso[midx], 2.0) / 2.0  # normalize to 0-1

            # Self-limiting: manipulation weakens at high population penetration
            pop_lyso_frac = (self.lysogenic_strength > c.repro_manip_threshold).sum() / max(self.pop, 1)
            if pop_lyso_frac > c.repro_manip_saturation:
                saturation_penalty = (pop_lyso_frac - c.repro_manip_saturation) / (1.0 - c.repro_manip_saturation)
                manip_strength *= (1.0 - saturation_penalty * 0.8)

            # 1. Trait bias: blend offspring weights toward lysogenic genome
            trait_blend = c.repro_manip_trait_bias * manip_strength
            child_weights[midx] = (child_weights[midx] * (1.0 - trait_blend[:, None])
                                  + parent_lyso_genome * trait_blend[:, None])

            # 2. Receptivity boost: make offspring more susceptible to HGT
            recept_idx = STANDALONE_OFFSET + SP_TRANSFER_RECEPTIVITY
            child_weights[midx, recept_idx] += c.repro_manip_receptivity_boost * manip_strength

            # 3. Viability filter: penalize offspring that diverge from lysogenic template
            divergence = np.sqrt(np.mean((child_weights[midx] - parent_lyso_genome) ** 2, axis=1))
            penalty_mask = divergence > c.repro_manip_divergence_thresh
            if penalty_mask.any():
                pen_idx = midx[penalty_mask]
                pen_strength = manip_strength[penalty_mask]
                energy_penalty = c.repro_manip_viability_cost * pen_strength * \
                    (divergence[penalty_mask] - c.repro_manip_divergence_thresh)
                # Store penalty to apply to offspring energy below
                # (offspring energy set to energy_initial, so we reduce it)
                child_energy = np.full(nb, c.energy_initial)
                child_energy[pen_idx] -= energy_penalty
                np.clip(child_energy, 5.0, c.energy_initial, out=child_energy)
            else:
                child_energy = np.full(nb, c.energy_initial)

            self.total_manipulated_births += n_manipulated
        else:
            child_energy = np.full(nb, c.energy_initial)

        self._append_organisms({
            "rows": np.clip(self.rows[pidx] + row_off, 0, N - 1),
            "cols": np.clip(self.cols[pidx] + col_off, 0, N - 1),
            "energy": child_energy,
            "age": np.zeros(nb, dtype=np.int32),
            "generation": self.generation[pidx] + 1,
            "ids": child_ids,
            "parent_ids": self.ids[pidx],
            "weights": child_weights,
            "module_present": child_modules,
            "module_active": child_modules.copy(),
            "transfer_count": np.zeros(nb, dtype=np.int32),
            "viral_load": np.zeros(nb, dtype=np.float64),
            "lysogenic_strength": self.lysogenic_strength[pidx] * c.lysogenic_inheritance,
            "lysogenic_genome": self.lysogenic_genome[pidx] * c.lysogenic_inheritance,
            "immune_experience": self.immune_experience[pidx] * 0.3,  # partial maternal immunity
            "relationship_score": np.zeros(nb, dtype=np.float64),     # fresh start
        })

    # ── Death and Decomposition ──

    def _kill_and_decompose(self):
        c = self.cfg
        N = c.grid_size

        bursting = self.viral_load >= c.viral_burst_threshold
        natural_death = (self.energy <= 0) | (self.age >= c.max_age)
        dead = bursting | natural_death

        if dead.any():
            dr, dc = self.rows[dead], self.cols[dead]
            de = np.maximum(0, self.energy[dead])
            dw, dm = self.weights[dead], self.module_present[dead]

            # Deposit decomp
            np.add.at(self.decomposition, (dr, dc), de * c.nutrient_from_decomp + c.decomp_death_deposit)

            # Nutrient cycling: dead organisms release nutrients proportional to module complexity
            module_complexity = dm.sum(axis=1).astype(np.float64)
            nutrient_deposit = module_complexity * c.nutrient_death_per_module
            np.add.at(self.nutrients, (dr, dc), nutrient_deposit)

            # Update strata with dead organism genomes
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
            self.strata_pool["recent"][ur, uc] = (
                self.strata_pool["recent"][ur, uc] * (1.0 - blend[:, None])
                + avg_w * blend[:, None])
            self.strata_modules["recent"][ur, uc] = (
                self.strata_modules["recent"][ur, uc] * (1.0 - blend[:, None])
                + avg_m * blend[:, None])
            self.strata_weight["recent"][ur, uc] = wt

            # Viral burst
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

            # Spontaneous viral shedding from natural deaths
            nat_dead = ~burst_mask
            if nat_dead.any():
                spont = self.rng.random(int(nat_dead.sum())) < 0.15
                if spont.any():
                    nat_idx = np.where(nat_dead)[0][spont]
                    np.add.at(self.viral_particles, (dr[nat_idx], dc[nat_idx]), 1.0)
                    np.add.at(self.viral_genome_weight, (dr[nat_idx], dc[nat_idx]), 1.0)

        self._filter_organisms(~dead)

    # ── Main Loop ──

    def update(self):
        if self.pop == 0:
            self._update_environment()
            self.timestep += 1
            self._record_stats(0)
            return

        self._update_density()
        readings = self._sense_local()

        # Compute behavioral hijack intensity for this step
        self._compute_hijack_intensity()

        effective_max = self._effective_energy_max()
        energy_gain = self._acquire_energy(readings)

        # Behavioral hijacking: heavy hijack suppresses energy acquisition
        if self.hijack_intensity.any():
            heavy = self.hijack_intensity > 0.3
            if heavy.any():
                suppress = self.hijack_intensity[heavy] * self.cfg.hijack_energy_suppress
                energy_gain[heavy] *= (1.0 - suppress)

        self.energy = np.minimum(self.energy + energy_gain, effective_max)
        self._apply_toxic_damage(readings)
        self._apply_detox()
        self.energy -= self._compute_total_costs()

        # Sessile crowding penalty (producers only — consumers thrive in crowds)
        no_move = ~self.module_active[:, M_MOVE]
        sessile_prod = no_move & ~self.module_active[:, M_CONSUME]
        if sessile_prod.any():
            crowding = np.maximum(0, readings[:, 3] - 2) * 0.15
            self.energy[sessile_prod] -= crowding[sessile_prod]

        self._execute_movement(self._decide_movement(readings))
        self.age += 1

        # Social field update and interactions
        if self.timestep % self.cfg.social_update_interval == 0:
            self._update_social_field()
        self._apply_social_interactions()

        # Mediator field update (pollination service availability)
        if self.timestep % self.cfg.mediate_update_interval == 0:
            self._update_mediator_field()

        # Nutrient cycling (emergent from DETOX/CONSUME/FORAGE interactions)
        self._apply_nutrient_cycling()

        kills = 0
        if self.timestep % self.cfg.predation_check_interval == 0:
            kills = self._predation()
        if self.timestep % self.cfg.transfer_check_interval == 0:
            self._horizontal_transfer()
        if self.timestep % self.cfg.viral_check_interval == 0:
            self._viral_dynamics()

        self._reproduce()
        self._kill_and_decompose()
        self._update_environment()
        self._record_stats(kills)
        self.timestep += 1

    # ── Stats ──

    def _classify_roles(self):
        n = self.pop
        if n == 0:
            return {"producer": 0, "herbivore": 0, "carnivore": 0, "detritivore": 0, "omnivore": 0}

        has_prod = self.module_present[:, M_PHOTO] | self.module_present[:, M_CHEMO]
        has_cons = self.module_present[:, M_CONSUME]
        producer = has_prod & ~has_cons
        omnivore = has_prod & has_cons
        obligate = has_cons & ~has_prod

        herbivore = carnivore = detritivore = np.zeros(n, dtype=bool)
        if obligate.any():
            uw = self._module_weights(M_CONSUME)
            dp = 1.0 / (1.0 + np.exp(-uw[:, 2]))   # decomp_pref
            ps = 1.0 / (1.0 + np.exp(-uw[:, 0]))   # prey_selectivity: 0=herbivore, 1=carnivore
            detritivore = obligate & (dp >= 0.5)
            hunter = obligate & (dp < 0.5)
            herbivore = hunter & (ps < 0.5)
            carnivore = hunter & (ps >= 0.5)

        return {
            "producer": int(producer.sum()),
            "herbivore": int(herbivore.sum()),
            "carnivore": int(carnivore.sum()),
            "detritivore": int(detritivore.sum()),
            "omnivore": int(omnivore.sum()),
        }

    def _record_stats(self, kills=0):
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
                "avg_immune_exp": round(float(self.immune_experience.mean()), 3),
                "avg_relationship": round(float(self.relationship_score.mean()), 3),
                "max_relationship": round(float(self.relationship_score.max()), 3),
                "mediator_field_mean": round(float(self.mediator_field.mean()), 4),
                "nutrient_mean": round(float(self.nutrients.mean()), 3),
                "lyso_fraction": round(float((self.lysogenic_strength > self.cfg.repro_manip_threshold).sum() / max(p, 1)), 3),
                "manipulated_births": self.total_manipulated_births,
                "hijack_fraction": round(float((self.hijack_intensity > 0.1).sum() / max(p, 1)), 3),
                "hijacked_steps": self.total_hijacked_steps,
            })
        else:
            self.stats_history.append({
                "t": self.timestep, "pop": 0, "avg_energy": 0, "max_gen": 0,
                "toxic_mean": round(float(self.toxic.mean()), 3),
                "decomp_mean": round(float(self.decomposition.mean()), 2),
                "avg_modules": 0, "module_counts": {MODULE_NAMES[m]: 0 for m in range(N_MODULES)},
                "roles": {"producer": 0, "herbivore": 0, "carnivore": 0, "detritivore": 0, "omnivore": 0},
                "kills": 0, "total_kills": self.total_predation_kills,
                "avg_immune_exp": 0, "avg_relationship": 0, "max_relationship": 0,
                "mediator_field_mean": 0, "nutrient_mean": round(float(self.nutrients.mean()), 3),
                "lyso_fraction": 0, "manipulated_births": self.total_manipulated_births,
                "hijack_fraction": 0, "hijacked_steps": self.total_hijacked_steps,
            })

    def save_snapshot(self, output_dir):
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


# ─────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────

def run_simulation(cfg=None):
    cfg = cfg or Config()
    world = World(cfg)

    print(f"The Shimmering Field — Phase 3 Step 8: Behavioral Hijacking")
    print(f"Grid: {cfg.grid_size}×{cfg.grid_size}  |  Pop: {cfg.initial_population}  |  Weights: {TOTAL_WEIGHT_PARAMS}")
    print(f"Module costs — PH:{MODULE_MAINTENANCE[M_PHOTO]+MODULE_EXPRESSION[M_PHOTO]:.2f}  "
          f"CH:{MODULE_MAINTENANCE[M_CHEMO]+MODULE_EXPRESSION[M_CHEMO]:.2f}  "
          f"CO:{MODULE_MAINTENANCE[M_CONSUME]+MODULE_EXPRESSION[M_CONSUME]:.2f}  "
          f"MV:{MODULE_MAINTENANCE[M_MOVE]+MODULE_EXPRESSION[M_MOVE]:.2f}  "
          f"FO:{MODULE_MAINTENANCE[M_FORAGE]+MODULE_EXPRESSION[M_FORAGE]:.2f}  "
          f"DE:{MODULE_MAINTENANCE[M_DEFENSE]+MODULE_EXPRESSION[M_DEFENSE]:.2f}  "
          f"DT:{MODULE_MAINTENANCE[M_DETOX]+MODULE_EXPRESSION[M_DETOX]:.2f}  "
          f"VR:{MODULE_MAINTENANCE[M_VRESIST]+MODULE_EXPRESSION[M_VRESIST]:.2f}  "
          f"SO:{MODULE_MAINTENANCE[M_SOCIAL]+MODULE_EXPRESSION[M_SOCIAL]:.2f}  "
          f"ME:{MODULE_MAINTENANCE[M_MEDIATE]+MODULE_EXPRESSION[M_MEDIATE]:.2f}")
    print(f"Repro manipulation: threshold={cfg.repro_manip_threshold}  "
          f"trait_bias={cfg.repro_manip_trait_bias}  saturation={cfg.repro_manip_saturation}")
    print(f"{'─' * 160}")

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
                f"gen={s['max_gen']:4d}  |  tox={s['toxic_mean']:.3f} ntr={s['nutrient_mean']:.3f}  |  "
                f"prod={r['producer']:4d} herb={r['herbivore']:3d} carn={r['carnivore']:3d} detr={r['detritivore']:3d} omni={r['omnivore']:3d}  |  "
                f"FO={mc['FORAGE']:4d} DE={mc['DEFENSE']:4d} DT={mc['DETOX']:4d} VR={mc['VRESIST']:4d} SO={mc['SOCIAL']:4d} ME={mc['MEDIATE']:3d}  |  "
                f"rel={s['avg_relationship']:.2f} med={s['mediator_field_mean']:.3f}  |  "
                f"lyso={s['lyso_fraction']:.2f} manip={s['manipulated_births']:5d}  |  "
                f"hjk={s['hijack_fraction']:.2f}  |  "
                f"kill={s['kills']:3d}  |  mod={s['avg_modules']:.2f}  |  {el:.1f}s"
            )

        if world.pop == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            break

    el = time.time() - start
    print(f"{'─' * 160}")
    print(f"Done in {el:.1f}s  |  Pop: {world.pop}  |  Total predation kills: {world.total_predation_kills}")

    if world.pop > 0:
        r = world.stats_history[-1]["roles"]
        mc = world.stats_history[-1]["module_counts"]
        s = world.stats_history[-1]
        print(f"Roles — Producers: {r['producer']}  Herbivores: {r['herbivore']}  "
              f"Carnivores: {r['carnivore']}  Detritivores: {r['detritivore']}  "
              f"Omnivores: {r['omnivore']}")
        print(f"Modules — PH:{mc['PHOTO']} CH:{mc['CHEMO']} CO:{mc['CONSUME']} "
              f"MV:{mc['MOVE']} FO:{mc['FORAGE']} DE:{mc['DEFENSE']} DT:{mc['DETOX']} "
              f"VR:{mc['VRESIST']} SO:{mc['SOCIAL']} ME:{mc['MEDIATE']} TP:{mc['TOXPROD']}")
        print(f"Social — avg_relationship: {s['avg_relationship']:.3f}  "
              f"max: {s['max_relationship']:.3f}  "
              f"avg_immune: {s['avg_immune_exp']:.3f}  "
              f"mediator: {s['mediator_field_mean']:.4f}  "
              f"nutrients: {s['nutrient_mean']:.3f}")
        print(f"Manipulation — lyso_fraction: {s['lyso_fraction']:.3f}  "
              f"total_manipulated_births: {s['manipulated_births']}  "
              f"hijack_fraction: {s['hijack_fraction']:.3f}  "
              f"total_hijacked_steps: {s['hijacked_steps']}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_summary.json"), 'w') as f:
        json.dump({"config": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                   "stats_history": world.stats_history}, f, indent=2)
    return world


if __name__ == "__main__":
    run_simulation()
