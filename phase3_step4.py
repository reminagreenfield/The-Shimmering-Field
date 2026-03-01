"""
The Shimmering Field — Phase 3 Step 4: FORAGE + DEFENSE + DETOX
============================================================
Activates three new modules simultaneously, completing the basic module set.

New modules:
  - FORAGE:  [extraction_eff, resource_discrim, storage_cap, cooperative_signal]
             Enhances resource gathering, enables energy storage, proto-mutualism
  - DEFENSE: [shell, camouflage, size_invest, counter_attack]
             Reduces predation vulnerability, creates predator-prey arms race
  - DETOX:   [detox_eff, toxin_tolerance, conversion_rate, selective_uptake]
             Active toxin metabolism, opens toxic zones to non-CHEMO organisms

Key interactions:
  - DEFENSE vs CONSUME: arms race — predators need more aggression to overcome defense
  - DETOX + CHEMO: synergy in toxic environments
  - DETOX + PHOTO: opens toxic zones to photosynthesizers
  - FORAGE + producers: better resource extraction, cooperative nutrient boost
  - DEFENSE cost: shell/size maintenance makes defended organisms slower to reproduce

Built on Phase 3 Step 3 (CONSUME module).
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

N_MODULES = 9

MODULE_NAMES = [
    "PHOTO", "CHEMO", "CONSUME", "MOVE", "FORAGE",
    "DEFENSE", "DETOX", "TOXPROD", "VRESIST"
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
    0,  # VRESIST: (future)
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
#                PH    CH    CO    MV    FO    DE    DT    TP    VR
MODULE_MAINTENANCE = np.array([0.20, 0.25, 0.12, 0.08, 0.12, 0.18, 0.20, 0.03, 0.15])
MODULE_EXPRESSION  = np.array([0.10, 0.12, 0.08, 0.06, 0.06, 0.10, 0.10, 0.02, 0.08])
BASE_MAINTENANCE = 0.05
# Cost examples:
#   PHOTO+MOVE+TOXPROD                    = 0.54
#   + FORAGE (efficient producer)         = 0.72
#   + DEFENSE (armored producer)          = 1.00
#   + CONSUME (all-in generalist)         = 1.20
#   + DETOX (toxic zone specialist)       = 1.50

GAINABLE_MODULES = [M_CHEMO, M_CONSUME, M_MOVE, M_FORAGE, M_DEFENSE, M_DETOX]


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
    initial_detritivore_fraction = 0.12
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
    producer_consume_penalty = 0.55  # omnivores hunt at 55% effectiveness
    consume_producer_penalty = 0.90  # omnivores produce at 90% effectiveness
    consumer_specialist_bonus = 0.5  # +50% for obligate consumers

    # Predation
    predation_check_interval = 1
    predation_base_success = 0.14
    predation_energy_fraction = 0.55
    predation_hunt_radius_base = 1
    predation_hunt_radius_mobile = 2
    predation_max_kills_per_cell = 1  # prevents local extinction cascades
    predator_satiation = 0.7  # probability a fed predator skips next hunt

    # Decomposition
    decomp_death_deposit = 3.5
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

    # Evolution
    mutation_rate = 0.08
    module_gain_rate = 0.01       # doubled — more modules available now
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
    output_dir = "output_p3s4_tuned"
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
        self.decomposition[mid-15:mid+15, mid-15:mid+15] += self.rng.uniform(1.0, 4.0, (30, 30))

        # Viral/HGT state
        self.transfer_count = np.zeros(pop, dtype=np.int32)
        self.viral_load = np.zeros(pop, dtype=np.float64)
        self.lysogenic_strength = np.zeros(pop, dtype=np.float64)
        self.lysogenic_genome = np.zeros((pop, TOTAL_WEIGHT_PARAMS), dtype=np.float64)

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

            # Target selection: prefer weaker prey
            cands = np.array(candidates)
            target_e = np.maximum(self.energy[cands], 1.0)
            inv_e = 1.0 / target_e
            inv_e /= inv_e.sum()
            ti = cands[self.rng.choice(len(cands), p=inv_e)]

            # Success probability
            target_difficulty = np.clip(self.energy[ti] / c.energy_max, 0.1, 1.0)
            prob = (base_prob[pi] - target_difficulty * 0.08) * omni_mult[pi]
            prob -= shell_val[ti]
            prob = np.clip(prob, 0.01, 0.60)

            if hunt_rolls[pi_idx] < prob:
                gained = max(0.0, self.energy[ti]) * frac_base[pi]
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

        # Directional scores
        lg_y, lg_x = readings[:, 4], readings[:, 5]
        tg_y, tg_x = readings[:, 6], readings[:, 7]
        dg_y, dg_x = readings[:, 9], readings[:, 10]
        sg_y, sg_x = readings[:, 11], readings[:, 12]
        ng_y, ng_x = readings[:, 13], readings[:, 14]

        gy = (light_w * lg_y - net_toxic * tg_y + consume_dens_w * dg_y
              + consume_scent_w * sg_y + forage_nutr_w * ng_y + detox_toxic_w * tg_y)
        gx = (light_w * lg_x - net_toxic * tg_x + consume_dens_w * dg_x
              + consume_scent_w * sg_x + forage_nutr_w * ng_x + detox_toxic_w * tg_x)
        sc[:, 1] = gy;  sc[:, 2] = -gy
        sc[:, 3] = gx;  sc[:, 4] = -gx

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

    # ── Viral Dynamics ──

    def _viral_dynamics(self):
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        sp = self._standalone_params()
        viral_resistance = 1.0 / (1.0 + np.exp(-sp[:, SP_VIRAL_RESISTANCE]))
        lysogenic_suppression = 1.0 / (1.0 + np.exp(-sp[:, SP_LYSO_SUPPRESSION]))
        local_viral = self.viral_particles[self.rows, self.cols]
        local_toxic = self.toxic[self.rows, self.cols]

        # New infections
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

        # Lytic damage
        lytic_mask = self.viral_load > 0
        if lytic_mask.any():
            lidx = np.where(lytic_mask)[0]
            self.viral_load[lidx] += c.viral_lytic_growth
            self.energy[lidx] -= c.viral_lytic_damage * self.viral_load[lidx]

        # Lysogenic activation under stress
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

    # ── Reproduction ──

    def _reproduce(self):
        c = self.cfg
        N = c.grid_size

        # Density-dependent threshold: harder to reproduce in crowded areas
        local_dens = self.density[self.rows, self.cols].astype(np.float64)
        effective_threshold = c.energy_reproduction_threshold + local_dens * c.repro_density_penalty

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

        effective_max = self._effective_energy_max()
        self.energy = np.minimum(self.energy + self._acquire_energy(readings), effective_max)
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
            return {"producer": 0, "carnivore": 0, "detritivore": 0, "omnivore": 0}

        has_prod = self.module_present[:, M_PHOTO] | self.module_present[:, M_CHEMO]
        has_cons = self.module_present[:, M_CONSUME]
        producer = has_prod & ~has_cons
        omnivore = has_prod & has_cons
        obligate = has_cons & ~has_prod

        carnivore = detritivore = np.zeros(n, dtype=bool)
        if obligate.any():
            uw = self._module_weights(M_CONSUME)
            dp = 1.0 / (1.0 + np.exp(-uw[:, 2]))
            carnivore = obligate & (dp < 0.5)
            detritivore = obligate & (dp >= 0.5)

        return {
            "producer": int(producer.sum()),
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
            })
        else:
            self.stats_history.append({
                "t": self.timestep, "pop": 0, "avg_energy": 0, "max_gen": 0,
                "toxic_mean": round(float(self.toxic.mean()), 3),
                "decomp_mean": round(float(self.decomposition.mean()), 2),
                "avg_modules": 0, "module_counts": {MODULE_NAMES[m]: 0 for m in range(N_MODULES)},
                "roles": {"producer": 0, "carnivore": 0, "detritivore": 0, "omnivore": 0},
                "kills": 0, "total_kills": self.total_predation_kills,
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
