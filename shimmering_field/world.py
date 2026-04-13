"""World class — core simulation state and update loop."""

import numpy as np
from scipy.ndimage import uniform_filter

from .constants import *
from .config import Config
from .environment import EnvironmentMixin
from .energy import EnergyMixin
from .predation import PredationMixin
from .movement import MovementMixin
from .reproduction import ReproductionMixin
from .lifecycle import LifecycleMixin
from .viral import ViralMixin
from .social import SocialMixin
from .endosymbiosis import EndosymbiosisMixin
from .collapse import CollapseMixin
from .stats import StatsMixin
from .pivotal_moments import PivotalMomentMixin


class World(
    EnvironmentMixin, EnergyMixin, PredationMixin, MovementMixin,
    ReproductionMixin, LifecycleMixin, ViralMixin, SocialMixin,
    EndosymbiosisMixin, CollapseMixin, StatsMixin, PivotalMomentMixin,
):
    """The simulation world. Inherits subsystem methods via mixins."""

    _ORG_FIELDS = [
        "rows", "cols", "energy", "age", "generation", "ids", "parent_ids",
        "weights", "module_present", "module_active", "transfer_count",
        "viral_load", "lysogenic_strength", "lysogenic_genome",
        "immune_experience", "relationship_score", "merger_count", "module_usage",
        "genomic_stress", "genomic_cascade_phase",
        "dev_progress", "is_mature",
    ]

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

        # Ecosystem integrity and collapse state (hysteresis)
        self.ecosystem_integrity = np.ones((N, N)) * 0.5  # starts at moderate
        self.zone_collapsed = np.zeros((N, N), dtype=bool)

        # Fungal network (grid-level infrastructure, not organisms)
        self.fungal_density = np.zeros((N, N))

        # Prey scent fields: diffused signals for predator navigation
        self.producer_scent = np.zeros((N, N))  # herbivores follow this
        self.animal_scent = np.zeros((N, N))    # carnivores follow this

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
        self.total_mergers = 0

        # Pivotal moment tracking
        self._init_pivotal()

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

    def _filter_organisms(self, mask):
        for name in self._ORG_FIELDS:
            setattr(self, name, getattr(self, name)[mask])

    def _append_organisms(self, d):
        for name in self._ORG_FIELDS:
            setattr(self, name, np.concatenate([getattr(self, name), d[name]]))

    # ── Environment ──

    def update(self):
        if self.pop == 0:
            self._update_environment()
            self._update_prey_scent()
            self.timestep += 1
            self._record_stats(0)
            return

        self._update_density()
        self._update_prey_scent()
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

        # Genomic incompatibility: cascade effects on energy
        if self.timestep % self.cfg.genomic_check_interval == 0:
            self._apply_genomic_incompatibility(energy_gain)

        # Developmental dependency: update progress, apply compromise
        self._update_development()
        energy_gain *= self._compute_dev_compromise()

        self.energy = np.minimum(self.energy + energy_gain, effective_max)
        self._apply_toxic_damage(readings)
        self._apply_detox()
        base_costs = self._compute_total_costs()

        # Compounding stress: overlapping failure modes multiply costs
        compound = self._compute_compounding_stress()
        self.energy -= base_costs * compound

        # (Sessile crowding penalty removed — sessile producers compete through canopy/roots, not escape)

        self._execute_movement(self._decide_movement(readings))
        # Consumers get a second movement step every other tick — animals are faster
        if self.pop > 0 and self.timestep % 2 == 0:
            has_cons_move = self.module_active[:, M_CONSUME] & self.module_active[:, M_MOVE]
            if has_cons_move.any():
                readings2 = self._sense_local()
                actions2 = self._decide_movement(readings2)
                actions2[~has_cons_move] = 0
                self._execute_movement(actions2)
        # Developmental compromise accelerates aging
        age_mult = np.ones(self.pop, dtype=np.float64)
        past_window = (self.age > self.cfg.dev_window_length) & ~self.is_mature
        if past_window.any():
            deficit = 1.0 - np.minimum(self.dev_progress[past_window], 1.0)
            age_mult[past_window] = 1.0 + deficit * (self.cfg.dev_compromise_aging - 1.0)
        self.age += age_mult.astype(np.int32)

        # Social field update and interactions
        if self.timestep % self.cfg.social_update_interval == 0:
            self._update_social_field()
        self._apply_social_interactions()

        # Mediator field update (pollination service availability)
        if self.timestep % self.cfg.mediate_update_interval == 0:
            self._update_mediator_field()

        # Nutrient cycling (emergent from DETOX/CONSUME/FORAGE interactions)
        self._apply_nutrient_cycling()

        # Fungal network: growth, transport, and toxic conduit
        if self.timestep % self.cfg.fungal_update_interval == 0:
            self._update_fungal_network()
            self._fungal_nutrient_transport()
            self._fungal_toxic_conduit()

        # Ecosystem integrity and collapse state (nonlinear dynamics)
        if self.timestep % 5 == 0:
            self._update_ecosystem_integrity()
            self._update_collapse_state()

            # Collapse zone energy penalty
            if self.pop > 0:
                in_collapsed = self.zone_collapsed[self.rows, self.cols]
                if in_collapsed.any():
                    self.energy[in_collapsed] -= self.cfg.collapse_energy_penalty

        # Capacity shedding: use-it-or-lose-it (accelerated in collapsed zones)
        if self.timestep % self.cfg.shedding_check_interval == 0:
            self._update_module_usage(readings)
            # Accelerate shedding in collapsed zones
            if self.pop > 0:
                in_collapsed = self.zone_collapsed[self.rows, self.cols]
                if in_collapsed.any():
                    cidx = np.where(in_collapsed)[0]
                    extra_decay = SHEDDING_DECAY[None, :] * (self.cfg.collapse_shedding_mult - 1.0)
                    self.module_usage[cidx] -= extra_decay * self.module_present[cidx]
                    np.clip(self.module_usage[cidx], 0.0, 1.0, out=self.module_usage[cidx])
            self._apply_capacity_shedding()

        kills = 0
        if self.timestep % self.cfg.predation_check_interval == 0:
            kills = self._predation()
        if self.timestep % self.cfg.transfer_check_interval == 0:
            self._horizontal_transfer()
        if self.timestep % self.cfg.viral_check_interval == 0:
            self._viral_dynamics()

        # Endosymbiosis: rare merger events
        mergers = 0
        if self.timestep % self.cfg.endo_check_interval == 0:
            mergers = self._attempt_endosymbiosis()

        self._reproduce()
        self._kill_and_decompose()
        self._update_environment()
        self._record_stats(kills)
        self._detect_pivotal_moments()
        self.timestep += 1

    # ── Stats ──

