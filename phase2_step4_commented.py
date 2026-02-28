"""
The Shimmering Field — Phase 2 Step 4: Viral Archaeology
==========================================================

This is the most complex version of the simulation so far. It builds on
Phase 2 Step 1 (horizontal gene transfer from dead organisms) and adds
TWO major new systems:

═══════════════════════════════════════════════════════════════════════════════
SYSTEM 1: STRATIFIED TEMPORAL LAYERS ("Viral Archaeology")
═══════════════════════════════════════════════════════════════════════════════
Instead of a single "fragment pool" where dead organisms deposit DNA,
there are now THREE layers (strata), like geological layers in rock:

  ┌─────────────────────────────────────────────────────────┐
  │  RECENT STRATUM (top layer)                             │
  │  • Freshly deposited DNA from organisms that just died  │
  │  • Decays fastest                                       │
  │  • Always accessible to living organisms                │
  │  • Smallest evolutionary jumps (10% blend rate)         │
  ├─────────────────────────────────────────────────────────┤
  │  INTERMEDIATE STRATUM (middle layer)                    │
  │  • Material that "sedimented" down from recent          │
  │  • Decays at medium rate                                │
  │  • Only accessible when local toxicity >= 0.3           │
  │  • Medium evolutionary jumps (18% blend rate)           │
  ├─────────────────────────────────────────────────────────┤
  │  ANCIENT STRATUM (deepest layer)                        │
  │  • "Fossil DNA" from organisms that lived long ago      │
  │  • Decays very slowly (preserved like amber)            │
  │  • Only accessible when local toxicity >= 0.8           │
  │  • Biggest evolutionary jumps (30% blend rate)          │
  └─────────────────────────────────────────────────────────┘

The key insight: TOXICITY UNLOCKS ANCIENT DNA. When things get bad
(high toxicity), organisms gain access to fossil genomes from organisms
that may have lived under completely different conditions. This creates
a pressure-release valve — crisis conditions unlock radical adaptation
by exposing organisms to maximally divergent genetic material.

Material flows downward over time: recent → intermediate → ancient.
This is "sedimentation" — just like how geological layers form.

═══════════════════════════════════════════════════════════════════════════════
SYSTEM 2: VIRAL DYNAMICS (Lytic/Lysogenic cycle)
═══════════════════════════════════════════════════════════════════════════════
A virus-like system that can infect organisms in two ways:

  LYTIC INFECTION (aggressive):
    • Virus actively replicates inside the organism
    • Drains energy each step (like being sick)
    • Viral load grows over time
    • When viral load hits threshold (1.0), the organism BURSTS:
      - Organism dies instantly
      - Releases a cloud of viral particles in a 3-cell radius
      - Those particles can infect nearby organisms
    • Infected organisms CANNOT reproduce (too sick)

  LYSOGENIC INFECTION (dormant):
    • Virus integrates its genome into the organism silently
    • NO immediate damage — organism functions normally
    • The viral genome is stored as "lysogenic_genome"
    • Can be passed to offspring (vertical transmission, 80% strength)
    • Under TOXIC STRESS (toxicity > 0.6), the dormant virus can
      ACTIVATE → switches to lytic mode!
    • When activating, the viral genome BLENDS into the organism's
      expressed genome, potentially changing its behavior

  Two genome-based defenses:
    • Gene 14 (viral_resistance): Reduces infection probability
    • Gene 15 (lysogenic_suppression): Reduces activation probability

  Free viral particles float on the grid:
    • Spread to neighboring cells (diffusion)
    • Decay over time
    • Carry a "viral genome pool" — the average genome of organisms
      that burst, which gets transferred to newly infected organisms

This creates ANOTHER channel of evolution: viruses shuttle genes between
organisms, sometimes killing them (lytic) and sometimes silently modifying
their lineage across generations (lysogenic).

Built on Step 2 (lytic/lysogenic viral system).
"""

# ─── IMPORTS ───────────────────────────────────────────────────────────────────
import numpy as np                    # Fast array math
import json                           # Save data as JSON
import os                             # File system operations
import time                           # Measure runtime
from scipy.ndimage import uniform_filter  # Smooth zone boundaries


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Grid & Environment (same as Phase 2 Step 1) ────────────────────────
    grid_size = 128                   # 128x128 grid
    light_max = 1.0                   # Brightest at top
    light_min = 0.05                  # Dimmest at bottom
    zone_count = 8                    # 8x8 zone grid (64 zones)

    # ── Toxicity (unchanged) ───────────────────────────────────────────────
    toxic_decay_rate = 0.01           # 1% base decay per step
    toxic_diffusion_rate = 0.06       # 6% spread per step
    toxic_production_rate = 0.015     # Waste per organism per step
    toxic_threshold_low = 0.3         # Below = safe
    toxic_threshold_medium = 0.8      # Above = damage
    toxic_threshold_high = 1.5        # Above = rapid death
    toxic_damage_medium = 1.5         # Energy drain above medium
    toxic_damage_high = 5.0           # Energy drain above high
    toxic_photo_penalty = 1.0         # Photosynthesis reduction

    # ── Nutrients (unchanged) ──────────────────────────────────────────────
    nutrient_base_rate = 0.002
    nutrient_from_decomp = 0.4
    nutrient_max = 3.0

    # ── Organisms (unchanged) ─────────────────────────────────────────────
    initial_population = 80
    energy_initial = 40.0
    energy_max = 150.0
    energy_reproduction_threshold = 80.0
    energy_reproduction_cost = 40.0
    energy_maintenance_cost = 0.6
    energy_movement_cost = 0.2
    photosynthesis_base = 3.0

    # ── Genome Layout ──────────────────────────────────────────────────────
    # The genome has grown to 16 genes (was 14 in Step 1, 12 in Phase 1):
    #
    #   Genes 0-7:   Movement parameters (how to navigate the world)
    #   Genes 8-11:  Photosynthesis parameters (how to harvest energy)
    #   Genes 12-13: Transfer parameters (horizontal gene transfer behavior)
    #                  Gene 12: Receptivity (probability of absorbing fragments)
    #                  Gene 13: Selectivity (how similar fragments must be)
    #   Genes 14-15: Viral parameters (NEW — defense against virus)
    #                  Gene 14: viral_resistance (reduces infection probability)
    #                  Gene 15: lysogenic_suppression (reduces lysogenic activation)
    movement_params_size = 8
    photo_params_size = 4
    transfer_params_size = 2
    viral_params_size = 2             # [viral_resistance, lysogenic_suppression]
    genome_size = movement_params_size + photo_params_size + transfer_params_size + viral_params_size  # = 16

    # ── Evolution (unchanged) ─────────────────────────────────────────────
    mutation_rate = 0.08
    min_reproduction_age = 8
    offspring_distance = 5
    max_age = 200
    sensing_range = 3

    # ── Horizontal Gene Transfer (now per-stratum blend rates) ─────────────
    transfer_check_interval = 5       # Check every 5 steps

    # Each stratum has a DIFFERENT blend rate. Deeper = bigger changes.
    # This means absorbing ancient DNA causes a more radical genome shift
    # than absorbing recent DNA — ancient organisms were very different!
    transfer_blend_rate_recent = 0.10       # 10% blend from recent (safe, small change)
    transfer_blend_rate_intermediate = 0.18 # 18% blend from intermediate (moderate change)
    transfer_blend_rate_ancient = 0.30      # 30% blend from ancient (radical change!)

    decomp_fragment_decay = 0.005     # Fragment decay per step (recent layer)
    decomp_fragment_diffusion = 0.02  # Fragment spread rate

    # ── Stratified Archaeology (NEW) ───────────────────────────────────────
    # Controls how material flows between strata over time.
    #
    # Think of it like geological sedimentation:
    #   Fresh dead organisms → recent layer (top)
    #   Over time, recent material sinks → intermediate layer (middle)
    #   Over more time, intermediate sinks → ancient layer (bottom)
    #
    # Deeper layers decay slower (preserved like fossils).

    sedimentation_rate_recent = 0.005       # 0.5% of recent → intermediate per step
    sedimentation_rate_intermediate = 0.002 # 0.2% of intermediate → ancient per step
    ancient_decay_rate = 0.001              # Ancient material barely decays (0.1%/step)

    # Toxic concentration required to access deeper strata.
    # The idea: toxicity disrupts the "soil", exposing buried layers.
    # Safe conditions = you can only reach fresh DNA.
    # Dangerous conditions = you can dig into fossil DNA.
    stratum_access_medium = 0.3       # Toxicity >= 0.3 = access to intermediate
    stratum_access_high = 0.8         # Toxicity >= 0.8 = access to ancient

    # ── Viral System (NEW) ─────────────────────────────────────────────────
    # Parameters governing the virus-like particles that float on the grid
    # and can infect organisms.

    viral_decay_rate = 0.01           # Free viral particles decay 1% per step
    viral_diffusion_rate = 0.08       # Viral particles spread to neighbors (8%/step)
    viral_infection_rate = 0.3        # Base probability of infection per exposure
    viral_lytic_damage = 2.0          # Energy drained per step during lytic infection.
                                      # Multiplied by current viral_load, so damage
                                      # accelerates as infection progresses.

    viral_lytic_growth = 0.1          # Viral_load increases by 0.1 per step during lytic.
                                      # Starting from 0.1, it takes ~9 steps to reach
                                      # burst threshold of 1.0.

    viral_burst_threshold = 1.0       # When viral_load >= 1.0, organism BURSTS and dies.
    viral_burst_amount = 8.0          # Total viral particles released on burst.
                                      # Spread evenly across a 7x7 area (burst_radius=3).
    viral_burst_radius = 3            # Burst particles spread in a 3-cell radius.

    lysogenic_probability = 0.4       # 40% of new infections go lysogenic (dormant),
                                      # 60% go lytic (aggressive). This ratio determines
                                      # the "personality" of the viral system.

    lysogenic_activation_toxic = 0.6  # Lysogenic material activates (goes lytic) when
                                      # local toxicity exceeds this level. The virus
                                      # "senses" that the host is stressed.

    lysogenic_blend_rate = 0.1        # When lysogenic activates, 10% of the viral genome
                                      # blends into the organism's expressed genome.
                                      # Multiplied by lysogenic_strength for actual effect.

    lysogenic_inheritance = 0.8       # 80% of lysogenic material passes to offspring.
                                      # This means dormant viral DNA propagates across
                                      # generations — a form of vertical transfer.

    viral_check_interval = 3          # Run viral dynamics every 3 steps.
                                      # More frequent than horizontal transfer (every 5)
                                      # because viruses are faster-acting.

    # ── Simulation ─────────────────────────────────────────────────────────
    total_timesteps = 10000
    snapshot_interval = 100
    output_dir = "output_p2s4"
    random_seed = 42


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD
# ═══════════════════════════════════════════════════════════════════════════════

class World:
    def __init__(self, cfg=None):
        """
        Initialize the world with all environment grids, stratified fragment
        pools, viral particle field, and the starting organism population.
        """
        self.cfg = cfg or Config()
        c = self.cfg
        self.rng = np.random.default_rng(c.random_seed)
        self.timestep = 0
        self.next_id = 0
        N = c.grid_size  # 128

        # ══════════════════════════════════════════════════════════════════════
        # ENVIRONMENT GRIDS (same as before)
        # ══════════════════════════════════════════════════════════════════════
        self.light = np.linspace(c.light_max, c.light_min, N)[:, None] * np.ones((1, N))
        self.toxic = self.rng.uniform(0.0, 0.005, (N, N))
        self.nutrients = self.rng.uniform(0.02, 0.08, (N, N))
        self.decomposition = np.zeros((N, N))
        self.density = np.zeros((N, N), dtype=np.int32)

        # Zone map: random toxicity multipliers per zone, smoothed at edges
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
        # STRATIFIED GENOME FRAGMENT POOLS (replaces the single pool from Step 1)
        # ══════════════════════════════════════════════════════════════════════
        # Three layers of buried DNA, each with:
        #   - pool: 128x128x16 array = the average genome at each cell
        #   - weight: 128x128 array = how much material is at each cell
        #
        # Material flows: dead organisms → recent → intermediate → ancient

        self.strata_pool = {
            "recent":       np.zeros((N, N, c.genome_size)),  # Freshly deposited DNA
            "intermediate": np.zeros((N, N, c.genome_size)),  # Weeks-old DNA
            "ancient":      np.zeros((N, N, c.genome_size)),  # Fossil DNA
        }
        self.strata_weight = {
            "recent":       np.zeros((N, N)),  # How much fresh DNA per cell
            "intermediate": np.zeros((N, N)),  # How much medium-age DNA per cell
            "ancient":      np.zeros((N, N)),  # How much fossil DNA per cell
        }

        # ══════════════════════════════════════════════════════════════════════
        # VIRAL PARTICLE FIELD (NEW)
        # ══════════════════════════════════════════════════════════════════════
        # Free-floating viral particles on the grid. Think of them like a
        # fog of virus that drifts around and can infect organisms it touches.
        #
        # viral_particles: 128x128 grid of particle density.
        # viral_genome_pool: 128x128x16 — the "genome" carried by local viruses.
        #   When an organism bursts from lytic infection, its genome gets
        #   deposited into this pool. When new organisms get infected, the
        #   viral genome can integrate into them (lysogenic).
        # viral_genome_weight: 128x128 — strength of viral genome signal.

        self.viral_particles = np.zeros((N, N))

        # Seed 8 viral hotspots to bootstrap the system.
        # Without these starting hotspots, the viral system would never get going
        # because there'd be no particles to infect anyone.
        n_seeds = 8
        for _ in range(n_seeds):
            sr = self.rng.integers(0, N)          # Random center row
            sc_ = self.rng.integers(0, N)         # Random center column
            radius = 5                             # 11x11 hotspot area
            r0, r1 = max(0, sr - radius), min(N, sr + radius + 1)
            c0_, c1_ = max(0, sc_ - radius), min(N, sc_ + radius + 1)
            # Each hotspot gets a random amount of viral particles (0.5 to 2.0)
            self.viral_particles[r0:r1, c0_:c1_] += self.rng.uniform(0.5, 2.0)

        # Viral genome pool: starts with random genomes at hotspot locations
        self.viral_genome_pool = np.zeros((N, N, c.genome_size))
        self.viral_genome_weight = np.zeros((N, N))

        # Give the hotspot cells random viral genomes (so first infections carry DNA)
        hotspot_mask = self.viral_particles > 0.1
        n_hotspot_cells = int(hotspot_mask.sum())
        self.viral_genome_pool[hotspot_mask] = self.rng.normal(0, 0.5,
            (n_hotspot_cells, c.genome_size))
        self.viral_genome_weight[hotspot_mask] = 1.0

        # ══════════════════════════════════════════════════════════════════════
        # ORGANISM ARRAY REGISTRY
        # ══════════════════════════════════════════════════════════════════════
        # Now includes THREE new per-organism arrays for the viral system:
        #   viral_load:        How intense the lytic infection is (0 = not infected)
        #   lysogenic_strength: How much dormant viral DNA is integrated
        #   lysogenic_genome:  The actual dormant viral genome (2D: n × 16)

        self._org_arrays = [
            ("rows",              np.int64,   None),
            ("cols",              np.int64,   None),
            ("energy",            np.float64, None),
            ("age",               np.int32,   0),
            ("generation",        np.int32,   None),
            ("ids",               np.int64,   None),
            ("parent_ids",        np.int64,   None),
            ("genomes",           np.float64, None),      # 2D: (n, 16) — expressed genome
            ("transfer_count",    np.int32,   0),         # Times absorbed fragment DNA
            # ── Viral state (NEW) ──
            ("viral_load",        np.float64, 0.0),       # Lytic infection intensity.
                                                          # 0 = healthy. Grows toward 1.0.
                                                          # At 1.0, organism bursts.
            ("lysogenic_strength",np.float64, None),      # Amount of dormant viral DNA integrated.
                                                          # Higher = stronger latent infection.
                                                          # Must be explicitly set (no default).
            ("lysogenic_genome",  np.float64, None),      # 2D: (n, 16) — the dormant viral genome
                                                          # stored inside the organism. Gets blended
                                                          # into expressed genome on activation.
        ]

        # ══════════════════════════════════════════════════════════════════════
        # INITIALIZE STARTING POPULATION
        # ══════════════════════════════════════════════════════════════════════
        pop = c.initial_population
        self.rows = self.rng.integers(0, N, size=pop).astype(np.int64)
        self.cols = self.rng.integers(0, N, size=pop).astype(np.int64)
        self.energy = np.full(pop, c.energy_initial)
        self.age = np.zeros(pop, dtype=np.int32)
        self.generation = np.zeros(pop, dtype=np.int32)
        self.ids = np.arange(pop, dtype=np.int64)
        self.parent_ids = np.full(pop, -1, dtype=np.int64)
        self.genomes = self.rng.normal(0, 0.5, (pop, c.genome_size))  # 16-gene genome
        self.transfer_count = np.zeros(pop, dtype=np.int32)
        self.viral_load = np.zeros(pop, dtype=np.float64)             # All start healthy
        self.lysogenic_strength = np.zeros(pop, dtype=np.float64)     # No dormant viruses
        self.lysogenic_genome = np.zeros((pop, c.genome_size), dtype=np.float64)  # Empty
        self.next_id = pop

        # ══════════════════════════════════════════════════════════════════════
        # GLOBAL COUNTERS (for tracking system-level metrics)
        # ══════════════════════════════════════════════════════════════════════
        self.total_transfers = 0         # Total horizontal gene transfers ever
        self.transfers_by_stratum = {    # Transfers broken down by which layer they came from
            "recent": 0,
            "intermediate": 0,
            "ancient": 0
        }
        self.total_lytic_deaths = 0              # Organisms killed by viral burst
        self.total_lysogenic_integrations = 0    # Times virus went dormant in an organism
        self.total_lysogenic_activations = 0     # Times dormant virus woke up (went lytic)
        self.stats_history = []

    # ══════════════════════════════════════════════════════════════════════════
    # ORGANISM ARRAY HELPERS (same as Step 1)
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def pop(self):
        """Current population count."""
        return len(self.rows)

    def _filter_organisms(self, mask):
        """Keep only organisms where mask is True. Handles all registered arrays."""
        for name, _, _ in self._org_arrays:
            setattr(self, name, getattr(self, name)[mask])

    def _append_organisms(self, new_data: dict):
        """
        Add new organisms to the population. Uses the registry to auto-fill
        arrays with defaults when not explicitly provided.
        Handles both 1D arrays (energy, age) and 2D arrays (genomes, lysogenic_genome).
        """
        for name, dtype, default in self._org_arrays:
            existing = getattr(self, name)
            if name in new_data:
                addition = np.asarray(new_data[name], dtype=existing.dtype)
            elif default is not None:
                n = len(next(iter(new_data.values())))
                if existing.ndim > 1:
                    # 2D array (like genomes): create (n, genome_size) filled with default
                    addition = np.full((n, *existing.shape[1:]), default, dtype=existing.dtype)
                else:
                    # 1D array: create (n,) filled with default
                    addition = np.full(n, default, dtype=existing.dtype)
            else:
                raise ValueError(f"Must provide '{name}' in new_data (no default)")
            setattr(self, name, np.concatenate([existing, addition]))

    def _get_dead_data(self, dead_mask):
        """Extract all arrays for dead organisms (before removing them)."""
        return {name: getattr(self, name)[dead_mask] for name, _, _ in self._org_arrays}

    # ══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT UPDATE
    # ══════════════════════════════════════════════════════════════════════════

    def _update_environment(self):
        """
        Update ALL environment systems for one timestep:
          1. Toxic diffusion, decay, organism waste production
          2. Nutrient regeneration + decomposition cycle
          3. Stratified fragment sedimentation (recent → intermediate → ancient)
          4. Fragment decay (each layer at different rates)
          5. Fragment diffusion (every 10 steps)
          6. Viral particle decay + diffusion
          7. Viral genome pool maintenance
          8. Clamp all values to valid ranges
        """
        c = self.cfg

        # ── Toxic Diffusion + Decay (unchanged from Step 1) ──────────────────
        k = c.toxic_diffusion_rate
        p = np.pad(self.toxic, 1, mode='edge')
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        self.toxic += k * (nb - self.toxic)
        self.toxic *= (1.0 - c.toxic_decay_rate / np.maximum(0.3, self.zone_map))

        # Organisms produce toxic waste
        if self.pop > 0:
            np.add.at(self.toxic, (self.rows, self.cols),
                      c.toxic_production_rate * self.zone_map[self.rows, self.cols])

        # ── Nutrients + Decomposition (unchanged) ────────────────────────────
        self.nutrients += c.nutrient_base_rate
        xfer = self.decomposition * 0.015
        self.nutrients += xfer
        self.decomposition -= xfer
        self.decomposition *= 0.998

        # ══════════════════════════════════════════════════════════════════════
        # STRATIFIED SEDIMENTATION (NEW)
        # ══════════════════════════════════════════════════════════════════════
        # Material slowly "sinks" from shallow layers to deeper layers,
        # like sediment settling in water.
        #
        # recent → intermediate: 0.5% of recent layer transfers down per step
        # intermediate → ancient: 0.2% of intermediate transfers down per step
        #
        # The genome content blends weighted-average style: if the intermediate
        # layer already has weight 10 and we're adding weight 0.5 from recent,
        # the new intermediate genome = (10/10.5 * old) + (0.5/10.5 * recent).

        # ── Recent → Intermediate ────────────────────────────────────────────
        sed_r = c.sedimentation_rate_recent  # 0.005
        xfer_w = self.strata_weight["recent"] * sed_r  # Amount of weight transferring down
        self.strata_weight["recent"] -= xfer_w          # Remove from recent

        old_iw = self.strata_weight["intermediate"]     # Existing intermediate weight
        new_iw = old_iw + xfer_w                        # New intermediate weight
        safe_iw = np.maximum(new_iw, 1e-8)              # Avoid division by zero
        blend_i = xfer_w / safe_iw                      # Fraction of new material in mix

        # Weighted blend: existing intermediate genome + incoming recent genome
        # blend_i is high where lots of new material is arriving relative to existing
        self.strata_pool["intermediate"] = (
            self.strata_pool["intermediate"] * (1.0 - blend_i[:, :, None])
            + self.strata_pool["recent"] * blend_i[:, :, None])
        self.strata_weight["intermediate"] = new_iw

        # ── Intermediate → Ancient ───────────────────────────────────────────
        # Same process, but slower (0.2% per step)
        sed_i = c.sedimentation_rate_intermediate  # 0.002
        xfer_w2 = self.strata_weight["intermediate"] * sed_i
        self.strata_weight["intermediate"] -= xfer_w2

        old_aw = self.strata_weight["ancient"]
        new_aw = old_aw + xfer_w2
        safe_aw = np.maximum(new_aw, 1e-8)
        blend_a = xfer_w2 / safe_aw

        self.strata_pool["ancient"] = (
            self.strata_pool["ancient"] * (1.0 - blend_a[:, :, None])
            + self.strata_pool["intermediate"] * blend_a[:, :, None])
        self.strata_weight["ancient"] = new_aw

        # ── Stratum Decay (each layer decays at different rates) ─────────────
        # Recent decays fastest (fresh DNA degrades quickly)
        # Intermediate decays at half rate
        # Ancient barely decays at all (fossil preservation)
        self.strata_weight["recent"] *= (1.0 - c.decomp_fragment_decay)          # 0.5%/step
        self.strata_weight["intermediate"] *= (1.0 - c.decomp_fragment_decay * 0.5)  # 0.25%/step
        self.strata_weight["ancient"] *= (1.0 - c.ancient_decay_rate)             # 0.1%/step

        # ── Stratum Diffusion (every 10 steps, for performance) ──────────────
        # All three layers spread spatially, like DNA diffusing through soil.
        # Same pad-and-average technique used for toxic diffusion.
        if self.timestep % 10 == 0:
            k2 = c.decomp_fragment_diffusion * 10  # Compensate for running every 10 steps
            for sname in ("recent", "intermediate", "ancient"):
                sw = self.strata_weight[sname]
                if sw.max() < 0.001:
                    continue  # Skip if basically empty (saves computation)

                # Diffuse weights
                pw = np.pad(sw, 1, mode='edge')
                wn = (pw[:-2, 1:-1] + pw[2:, 1:-1] + pw[1:-1, :-2] + pw[1:-1, 2:]) / 4.0
                self.strata_weight[sname] += k2 * (wn - sw)
                self.strata_weight[sname] = np.maximum(self.strata_weight[sname], 0.0)

                # Diffuse genome content
                sp = self.strata_pool[sname]
                pp = np.pad(sp, ((1, 1), (1, 1), (0, 0)), mode='edge')
                gn = (pp[:-2, 1:-1, :] + pp[2:, 1:-1, :] + pp[1:-1, :-2, :] + pp[1:-1, 2:, :]) / 4.0
                self.strata_pool[sname] += k2 * (gn - sp)

        # ══════════════════════════════════════════════════════════════════════
        # VIRAL PARTICLE DYNAMICS (NEW)
        # ══════════════════════════════════════════════════════════════════════
        # Free-floating viral particles decay and spread, just like toxicity.

        # Decay: 1% of particles disappear per step
        self.viral_particles *= (1.0 - c.viral_decay_rate)

        # Diffusion: particles spread to neighbors (8% per step)
        kv = c.viral_diffusion_rate
        pv = np.pad(self.viral_particles, 1, mode='edge')
        vn = (pv[:-2, 1:-1] + pv[2:, 1:-1] + pv[1:-1, :-2] + pv[1:-1, 2:]) / 4.0
        self.viral_particles += kv * (vn - self.viral_particles)
        self.viral_particles = np.maximum(self.viral_particles, 0.0)

        # Viral genome pool follows particles (decay + diffuse every 10 steps)
        # This is the genetic payload carried by the virus — when organisms
        # burst, their genome gets encoded into this pool and spread.
        if self.timestep % 10 == 0 and self.viral_genome_weight.max() > 0.001:
            self.viral_genome_weight *= (1.0 - c.viral_decay_rate * 10)  # Compensated decay
            self.viral_genome_weight = np.maximum(self.viral_genome_weight, 0.0)
            # Diffuse viral genomes
            pp2 = np.pad(self.viral_genome_pool, ((1, 1), (1, 1), (0, 0)), mode='edge')
            gn2 = (pp2[:-2, 1:-1, :] + pp2[2:, 1:-1, :] + pp2[1:-1, :-2, :] + pp2[1:-1, 2:, :]) / 4.0
            self.viral_genome_pool += kv * (gn2 - self.viral_genome_pool)

        # ── Clamp Everything ─────────────────────────────────────────────────
        np.clip(self.toxic, 0, 5.0, out=self.toxic)
        np.clip(self.nutrients, 0, c.nutrient_max, out=self.nutrients)
        np.clip(self.decomposition, 0, 10.0, out=self.decomposition)
        np.clip(self.viral_particles, 0, 20.0, out=self.viral_particles)  # Cap at 20

    def _update_density(self):
        """Count organisms per cell."""
        self.density[:] = 0
        if self.pop > 0:
            np.add.at(self.density, (self.rows, self.cols), 1)

    # ══════════════════════════════════════════════════════════════════════════
    # SENSING (unchanged)
    # ══════════════════════════════════════════════════════════════════════════

    def _sense_local(self):
        """
        8 sensor readings per organism:
          [0] light, [1] toxicity, [2] nutrients, [3] density,
          [4-5] light gradients (vertical/horizontal),
          [6-7] toxicity gradients (vertical/horizontal)
        """
        rows, cols = self.rows, self.cols
        N = self.cfg.grid_size
        ru = np.clip(rows - 1, 0, N - 1)
        rd = np.clip(rows + 1, 0, N - 1)
        cl = np.clip(cols - 1, 0, N - 1)
        cr = np.clip(cols + 1, 0, N - 1)
        return np.column_stack([
            self.light[rows, cols], self.toxic[rows, cols],
            self.nutrients[rows, cols], self.density[rows, cols].astype(np.float64),
            self.light[ru, cols] - self.light[rd, cols],
            self.light[rows, cr] - self.light[rows, cl],
            self.toxic[ru, cols] - self.toxic[rd, cols],
            self.toxic[rows, cr] - self.toxic[rows, cl],
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # PHOTOSYNTHESIS, TOXIC DAMAGE, MOVEMENT (unchanged from Step 1)
    # ══════════════════════════════════════════════════════════════════════════

    def _photosynthesize(self, readings):
        """
        Energy from sunlight. Five multiplicative factors:
        base efficiency × toxic penalty × light sensitivity × storage × crowding.
        Uses genome genes 8-11. See Phase 1 comments for full detail.
        """
        c = self.cfg
        ph = self.genomes[:, c.movement_params_size:c.movement_params_size + c.photo_params_size]
        ll, lt, ld = readings[:, 0], readings[:, 1], readings[:, 3]
        eff = c.photosynthesis_base * (1.0 + 0.3 * np.tanh(ph[:, 0]))
        tol = np.maximum(0.1, 1.0 + 0.5 * np.tanh(ph[:, 1]))
        tp = np.maximum(0.0, 1.0 - lt * c.toxic_photo_penalty / tol)
        le = np.maximum(0.3, 1.0 - 0.3 * np.tanh(ph[:, 2]))
        lm = np.power(np.maximum(ll, 0.01), le)
        st = 0.5 + 0.5 / (1.0 + np.exp(-ph[:, 3]))
        sh = 1.0 / np.maximum(1.0, ld * 0.5)
        return eff * tp * lm * st * sh

    def _apply_toxic_damage(self, readings):
        """
        Direct energy damage from toxicity. Two stacking tiers:
        medium (>0.8) and high (>1.5). Creates natural carrying capacity.
        """
        c = self.cfg
        lt = readings[:, 1]
        dmg = np.zeros(self.pop)
        m = lt > c.toxic_threshold_medium
        if m.any(): dmg[m] += (lt[m] - c.toxic_threshold_medium) * c.toxic_damage_medium
        h = lt > c.toxic_threshold_high
        if h.any(): dmg[h] += (lt[h] - c.toxic_threshold_high) * c.toxic_damage_high
        self.energy -= dmg

    def _decide_movement(self, readings):
        """
        Score 5 actions (stay/up/down/right/left) using movement genes 0-7
        and sensor readings. Highest score wins. Includes exploration noise.
        """
        c = self.cfg
        n = self.pop
        if n == 0: return np.array([], dtype=np.int32)
        mp = self.genomes[:, :c.movement_params_size]
        sc = np.zeros((n, 5))
        sc[:, 0] = mp[:, 5] + mp[:, 2] * readings[:, 2] - mp[:, 1] * np.minimum(readings[:, 3], 10) * 0.1
        sc[:, 1] = mp[:, 6] * readings[:, 4] - mp[:, 7] * readings[:, 6]
        sc[:, 2] = -mp[:, 6] * readings[:, 4] + mp[:, 7] * readings[:, 6]
        sc[:, 3] = mp[:, 6] * readings[:, 5] - mp[:, 7] * readings[:, 7]
        sc[:, 4] = -mp[:, 6] * readings[:, 5] + mp[:, 7] * readings[:, 7]
        sc += mp[:, 4:5] * self.rng.normal(0, 0.3, (n, 5))
        return np.argmax(sc, axis=1)

    def _execute_movement(self, actions):
        """Move organisms. Clamp to grid. Deduct movement cost (0.2 energy)."""
        N = self.cfg.grid_size
        m = actions == 1; self.rows[m] = np.maximum(0, self.rows[m] - 1)
        m = actions == 2; self.rows[m] = np.minimum(N - 1, self.rows[m] + 1)
        m = actions == 3; self.cols[m] = np.minimum(N - 1, self.cols[m] + 1)
        m = actions == 4; self.cols[m] = np.maximum(0, self.cols[m] - 1)
        self.energy[actions > 0] -= self.cfg.energy_movement_cost

    # ══════════════════════════════════════════════════════════════════════════
    # HORIZONTAL GENE TRANSFER (now stratified)
    # ══════════════════════════════════════════════════════════════════════════

    def _horizontal_transfer(self):
        """
        Organisms absorb DNA fragments from the stratified decomposition layers.
        
        KEY CHANGE FROM STEP 1: Instead of one fragment pool, there are three
        strata (recent, intermediate, ancient) and access is GATED BY TOXICITY.
        
        Access rules:
          - Recent:       ALWAYS accessible (any organism can try)
          - Intermediate: Only if local toxicity >= 0.3 (mild stress)
          - Ancient:      Only if local toxicity >= 0.8 (severe stress)
        
        Each stratum also has a different blend rate and selectivity tolerance:
          - Recent:       10% blend, normal selectivity (safe, small changes)
          - Intermediate: 18% blend, 1.5x relaxed selectivity (bigger jumps)
          - Ancient:      30% blend, 2.5x relaxed selectivity (radical changes)
        
        This means: when things get bad (high toxicity), organisms can access
        increasingly ancient and divergent DNA, enabling more radical adaptation.
        It's like a pressure-release valve for evolution.
        
        The selectivity relaxation for deeper strata is important: ancient DNA
        is very different from modern organisms, so the normal selectivity filter
        would reject most of it. The depth_factor loosens this requirement.
        """
        c = self.cfg
        n = self.pop
        if n == 0: return

        # Extract the 2 transfer genes (indices 12-13)
        tp = self.genomes[:, c.movement_params_size + c.photo_params_size:
                          c.movement_params_size + c.photo_params_size + c.transfer_params_size]
        receptivity = 1.0 / (1.0 + np.exp(-tp[:, 0]))   # Sigmoid → 0-1 probability
        selectivity = np.abs(tp[:, 1])                    # How picky (always positive)

        # Check local toxicity for each organism (determines stratum access)
        local_toxic = self.toxic[self.rows, self.cols]

        # ── Determine stratum access masks ───────────────────────────────────
        can_recent = np.ones(n, dtype=bool)                           # Everyone
        can_intermediate = local_toxic >= c.stratum_access_medium     # Toxicity >= 0.3
        can_ancient = local_toxic >= c.stratum_access_high            # Toxicity >= 0.8

        # ── Process each stratum ─────────────────────────────────────────────
        # Each stratum has: (name, access_mask, blend_rate)
        strata_config = [
            ("recent",       can_recent,       c.transfer_blend_rate_recent),       # 10%
            ("intermediate", can_intermediate,  c.transfer_blend_rate_intermediate), # 18%
            ("ancient",      can_ancient,       c.transfer_blend_rate_ancient),      # 30%
        ]

        for sname, access_mask, blend_rate in strata_config:
            sw = self.strata_weight[sname]
            local_sw = sw[self.rows, self.cols]  # Fragment weight at each organism's cell

            # Three conditions to attempt absorption:
            #   1. Has access (based on toxicity level)
            #   2. Local material exists (weight > 0.1)
            #   3. Passes random receptivity check
            eligible = access_mask & (local_sw > 0.1) & (self.rng.random(n) < receptivity)
            eidx = np.where(eligible)[0]
            if len(eidx) == 0:
                continue  # Nobody eligible for this stratum

            # ── Selectivity filter (with depth adjustment) ───────────────────
            # Compute genetic distance between organism and fragment
            local_frags = self.strata_pool[sname][self.rows[eidx], self.cols[eidx]]
            dists = np.sqrt(np.mean((self.genomes[eidx] - local_frags) ** 2, axis=1))

            # Deeper strata have RELAXED selectivity — organisms accept
            # more divergent DNA from deeper layers.
            # Recent: threshold = 2.0 / (1 + selectivity)   ← normal
            # Intermediate: threshold = 3.0 / (1 + selectivity)  ← 1.5x more permissive
            # Ancient: threshold = 5.0 / (1 + selectivity)  ← 2.5x more permissive
            depth_factor = {"recent": 1.0, "intermediate": 1.5, "ancient": 2.5}[sname]
            thresh = (2.0 * depth_factor) / (1.0 + selectivity[eidx])

            # Only keep organisms whose distance is within their threshold
            tidx = eidx[dists < thresh]
            if len(tidx) == 0:
                continue

            # ── Perform genome blending ──────────────────────────────────────
            frags = self.strata_pool[sname][self.rows[tidx], self.cols[tidx]]
            self.genomes[tidx] = (1.0 - blend_rate) * self.genomes[tidx] + blend_rate * frags

            # ── Track transfers ──────────────────────────────────────────────
            self.transfer_count[tidx] += 1
            self.total_transfers += len(tidx)
            self.transfers_by_stratum[sname] += len(tidx)

    # ══════════════════════════════════════════════════════════════════════════
    # VIRAL DYNAMICS (THE BIG NEW SYSTEM)
    # ══════════════════════════════════════════════════════════════════════════

    def _viral_dynamics(self):
        """
        Simulate the viral infection cycle. Three main phases:
        
        Phase 1: NEW INFECTIONS
          - Uninfected organisms on cells with viral particles may get infected
          - Infection probability depends on: particle density × (1 - resistance)
          - New infections randomly become either LYTIC (60%) or LYSOGENIC (40%)
          - Lytic: viral_load starts at 0.1, begins draining energy
          - Lysogenic: viral genome integrates silently into lysogenic_genome
        
        Phase 2: LYTIC PROGRESSION
          - Already-infected organisms (viral_load > 0) get sicker each step
          - viral_load grows by 0.1 per step
          - Energy drain = 2.0 × current viral_load (accelerating damage)
          - When viral_load reaches 1.0, organism bursts (handled in _kill_and_decompose)
        
        Phase 3: LYSOGENIC ACTIVATION
          - Organisms with dormant viral DNA check if they should "wake up"
          - Activation requires: has lysogenic material + high toxicity (>0.6) + not already lytic
          - Probability depends on: stress level × (1 - lysogenic_suppression gene)
          - On activation:
            • Lysogenic genome blends into expressed genome (genome changes!)
            • viral_load starts at 0.2 (lytic infection begins)
            • lysogenic_strength partially consumed
        """
        c = self.cfg
        n = self.pop
        if n == 0: return

        # ── Extract viral defense genes ──────────────────────────────────────
        # Gene 14 (viral_resistance): sigmoid → 0-1, higher = more resistant
        # Gene 15 (lysogenic_suppression): sigmoid → 0-1, higher = better at
        #   keeping dormant viruses from activating
        vp_start = c.movement_params_size + c.photo_params_size + c.transfer_params_size  # Index 14
        viral_resistance = 1.0 / (1.0 + np.exp(-self.genomes[:, vp_start]))         # 0-1
        lysogenic_suppression = 1.0 / (1.0 + np.exp(-self.genomes[:, vp_start + 1]))  # 0-1

        # Read local environment
        local_viral = self.viral_particles[self.rows, self.cols]  # Virus particles at each organism
        local_toxic = self.toxic[self.rows, self.cols]             # Toxicity at each organism

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1: NEW INFECTIONS FROM FREE PARTICLES
        # ══════════════════════════════════════════════════════════════════════
        # Only uninfected organisms (viral_load == 0) can get newly infected.
        # They must be on a cell with enough viral particles (> 0.05).

        uninfected = self.viral_load == 0
        exposure = local_viral > 0.05
        candidates = uninfected & exposure  # Organisms that COULD get infected

        if candidates.any():
            cidx = np.where(candidates)[0]

            # Infection probability calculation:
            #   Base rate (0.3) × particle density (normalized to 0-1 via /5.0 cap)
            #   × resistance penalty (high resistance gene → lower probability)
            #   Resistance reduces probability by up to 80% (not 100% — nothing is perfect)
            inf_prob = c.viral_infection_rate * np.minimum(local_viral[cidx], 5.0) / 5.0
            inf_prob *= (1.0 - viral_resistance[cidx] * 0.8)

            # Random roll: each candidate independently rolls against their probability
            rolls = self.rng.random(len(cidx))
            infected = cidx[rolls < inf_prob]  # These organisms are now infected!

            if len(infected) > 0:
                # ── Split infections into lytic vs lysogenic ─────────────────
                # 40% go lysogenic (dormant), 60% go lytic (aggressive)
                lyso_rolls = self.rng.random(len(infected))
                goes_lytic = infected[lyso_rolls >= c.lysogenic_probability]      # 60%
                goes_lysogenic = infected[lyso_rolls < c.lysogenic_probability]   # 40%

                # ── LYTIC: Start active infection ────────────────────────────
                # Set viral_load to 0.1 (infection begins, will grow each step)
                if len(goes_lytic) > 0:
                    self.viral_load[goes_lytic] = 0.1

                # ── LYSOGENIC: Silently integrate viral genome ───────────────
                # The virus doesn't attack — instead, it stores its genome
                # inside the organism's lysogenic_genome. This DNA sits dormant
                # until toxic stress triggers activation.
                if len(goes_lysogenic) > 0:
                    # Check if there's viral genome material at this cell
                    vg_weight = self.viral_genome_weight[
                        self.rows[goes_lysogenic], self.cols[goes_lysogenic]]
                    has_material = vg_weight > 0.01
                    lyso_with_material = goes_lysogenic[has_material]

                    if len(lyso_with_material) > 0:
                        # Get the viral genome from the pool at these cells
                        viral_genomes = self.viral_genome_pool[
                            self.rows[lyso_with_material], self.cols[lyso_with_material]]

                        # Blend viral genome into the organism's lysogenic storage.
                        # Uses weighted averaging: new_strength = old + 0.3
                        # blend = 0.3 / new_strength (so new material has proportional influence)
                        old_str = self.lysogenic_strength[lyso_with_material]
                        new_str = old_str + 0.3  # Increase strength by 0.3 per integration
                        blend = 0.3 / np.maximum(new_str, 0.01)
                        self.lysogenic_genome[lyso_with_material] = (
                            self.lysogenic_genome[lyso_with_material] * (1.0 - blend[:, None])
                            + viral_genomes * blend[:, None])
                        self.lysogenic_strength[lyso_with_material] = new_str
                        self.total_lysogenic_integrations += len(lyso_with_material)

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2: LYTIC INFECTION PROGRESSION
        # ══════════════════════════════════════════════════════════════════════
        # Organisms currently fighting a lytic infection get sicker:
        #   - viral_load grows by 0.1 per step
        #   - Energy drain = 2.0 × viral_load (accelerating damage)
        #   - Example timeline:
        #       Step 0: load=0.1, drain=0.2
        #       Step 3: load=0.4, drain=0.8
        #       Step 6: load=0.7, drain=1.4
        #       Step 9: load=1.0 → BURST (dies, releases particles)

        lytic = self.viral_load > 0
        if lytic.any():
            lidx = np.where(lytic)[0]
            self.viral_load[lidx] += c.viral_lytic_growth                 # Load grows
            self.energy[lidx] -= c.viral_lytic_damage * self.viral_load[lidx]  # Damage accelerates

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 3: LYSOGENIC ACTIVATION UNDER TOXIC STRESS
        # ══════════════════════════════════════════════════════════════════════
        # Dormant viral DNA can "wake up" and go lytic when the organism
        # is stressed by high toxicity. This is inspired by real biology:
        # bacteriophages in lysogenic cycle activate when the host cell
        # is under stress (UV radiation, chemicals, etc.)
        #
        # Requirements for activation:
        #   1. Has lysogenic material (strength > 0.01)
        #   2. Local toxicity > 0.6 (organism is stressed)
        #   3. Not already in lytic infection
        #
        # When activated:
        #   - Lysogenic genome BLENDS into the organism's expressed genome
        #     (this is the genome-altering effect — the virus changes the organism!)
        #   - Lytic infection starts (viral_load = 0.2)
        #   - Lysogenic strength drops to 30% (partially consumed, not fully)

        has_lysogenic = self.lysogenic_strength > 0.01
        stressed = local_toxic > c.lysogenic_activation_toxic  # Toxicity > 0.6
        not_lytic = self.viral_load == 0
        activation_candidates = has_lysogenic & stressed & not_lytic

        if activation_candidates.any():
            acidx = np.where(activation_candidates)[0]

            # Activation probability: higher toxic stress + lower suppression = more likely
            # stress_level: how much above the threshold (normalized)
            # Capped at 50% max activation probability per step
            stress_level = (local_toxic[acidx] - c.lysogenic_activation_toxic) / c.lysogenic_activation_toxic
            act_prob = np.minimum(stress_level * 0.3, 0.5) * (1.0 - lysogenic_suppression[acidx] * 0.7)

            act_rolls = self.rng.random(len(acidx))
            activated = acidx[act_rolls < act_prob]

            if len(activated) > 0:
                # ── Activation: lysogenic → lytic with genome blending ───────
                # The big moment: dormant viral DNA merges into the organism's
                # actual expressed genome, potentially changing its behavior.
                # blend = 10% × lysogenic_strength, capped at 40%
                blend = c.lysogenic_blend_rate * self.lysogenic_strength[activated]
                blend = np.minimum(blend, 0.4)  # Never blend more than 40%

                self.genomes[activated] = (
                    self.genomes[activated] * (1.0 - blend[:, None])
                    + self.lysogenic_genome[activated] * blend[:, None])

                self.viral_load[activated] = 0.2               # Start lytic (slightly ahead)
                self.lysogenic_strength[activated] *= 0.3      # Mostly consumed
                self.total_lysogenic_activations += len(activated)

    # ══════════════════════════════════════════════════════════════════════════
    # REPRODUCTION
    # ══════════════════════════════════════════════════════════════════════════

    def _reproduce(self):
        """
        Asexual reproduction with two key differences from earlier phases:
        
        1. LYTIC-INFECTED ORGANISMS CANNOT REPRODUCE (viral_load must be 0).
           This is a huge evolutionary pressure: if you get infected, your
           lineage stops. Strong incentive to evolve viral_resistance.
        
        2. LYSOGENIC MATERIAL IS INHERITED by offspring (vertical transfer).
           80% of the parent's lysogenic_genome and lysogenic_strength passes
           to the child. This means dormant viruses propagate across generations,
           creating lineages that carry viral DNA — which might activate later
           under stress, changing the descendants' genomes.
        """
        c = self.cfg

        # Reproduction requires: enough energy + old enough + NOT lytically infected
        can = ((self.energy >= c.energy_reproduction_threshold) &
               (self.age >= c.min_reproduction_age) &
               (self.viral_load == 0))  # Sick organisms can't reproduce!
        pidx = np.where(can)[0]
        nb = len(pidx)
        if nb == 0: return

        # Parent pays reproduction cost
        self.energy[pidx] -= c.energy_reproduction_cost

        # Generate unique IDs for babies
        child_ids = np.arange(self.next_id, self.next_id + nb, dtype=np.int64)
        self.next_id += nb

        # ── Lysogenic inheritance ────────────────────────────────────────────
        # Children inherit 80% of parent's dormant viral DNA.
        # Over many generations, this dilutes unless reinforced by new infections.
        child_lyso_genome = self.lysogenic_genome[pidx] * c.lysogenic_inheritance    # 80%
        child_lyso_strength = self.lysogenic_strength[pidx] * c.lysogenic_inheritance  # 80%

        # Add all babies via the registry system
        self._append_organisms({
            "rows": np.clip(
                self.rows[pidx] + self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, nb),
                0, c.grid_size - 1),
            "cols": np.clip(
                self.cols[pidx] + self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, nb),
                0, c.grid_size - 1),
            "energy": np.full(nb, c.energy_initial),
            "generation": self.generation[pidx] + 1,
            "ids": child_ids,
            "parent_ids": self.ids[:self.pop][pidx],
            "genomes": self.genomes[pidx] + self.rng.normal(0, c.mutation_rate, (nb, c.genome_size)),
            "lysogenic_genome": child_lyso_genome,         # Inherited viral DNA
            "lysogenic_strength": child_lyso_strength,     # Inherited viral strength
        })
        # age, transfer_count, viral_load all default to 0 automatically

    # ══════════════════════════════════════════════════════════════════════════
    # DEATH & DECOMPOSITION (now with viral burst mechanics)
    # ══════════════════════════════════════════════════════════════════════════

    def _kill_and_decompose(self):
        """
        Remove dead organisms and handle their remains.
        
        THREE causes of death:
          1. VIRAL BURST: viral_load >= 1.0 → organism explodes, releasing particles
          2. STARVATION: energy <= 0
          3. OLD AGE: age >= 200
        
        All dead organisms deposit:
          - Energy → decomposition grid (becomes nutrients for survivors)
          - Genome → recent stratum of fragment pool (becomes absorbable DNA)
        
        VIRAL BURST organisms additionally:
          - Release viral particles in a 3-cell radius
          - Deposit their genome into the viral genome pool (future infections carry it)
        
        NATURAL DEATH organisms have a 15% chance of spontaneously releasing
        a small amount of viral particles. This keeps the viral system alive
        even during periods without active lytic infections.
        """
        c = self.cfg
        N = c.grid_size

        # ── Determine who dies and why ───────────────────────────────────────
        bursting = self.viral_load >= c.viral_burst_threshold  # Viral burst (load >= 1.0)
        natural_death = (self.energy <= 0) | (self.age >= c.max_age)  # Starvation or old age
        dead = bursting | natural_death  # Anyone who meets any death condition

        if dead.any():
            dd = self._get_dead_data(dead)  # Extract all data for dead organisms
            dr, dc = dd["rows"], dd["cols"]
            de = np.maximum(0, dd["energy"])
            dg = dd["genomes"]

            # ── Standard decomposition (all dead organisms) ──────────────────
            # Deposit nutrients: 40% of remaining energy + 0.5 base
            np.add.at(self.decomposition, (dr, dc), de * c.nutrient_from_decomp + 0.5)

            # ── Deposit genomes into RECENT stratum ──────────────────────────
            # Group dead organisms by cell, average their genomes, blend with
            # existing fragment pool (same technique as Step 1).
            cell_ids = dr * N + dc
            unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
            nc = len(unique_cells)
            genome_sums = np.zeros((nc, c.genome_size))
            counts = np.zeros(nc)
            np.add.at(genome_sums, inverse, dg)
            np.add.at(counts, inverse, 1.0)
            ur = unique_cells // N
            uc = unique_cells % N
            w0 = self.strata_weight["recent"][ur, uc]
            wt = w0 + counts
            avg_new = genome_sums / counts[:, None]
            blend = counts / wt
            self.strata_pool["recent"][ur, uc] = (
                self.strata_pool["recent"][ur, uc] * (1.0 - blend[:, None]) + avg_new * blend[:, None])
            self.strata_weight["recent"][ur, uc] = wt

            # ══════════════════════════════════════════════════════════════════
            # VIRAL BURST: Organisms that died from virus explode
            # ══════════════════════════════════════════════════════════════════
            # When an organism's viral_load reaches the burst threshold (1.0),
            # it dies and sprays viral particles in a 7x7 area (radius 3).
            # The particles carry the organism's genome — when they infect
            # new organisms, that genome can get integrated (lysogenic)
            # or just cause damage (lytic).

            burst_mask = dd["viral_load"] >= c.viral_burst_threshold

            if burst_mask.any():
                br = dd["rows"][burst_mask]      # Burst locations (rows)
                bc = dd["cols"][burst_mask]       # Burst locations (cols)
                bg = dd["genomes"][burst_mask]    # Burst organisms' genomes
                n_burst = len(br)
                self.total_lytic_deaths += n_burst

                # For each bursting organism, spread particles in a radius
                for i in range(n_burst):
                    # Calculate the rectangular area around the burst
                    r0 = max(0, br[i] - c.viral_burst_radius)
                    r1 = min(N, br[i] + c.viral_burst_radius + 1)
                    c0 = max(0, bc[i] - c.viral_burst_radius)
                    c1 = min(N, bc[i] + c.viral_burst_radius + 1)
                    area = (r1 - r0) * (c1 - c0)  # Number of cells in burst area

                    # Spread 8.0 total particles evenly across the burst area
                    self.viral_particles[r0:r1, c0:c1] += c.viral_burst_amount / area

                    # ── Deposit genome into viral genome pool ────────────────
                    # The burst organism's genome becomes the "payload" of the
                    # viral particles. Future infections will carry this genome.
                    # Uses weighted blending with existing viral genome pool.
                    w_old = self.viral_genome_weight[r0:r1, c0:c1]
                    w_add = c.viral_burst_amount / area
                    w_new = w_old + w_add
                    blend_v = w_add / np.maximum(w_new, 1e-8)  # Avoid /0

                    # Blend: existing viral genome × (1-blend) + burst genome × blend
                    self.viral_genome_pool[r0:r1, c0:c1] = (
                        self.viral_genome_pool[r0:r1, c0:c1] * (1.0 - blend_v[:, :, None])
                        + bg[i][None, None, :] * blend_v[:, :, None])
                    self.viral_genome_weight[r0:r1, c0:c1] = w_new

            # ── Spontaneous viral release from natural deaths ────────────────
            # Even organisms that die naturally (not from virus) have a 15%
            # chance of releasing a small amount of viral particles. This is
            # a background seeding mechanism that keeps the viral system from
            # going extinct during calm periods.
            natural_only = (~burst_mask) if burst_mask.any() else np.ones(len(dr), dtype=bool)
            if natural_only.any():
                spont_roll = self.rng.random(int(natural_only.sum()))
                spont_mask_local = spont_roll < 0.15  # 15% chance
                if spont_mask_local.any():
                    nat_idx = np.where(natural_only)[0][spont_mask_local]
                    # Add 1.0 particle + weight at each spontaneous release location
                    np.add.at(self.viral_particles,
                              (dd["rows"][nat_idx], dd["cols"][nat_idx]), 1.0)
                    np.add.at(self.viral_genome_weight,
                              (dd["rows"][nat_idx], dd["cols"][nat_idx]), 1.0)

        # ── Remove dead organisms from population ────────────────────────────
        self._filter_organisms(~dead)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN SIMULATION LOOP (one timestep)
    # ══════════════════════════════════════════════════════════════════════════

    def update(self):
        """
        Execute one timestep. The order of operations:
        
          1.  Count density (organisms per cell)
          2.  Sense local environment (8 readings per organism)
          3.  Photosynthesis (gain energy from light)
          4.  Toxic damage (lose energy in polluted areas)
          5.  Movement (decide direction and move)
          6.  Aging + maintenance cost
          7.  Horizontal transfer (every 5 steps — absorb DNA from strata)
          8.  Viral dynamics (every 3 steps — infections, progression, activation)
          9.  Reproduction (if healthy, energized, and old enough)
          10. Death + decomposition (remove dead, deposit remains + viral burst)
          11. Environment update (toxicity, nutrients, strata, viral particles)
        
        Steps 7 and 8 are the two new evolutionary channels:
          - Step 7: absorb ancient DNA from the ground (archaeology)
          - Step 8: viral infection can kill or silently alter genomes
        """
        if self.pop == 0:
            self._update_environment()
            self.timestep += 1
            self._record_stats()
            return

        self._update_density()                                                  # 1
        readings = self._sense_local()                                          # 2
        self.energy = np.minimum(                                               # 3
            self.energy + self._photosynthesize(readings), self.cfg.energy_max)
        self._apply_toxic_damage(readings)                                      # 4
        self._execute_movement(self._decide_movement(readings))                 # 5
        self.age += 1                                                           # 6a
        self.energy -= self.cfg.energy_maintenance_cost                         # 6b

        if self.timestep % self.cfg.transfer_check_interval == 0:               # 7
            self._horizontal_transfer()
        if self.timestep % self.cfg.viral_check_interval == 0:                  # 8
            self._viral_dynamics()

        self._reproduce()                                                       # 9
        self._kill_and_decompose()                                              # 10
        self._update_environment()                                              # 11
        self.timestep += 1
        self._record_stats()

    # ══════════════════════════════════════════════════════════════════════════
    # STATS & SNAPSHOTS
    # ══════════════════════════════════════════════════════════════════════════

    def _record_stats(self):
        """
        Record metrics for this timestep. New stats compared to Step 1:
          - strata_recent/intermediate/ancient: avg weight of each stratum layer
          - viral_mean/viral_max: viral particle density across grid
          - xfer_recent/intermediate/ancient: transfers per stratum (cumulative)
          - n_lytic: organisms currently fighting lytic infection
          - n_lysogenic: organisms carrying dormant viral DNA
          - total_lytic_deaths: cumulative organisms killed by viral burst
          - total_lyso_integrations: cumulative lysogenic infections
          - total_lyso_activations: cumulative dormant → lytic activations
        """
        p = self.pop
        c = self.cfg
        if p > 0:
            lt = self.toxic[self.rows, self.cols]
            in_low = int(np.sum(lt < c.toxic_threshold_low))
            in_med = int(np.sum((lt >= c.toxic_threshold_low) & (lt < c.toxic_threshold_medium)))
            in_high = int(np.sum(lt >= c.toxic_threshold_medium))
            xferd = int(np.sum(self.transfer_count > 0))
            n_lytic = int(np.sum(self.viral_load > 0))               # Currently lytic-infected
            n_lysogenic = int(np.sum(self.lysogenic_strength > 0.01)) # Carrying dormant virus
        else:
            in_low = in_med = in_high = xferd = n_lytic = n_lysogenic = 0

        self.stats_history.append({
            "t": self.timestep, "pop": p,
            "avg_energy": float(self.energy.mean()) if p > 0 else 0,
            "max_gen": int(self.generation.max()) if p > 0 else 0,
            "toxic_mean": float(self.toxic.mean()),
            "toxic_max": float(self.toxic.max()),
            "decomp_mean": float(self.decomposition.mean()),
            # Stratum weights (how much DNA is in each layer)
            "strata_recent": float(self.strata_weight["recent"].mean()),
            "strata_intermediate": float(self.strata_weight["intermediate"].mean()),
            "strata_ancient": float(self.strata_weight["ancient"].mean()),
            # Viral field
            "viral_mean": float(self.viral_particles.mean()),
            "viral_max": float(self.viral_particles.max()),
            # Toxicity zone distribution
            "in_low_toxic": in_low, "in_med_toxic": in_med, "in_high_toxic": in_high,
            # Horizontal transfer stats
            "orgs_with_transfers": xferd,
            "total_transfers": self.total_transfers,
            "xfer_recent": self.transfers_by_stratum["recent"],
            "xfer_intermediate": self.transfers_by_stratum["intermediate"],
            "xfer_ancient": self.transfers_by_stratum["ancient"],
            # Viral stats
            "n_lytic": n_lytic,
            "n_lysogenic": n_lysogenic,
            "total_lytic_deaths": self.total_lytic_deaths,
            "total_lyso_integrations": self.total_lysogenic_integrations,
            "total_lyso_activations": self.total_lysogenic_activations,
        })

    def save_snapshot(self, output_dir):
        """
        Save JSON snapshot with organism data.
        Now includes viral_load and lysogenic_strength per organism.
        """
        os.makedirs(output_dir, exist_ok=True)
        p = self.pop
        idx = self.rng.choice(p, min(p, 500), replace=False) if p > 500 else np.arange(p)
        orgs = [{
            "id": int(self.ids[i]),
            "row": int(self.rows[i]),
            "col": int(self.cols[i]),
            "energy": round(float(self.energy[i]), 2),
            "age": int(self.age[i]),
            "generation": int(self.generation[i]),
            "viral_load": round(float(self.viral_load[i]), 3),            # Lytic intensity
            "lysogenic_strength": round(float(self.lysogenic_strength[i]), 3),  # Dormant viral DNA
            "transfer_count": int(self.transfer_count[i])
        } for i in idx]
        s = self.stats_history[-1] if self.stats_history else {}
        with open(os.path.join(output_dir, f"snapshot_{self.timestep:06d}.json"), 'w') as f:
            json.dump({"timestep": self.timestep, "population": p, "organisms": orgs, "stats": s}, f)

    def save_env(self, output_dir):
        """
        Save environment grids. Now includes all three strata weights
        and viral particle density.
        """
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(os.path.join(output_dir, f"env_{self.timestep:06d}.npz"),
            toxic=self.toxic.astype(np.float32),
            decomposition=self.decomposition.astype(np.float32),
            density=self.density.astype(np.int16),
            strata_recent_w=self.strata_weight["recent"].astype(np.float32),
            strata_intermediate_w=self.strata_weight["intermediate"].astype(np.float32),
            strata_ancient_w=self.strata_weight["ancient"].astype(np.float32),
            viral_particles=self.viral_particles.astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(cfg=None):
    """
    Run the full simulation. Progress output now includes:
      - R/I/A: average weight of recent/intermediate/ancient strata
      - xR/xI/xA: cumulative transfers from each stratum
      - lyt/lys: current count of lytic-infected and lysogenic organisms
    """
    cfg = cfg or Config()
    world = World(cfg)

    # Print header with key parameters
    print(f"The Shimmering Field — Phase 2 Step 4: Viral Archaeology")
    print(f"Grid: {cfg.grid_size}x{cfg.grid_size}  |  Pop: {cfg.initial_population}  |  Genome: {cfg.genome_size}")
    print(f"Strata access: recent=always  intermediate=tox>{cfg.stratum_access_medium}  ancient=tox>{cfg.stratum_access_high}")
    print(f"{'─' * 115}")

    start = time.time()

    for t in range(cfg.total_timesteps):
        world.update()

        # Print progress every 100 steps
        if world.timestep % cfg.snapshot_interval == 0:
            world.save_snapshot(cfg.output_dir)
            s = world.stats_history[-1]
            el = time.time() - start
            print(f"  t={s['t']:5d}  |  pop={s['pop']:5d}  |  e={s['avg_energy']:5.1f}  |  "
                  f"gen={s['max_gen']:4d}  |  tox={s['toxic_mean']:.3f}  |  "
                  f"R={s['strata_recent']:.2f} I={s['strata_intermediate']:.2f} A={s['strata_ancient']:.2f}  |  "
                  f"xR={s['xfer_recent']:5d} xI={s['xfer_intermediate']:5d} xA={s['xfer_ancient']:5d}  |  "
                  f"lyt={s['n_lytic']:3d} lys={s['n_lysogenic']:4d}  |  {el:.1f}s")

        # Save full environment grids every 500 steps
        if world.timestep % 500 == 0:
            world.save_env(cfg.output_dir)

        # Handle extinction
        if world.pop == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            break

    # ── Wrap up ──────────────────────────────────────────────────────────────
    el = time.time() - start
    print(f"{'─' * 115}")
    tbs = world.transfers_by_stratum
    print(f"Done in {el:.1f}s  |  Pop: {world.pop}  |  Lytic deaths: {world.total_lytic_deaths}")
    print(f"Transfers — Recent: {tbs['recent']}  Intermediate: {tbs['intermediate']}  Ancient: {tbs['ancient']}")

    # Save summary JSON
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_summary.json"), 'w') as f:
        json.dump({
            "config": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
            "stats_history": world.stats_history
        }, f, indent=2)

    return world


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_simulation()
