"""
The Shimmering Field — Phase 2 Step 1 (Refactored)
====================================================

This builds on Phase 1 v3 (the basic artificial life simulation) and adds
TWO major new features:

1. ORGANISM ARRAY REGISTRY — A cleaner architecture for managing organism data.
   In Phase 1, every time we added a new property to organisms (like a new gene
   or a counter), we had to manually update the reproduction code, the death
   code, the filter code, etc. Now there's a central registry that lists ALL
   organism arrays. Adding a new property is just one line — the reproduction
   and death code automatically handles it.

2. HORIZONTAL GENE TRANSFER — When organisms die, their genomes don't just
   vanish. Instead, fragments of their DNA are deposited into a "fragment pool"
   on the ground at the cell where they died. Living organisms can then
   ABSORB these fragments, blending the dead organism's genes into their own
   genome. This is inspired by real biology: bacteria can pick up DNA from
   their environment (from dead cells) and incorporate it into their own genome.
   This creates a second channel of evolution beyond parent-to-child inheritance.

Everything else (light gradient, toxicity mechanics, photosynthesis, movement,
reproduction, death) works the same as Phase 1 v3.

Mechanics: tuned ecology + horizontal transfer from decomposition layer.
"""

# ─── IMPORTS ───────────────────────────────────────────────────────────────────
import numpy as np          # Fast array math (process all organisms at once)
import json                 # Save data as human-readable JSON files
import os                   # File system operations (create folders, paths)
import time                 # Measure simulation runtime
from scipy.ndimage import uniform_filter  # Smooth the zone map boundaries


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# All tunable parameters for the simulation. Same idea as Phase 1 Config,
# but with a few additions for horizontal gene transfer.

class Config:
    # ── Grid ───────────────────────────────────────────────────────────────────
    # Smaller grid than Phase 1 (128x128 instead of 256x256 = 16,384 cells).
    # This makes the simulation faster while still being large enough for
    # interesting spatial dynamics.
    grid_size = 128

    # ── Light Gradient ─────────────────────────────────────────────────────────
    # Light is brightest at the top (row 0) and dimmest at the bottom (row 127).
    # This creates a natural "north = good" incentive for organisms.
    light_max = 1.0       # Full brightness at top
    light_min = 0.05      # Near-dark at bottom

    # ── Zone Map ───────────────────────────────────────────────────────────────
    # 8x8 = 64 zones, each 16x16 cells. Each zone randomly absorbs or amplifies
    # toxicity, creating spatial diversity (safe havens and danger zones).
    zone_count = 8

    # ── Toxicity ───────────────────────────────────────────────────────────────
    # Same three-threshold system as Phase 1:
    #   Below 0.3: safe. Between 0.3-0.8: stressful. Above 0.8: damaging.
    #   Above 1.5: lethal crisis.
    toxic_decay_rate = 0.01           # 1% base decay per step (slow — toxicity lingers)
    toxic_diffusion_rate = 0.06       # 6% spread to neighbors per step
    toxic_production_rate = 0.015     # Each organism produces this much waste per step
    toxic_threshold_low = 0.3         # Below this = safe
    toxic_threshold_medium = 0.8      # Above this = organisms take damage
    toxic_threshold_high = 1.5        # Above this = rapid death
    toxic_damage_medium = 1.5         # Energy loss per unit excess above medium threshold
    toxic_damage_high = 5.0           # Energy loss per unit excess above high threshold
    toxic_photo_penalty = 1.0         # How much toxicity reduces photosynthesis

    # ── Nutrients ──────────────────────────────────────────────────────────────
    nutrient_base_rate = 0.002        # Tiny natural nutrient trickle each step
    nutrient_from_decomp = 0.4        # 40% of dead organism's energy becomes nutrients
    nutrient_max = 3.0                # Maximum nutrients per cell

    # ── Organisms ──────────────────────────────────────────────────────────────
    initial_population = 80           # Starting population (slightly less than Phase 1's 100)
    energy_initial = 40.0             # Starting energy for newborns
    energy_max = 150.0                # Energy cap
    energy_reproduction_threshold = 80.0  # Must have >= 80 energy to reproduce
    energy_reproduction_cost = 40.0   # Energy spent to create one baby
    energy_maintenance_cost = 0.6     # Energy cost per step just for being alive
    energy_movement_cost = 0.2        # Extra energy cost for moving
    photosynthesis_base = 3.0         # Base energy from photosynthesis

    # ── Genome ─────────────────────────────────────────────────────────────────
    # The genome is now 14 genes (was 12 in Phase 1):
    #   Genes 0-7:   Movement parameters (same as Phase 1)
    #   Genes 8-11:  Photosynthesis parameters (same as Phase 1)
    #   Genes 12-13: Transfer parameters (NEW — controls horizontal gene transfer)
    #     Gene 12: "Receptivity" — how likely is this organism to absorb foreign DNA?
    #     Gene 13: "Selectivity" — how similar must the foreign DNA be to absorb it?
    movement_params_size = 8          # 8 movement genes
    photo_params_size = 4             # 4 photosynthesis genes
    transfer_params_size = 2          # 2 horizontal transfer genes (NEW)
    genome_size = movement_params_size + photo_params_size + transfer_params_size  # = 14

    # ── Evolution ──────────────────────────────────────────────────────────────
    mutation_rate = 0.08              # Random noise added to genes during reproduction
    min_reproduction_age = 8          # Must be 8+ steps old to reproduce
    offspring_distance = 5            # Babies appear within 5 cells of parent
    max_age = 200                     # Die of old age at 200 steps
    sensing_range = 3                 # (Not directly used — sensing is immediate neighbors)

    # ── Horizontal Gene Transfer Settings (NEW) ───────────────────────────────
    # These control the new "pick up DNA from dead organisms" mechanic.
    #
    # How it works:
    #   1. When organisms die, their genome is deposited into a "fragment pool"
    #      at their cell location (like DNA floating in the soil)
    #   2. Every 5 steps, living organisms can try to absorb nearby fragments
    #   3. If they absorb, their genome blends with the fragment (15% fragment, 85% self)
    #   4. Fragments slowly decay and spread to neighboring cells over time
    #
    # This mimics bacterial horizontal gene transfer — a way for organisms to
    # acquire genes WITHOUT reproduction. It can accelerate evolution by allowing
    # useful genes from dead organisms to spread to survivors.

    transfer_check_interval = 5       # Check for transfer every 5 steps (not every step,
                                      # to save computation and make it less dominant)

    transfer_blend_rate = 0.15        # When transfer happens, the organism's genome becomes
                                      # 85% its own genes + 15% the absorbed fragment.
                                      # Small enough to not overwrite the organism's identity,
                                      # but large enough to have a meaningful effect.

    decomp_fragment_decay = 0.005     # Fragment pool strength decays by 0.5% per step.
                                      # Fragments don't last forever — old DNA degrades.

    decomp_fragment_diffusion = 0.02  # Fragments slowly spread to neighboring cells,
                                      # like DNA diffusing through the soil.

    # ── Simulation ─────────────────────────────────────────────────────────────
    total_timesteps = 10000           # Run for 10,000 steps
    snapshot_interval = 100           # Save a snapshot every 100 steps
    output_dir = "output_p2s1r"       # Output folder name ("p2s1r" = Phase 2, Step 1, Refactored)
    random_seed = 42                  # Fixed seed for reproducibility


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD
# ═══════════════════════════════════════════════════════════════════════════════
# The main simulation class. Holds all environment and organism data,
# plus all the logic for each timestep.

class World:
    def __init__(self, cfg=None):
        """
        Initialize the world: create environment grids, set up the organism
        registry, place initial organisms, and initialize the fragment pool.
        """
        self.cfg = cfg or Config()
        c = self.cfg
        self.rng = np.random.default_rng(c.random_seed)  # Reproducible RNG
        self.timestep = 0
        self.next_id = 0       # Counter for unique organism IDs
        N = c.grid_size        # 128

        # ══════════════════════════════════════════════════════════════════════
        # ENVIRONMENT GRIDS (same as Phase 1)
        # ══════════════════════════════════════════════════════════════════════

        # Light: gradient from bright (top) to dim (bottom)
        # Creates a 128x128 grid where each row has the same brightness
        self.light = np.linspace(c.light_max, c.light_min, N)[:, None] * np.ones((1, N))

        # Toxicity: starts near zero, will build up from organism waste
        self.toxic = self.rng.uniform(0.0, 0.005, (N, N))

        # Nutrients: small random starting amounts
        self.nutrients = self.rng.uniform(0.02, 0.08, (N, N))

        # Decomposition: organic matter from dead organisms (starts empty)
        self.decomposition = np.zeros((N, N))

        # Density: how many organisms per cell (recalculated each step)
        self.density = np.zeros((N, N), dtype=np.int32)

        # ── Zone Map ─────────────────────────────────────────────────────────
        # Divide grid into 8x8 zones, each with a random toxicity multiplier.
        # Values < 1 absorb toxicity (safe), values > 1 amplify it (dangerous).
        zone_size = N // c.zone_count  # 128 / 8 = 16 cells per zone
        self.zone_map = np.ones((N, N))
        for zi in range(c.zone_count):
            for zj in range(c.zone_count):
                r0, r1 = zi * zone_size, (zi + 1) * zone_size
                c0, c1 = zj * zone_size, (zj + 1) * zone_size
                # Randomly choose: absorbing (0.3-0.7), neutral (1.0), or amplifying (1.2-2.0)
                self.zone_map[r0:r1, c0:c1] = self.rng.choice(
                    [0.3, 0.5, 0.7, 1.0, 1.0, 1.2, 1.5, 2.0])
        # Smooth zone boundaries so transitions aren't abrupt
        self.zone_map = uniform_filter(self.zone_map, size=8)

        # ══════════════════════════════════════════════════════════════════════
        # FRAGMENT POOL (NEW IN PHASE 2)
        # ══════════════════════════════════════════════════════════════════════
        # When organisms die, their genome gets deposited here. Think of it as
        # DNA fragments floating in the soil at each cell.
        #
        # fragment_pool: A 128x128x14 array. Each cell holds a "representative
        #   genome" — the weighted average of all genomes deposited there.
        #   Shape: (grid_size, grid_size, genome_size)
        #
        # fragment_weight: A 128x128 array. Tracks how much genetic material
        #   has accumulated at each cell. Higher weight = more DNA fragments
        #   = more likely for organisms to pick something up.
        #   Weight of 0 = no fragments here. Weight of 5 = lots of DNA.

        self.fragment_pool = np.zeros((N, N, c.genome_size))  # Average genome at each cell
        self.fragment_weight = np.zeros((N, N))                # Strength of fragments at each cell

        # ══════════════════════════════════════════════════════════════════════
        # ORGANISM ARRAY REGISTRY (NEW ARCHITECTURE)
        # ══════════════════════════════════════════════════════════════════════
        # This is a KEY improvement over Phase 1. Instead of manually handling
        # each organism array in reproduce/kill/filter, we maintain a REGISTRY
        # that lists every per-organism array.
        #
        # Each entry is a tuple: (attribute_name, numpy_dtype, default_value)
        #   - attribute_name: The name of the array on self (e.g., "rows")
        #   - dtype: The numpy data type (int64, float64, etc.)
        #   - default_value: Value to use for new organisms if not explicitly
        #     provided. None means "you MUST provide this value" (no default).
        #
        # WHY THIS MATTERS:
        #   In Phase 1, adding a new organism property (like transfer_count)
        #   would require editing _reproduce(), _kill_and_decompose(), and
        #   any filtering code. With the registry, you just add one line here,
        #   and _append_organisms() / _filter_organisms() / _get_dead_data()
        #   handle it automatically. This makes Phase 2+ much easier to extend.

        self._org_arrays = [
            ("rows",           np.int64,   None),    # Grid row position (must provide)
            ("cols",           np.int64,   None),    # Grid column position (must provide)
            ("energy",         np.float64, None),    # Current energy level (must provide)
            ("age",            np.int32,   0),       # Age in timesteps (defaults to 0 for babies)
            ("generation",     np.int32,   None),    # Generation number (must provide)
            ("ids",            np.int64,   None),    # Unique organism ID (must provide)
            ("parent_ids",     np.int64,   None),    # Parent's ID (must provide)
            ("genomes",        np.float64, None),    # Genome array, 2D: (n, 14) (must provide)
            ("transfer_count", np.int32,   0),       # How many times this organism has absorbed
                                                     # foreign DNA (defaults to 0 for babies)
        ]

        # ══════════════════════════════════════════════════════════════════════
        # INITIALIZE STARTING POPULATION
        # ══════════════════════════════════════════════════════════════════════
        pop = c.initial_population  # 80 organisms
        self.rows = self.rng.integers(0, N, size=pop).astype(np.int64)  # Random positions
        self.cols = self.rng.integers(0, N, size=pop).astype(np.int64)
        self.energy = np.full(pop, c.energy_initial)          # All start with 40 energy
        self.age = np.zeros(pop, dtype=np.int32)              # All age 0
        self.generation = np.zeros(pop, dtype=np.int32)       # All generation 0 (founders)
        self.ids = np.arange(pop, dtype=np.int64)             # IDs: 0, 1, 2, ..., 79
        self.parent_ids = np.full(pop, -1, dtype=np.int64)    # No parents (founders)
        self.genomes = self.rng.normal(0, 0.5, (pop, c.genome_size))  # Random 14-gene genomes
        self.transfer_count = np.zeros(pop, dtype=np.int32)   # No transfers yet
        self.next_id = pop  # Next available ID = 80

        # Running total of all horizontal transfers that have occurred
        self.total_transfers = 0

        # Stats history (one dict per timestep)
        self.stats_history = []

    # ══════════════════════════════════════════════════════════════════════════
    # ORGANISM ARRAY HELPERS (NEW ARCHITECTURE)
    # ══════════════════════════════════════════════════════════════════════════
    # These three methods replace the manual array manipulation that was
    # scattered throughout Phase 1's code. They use the _org_arrays registry
    # to automatically handle ALL organism properties at once.

    @property
    def pop(self):
        """
        Convenience property: current population count.
        Just returns how many rows we have (all parallel arrays have the same length).
        """
        return len(self.rows)

    def _filter_organisms(self, mask):
        """
        Keep only organisms where mask is True, remove the rest.
        
        This replaces the Phase 1 pattern of:
            self.rows = self.rows[alive]
            self.cols = self.cols[alive]
            self.energy = self.energy[alive]
            ... (repeated for EVERY array)
        
        Now it's ONE call that handles all registered arrays automatically.
        
        Args:
            mask: Boolean array of length pop. True = keep, False = remove.
        """
        for name, _, _ in self._org_arrays:
            # For each registered array, filter it by the mask
            setattr(self, name, getattr(self, name)[mask])

    def _append_organisms(self, new_data: dict):
        """
        Add new organisms (babies) to the population.
        
        new_data is a dictionary mapping array names to their values for the
        new organisms. For example:
            {"rows": [50, 60], "cols": [30, 40], "energy": [40, 40], ...}
        
        For any registered array NOT in new_data, the default value from the
        registry is used. If there's no default (None) and it's not in new_data,
        an error is raised — this catches bugs where we forget to provide
        required data for new organisms.
        
        This replaces Phase 1's long chain of np.concatenate calls.
        
        Args:
            new_data: Dict mapping attribute names to arrays of values for new organisms.
        """
        for name, dtype, default in self._org_arrays:
            existing = getattr(self, name)  # The current array for all living organisms
            
            if name in new_data:
                # Explicit values were provided — use them
                addition = np.asarray(new_data[name], dtype=existing.dtype)
            elif default is not None:
                # No explicit values, but we have a default — create an array of defaults.
                # Figure out how many new organisms there are from any provided array.
                n = len(next(iter(new_data.values())))
                addition = np.full(n, default, dtype=existing.dtype)
                
                # Special handling for 2D arrays (like genomes):
                # np.full creates 1D, but we need matching shape
                if existing.ndim > 1:
                    addition = np.zeros((n, *existing.shape[1:]), dtype=existing.dtype) + default
            else:
                # No values provided AND no default — this is a required field!
                raise ValueError(f"Must provide '{name}' in new_data (no default)")
            
            # Concatenate existing organisms + new organisms
            setattr(self, name, np.concatenate([existing, addition]))

    def _get_dead_data(self, dead_mask):
        """
        Extract data for dead organisms (before they're removed).
        Returns a dict with all registered arrays filtered to just the dead ones.
        
        This is used by _kill_and_decompose to grab the dead organisms' genomes,
        positions, and energy before removing them from the population.
        
        Args:
            dead_mask: Boolean array. True = dead organism.
        
        Returns:
            Dict mapping attribute names to arrays of dead organism data.
        """
        return {name: getattr(self, name)[dead_mask] for name, _, _ in self._org_arrays}

    # ══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT UPDATE
    # ══════════════════════════════════════════════════════════════════════════

    def _update_environment(self):
        """
        Update all environment grids for one timestep.
        Same as Phase 1 (toxic diffusion, decay, organism waste, nutrients,
        decomposition) PLUS the new fragment pool decay and diffusion.
        """
        c = self.cfg

        # ── Toxic Diffusion ──────────────────────────────────────────────────
        # Each cell's toxicity moves 6% toward the average of its 4 neighbors.
        # This gradually smooths out toxic hotspots.
        k = c.toxic_diffusion_rate
        p = np.pad(self.toxic, 1, mode='edge')  # Pad edges so border cells have "neighbors"
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0  # Neighbor average
        self.toxic += k * (nb - self.toxic)  # Blend toward neighbors

        # ── Toxic Decay (zone-modulated) ─────────────────────────────────────
        # Toxicity fades each step. Absorbing zones (low zone_map) decay faster.
        self.toxic *= (1.0 - c.toxic_decay_rate / np.maximum(0.3, self.zone_map))

        # ── Organisms Produce Toxic Waste ────────────────────────────────────
        # Each living organism adds toxicity at its cell, amplified by the zone.
        if self.pop > 0:
            np.add.at(self.toxic, (self.rows, self.cols),
                      c.toxic_production_rate * self.zone_map[self.rows, self.cols])

        # ── Nutrient Regeneration + Decomposition Cycle ──────────────────────
        self.nutrients += c.nutrient_base_rate  # Tiny natural nutrient trickle
        xfer = self.decomposition * 0.015       # 1.5% of decomp → nutrients
        self.nutrients += xfer
        self.decomposition -= xfer
        self.decomposition *= 0.998             # Remaining decomp decays slowly

        # ══════════════════════════════════════════════════════════════════════
        # FRAGMENT POOL DECAY & DIFFUSION (NEW IN PHASE 2)
        # ══════════════════════════════════════════════════════════════════════
        # The genetic fragments left by dead organisms slowly degrade and
        # spread to neighboring cells, just like toxicity does.

        # ── Fragment Decay ───────────────────────────────────────────────────
        # Fragment "weight" (how much DNA is at each cell) decays by 0.5% per step.
        # This means old fragments gradually fade away — DNA degrades over time.
        self.fragment_weight *= (1.0 - c.decomp_fragment_decay)

        # ── Fragment Diffusion (every 10 steps for performance) ──────────────
        # Every 10 steps, spread fragments to neighboring cells (like DNA
        # diffusing through water/soil). This is computationally expensive
        # because fragment_pool is 3D (128x128x14), so we don't do it every step.
        #
        # We also skip if there are barely any fragments (max weight < 0.001).
        if self.timestep % 10 == 0 and self.fragment_weight.max() > 0.001:
            # Diffusion rate is multiplied by 10 to compensate for only running
            # every 10 steps (same total diffusion over time).
            k2 = c.decomp_fragment_diffusion * 10

            # Diffuse the WEIGHTS (how much fragment is at each cell)
            # Same pad-and-average technique as toxic diffusion
            pw = np.pad(self.fragment_weight, 1, mode='edge')
            wn = (pw[:-2, 1:-1] + pw[2:, 1:-1] + pw[1:-1, :-2] + pw[1:-1, 2:]) / 4.0
            self.fragment_weight += k2 * (wn - self.fragment_weight)
            self.fragment_weight = np.maximum(self.fragment_weight, 0.0)  # Can't go negative

            # Diffuse the GENOMES themselves (the actual DNA content)
            # Pad the 3D array along spatial axes (not the genome axis)
            pp = np.pad(self.fragment_pool, ((1, 1), (1, 1), (0, 0)), mode='edge')
            gn = (pp[:-2, 1:-1, :] + pp[2:, 1:-1, :] + pp[1:-1, :-2, :] + pp[1:-1, 2:, :]) / 4.0
            self.fragment_pool += k2 * (gn - self.fragment_pool)

        # ── Clamp Everything ─────────────────────────────────────────────────
        np.clip(self.toxic, 0, 5.0, out=self.toxic)
        np.clip(self.nutrients, 0, c.nutrient_max, out=self.nutrients)
        np.clip(self.decomposition, 0, 10.0, out=self.decomposition)

    def _update_density(self):
        """
        Count organisms per cell. Used for crowding/competition calculations.
        """
        self.density[:] = 0
        if self.pop > 0:
            np.add.at(self.density, (self.rows, self.cols), 1)

    # ══════════════════════════════════════════════════════════════════════════
    # SENSING
    # ══════════════════════════════════════════════════════════════════════════

    def _sense_local(self):
        """
        Each organism reads 8 sensor values from its local environment:
          [0] local_light     — Brightness at this cell
          [1] local_toxic     — Toxicity at this cell
          [2] local_nutrients — Nutrients at this cell
          [3] local_density   — How many organisms are on this cell
          [4] light_grad_y    — Vertical light gradient (positive = more light above)
          [5] light_grad_x    — Horizontal light gradient (positive = more light right)
          [6] toxic_grad_y    — Vertical toxicity gradient (positive = more toxic above)
          [7] toxic_grad_x    — Horizontal toxicity gradient (positive = more toxic right)
        
        Returns: (num_organisms, 8) matrix of sensor readings.
        """
        rows, cols = self.rows, self.cols
        N = self.cfg.grid_size

        # Neighbor positions (clamped to grid edges)
        ru = np.clip(rows - 1, 0, N - 1)   # Row above
        rd = np.clip(rows + 1, 0, N - 1)   # Row below
        cl = np.clip(cols - 1, 0, N - 1)   # Column left
        cr = np.clip(cols + 1, 0, N - 1)   # Column right

        return np.column_stack([
            self.light[rows, cols],                        # [0] Light here
            self.toxic[rows, cols],                        # [1] Toxicity here
            self.nutrients[rows, cols],                    # [2] Nutrients here
            self.density[rows, cols].astype(np.float64),   # [3] Crowding here
            self.light[ru, cols] - self.light[rd, cols],   # [4] Light gradient (up vs down)
            self.light[rows, cr] - self.light[rows, cl],   # [5] Light gradient (right vs left)
            self.toxic[ru, cols] - self.toxic[rd, cols],   # [6] Toxic gradient (up vs down)
            self.toxic[rows, cr] - self.toxic[rows, cl],   # [7] Toxic gradient (right vs left)
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # PHOTOSYNTHESIS, TOXIC DAMAGE, MOVEMENT
    # ══════════════════════════════════════════════════════════════════════════
    # These work identically to Phase 1. See phase1_v3_commented.py for
    # detailed explanations of each factor. Brief summary here.

    def _photosynthesize(self, readings):
        """
        Calculate energy gained from photosynthesis for each organism.
        
        Uses genome genes 8-11 (the 4 photosynthesis genes) and multiplies
        five factors together:
          1. Base efficiency (gene 8) — natural photosynthesis ability
          2. Toxic penalty (gene 9) — toxicity tolerance for photosynthesis
          3. Light sensitivity (gene 10) — ability to use dim light
          4. Storage capacity (gene 11) — how much energy can be absorbed
          5. Density competition — sharing light with neighbors
        
        Returns: Array of energy gains, one per organism.
        """
        c = self.cfg
        # Extract the 4 photosynthesis genes (indices 8-11 in the 14-gene genome)
        ph = self.genomes[:, c.movement_params_size:c.movement_params_size + c.photo_params_size]

        # Shorthand for sensor readings we need
        ll = readings[:, 0]   # local light
        lt = readings[:, 1]   # local toxicity
        ld = readings[:, 3]   # local density (crowding)

        # Factor 1: Base efficiency — tanh maps gene to range (0.7x to 1.3x base)
        eff = c.photosynthesis_base * (1.0 + 0.3 * np.tanh(ph[:, 0]))

        # Factor 2: Toxic penalty — higher tolerance gene = less affected by toxicity
        tol = np.maximum(0.1, 1.0 + 0.5 * np.tanh(ph[:, 1]))
        tp = np.maximum(0.0, 1.0 - lt * c.toxic_photo_penalty / tol)

        # Factor 3: Light sensitivity — lower exponent = better at using dim light
        le = np.maximum(0.3, 1.0 - 0.3 * np.tanh(ph[:, 2]))
        lm = np.power(np.maximum(ll, 0.01), le)

        # Factor 4: Storage — sigmoid maps gene to range (0.5 to 1.0)
        st = 0.5 + 0.5 / (1.0 + np.exp(-ph[:, 3]))

        # Factor 5: Density competition — more crowded = less light per organism
        sh = 1.0 / np.maximum(1.0, ld * 0.5)

        # All five factors multiplied together
        return eff * tp * lm * st * sh

    def _apply_toxic_damage(self, readings):
        """
        Deal direct energy damage to organisms in toxic areas.
        Two damage tiers that stack:
          - Above 0.8 (medium): moderate energy drain
          - Above 1.5 (high): severe additional drain
        This is what creates natural carrying capacity.
        """
        c = self.cfg
        lt = readings[:, 1]  # Local toxicity for each organism
        dmg = np.zeros(self.pop)

        # Medium threshold damage (above 0.8)
        m = lt > c.toxic_threshold_medium
        if m.any():
            dmg[m] += (lt[m] - c.toxic_threshold_medium) * c.toxic_damage_medium

        # High threshold damage (above 1.5, stacks with medium)
        h = lt > c.toxic_threshold_high
        if h.any():
            dmg[h] += (lt[h] - c.toxic_threshold_high) * c.toxic_damage_high

        self.energy -= dmg

    def _decide_movement(self, readings):
        """
        Each organism scores 5 options (stay, up, down, right, left) using its
        8 movement genes and the 8 sensor readings, then picks the highest score.
        
        Key gene roles:
          Gene 1: Crowd avoidance weight
          Gene 2: Nutrient attraction weight
          Gene 4: Exploration noise level
          Gene 5: Base "stay" preference
          Gene 6: Light-seeking weight
          Gene 7: Toxic-avoidance weight
        
        Returns: Array of actions (0=stay, 1=up, 2=down, 3=right, 4=left)
        """
        c = self.cfg
        n = self.pop
        if n == 0:
            return np.array([], dtype=np.int32)

        mp = self.genomes[:, :c.movement_params_size]  # Movement genes (0-7)
        sc = np.zeros((n, 5))  # Scores for 5 possible actions

        # Stay score: base preference + nutrient bonus - crowding penalty
        sc[:, 0] = mp[:, 5] + mp[:, 2] * readings[:, 2] - mp[:, 1] * np.minimum(readings[:, 3], 10) * 0.1

        # Directional scores: follow light gradients, avoid toxic gradients
        sc[:, 1] = mp[:, 6] * readings[:, 4] - mp[:, 7] * readings[:, 6]   # Up
        sc[:, 2] = -mp[:, 6] * readings[:, 4] + mp[:, 7] * readings[:, 6]  # Down
        sc[:, 3] = mp[:, 6] * readings[:, 5] - mp[:, 7] * readings[:, 7]   # Right
        sc[:, 4] = -mp[:, 6] * readings[:, 5] + mp[:, 7] * readings[:, 7]  # Left

        # Add exploration noise (gene 4 controls noise amplitude)
        sc += mp[:, 4:5] * self.rng.normal(0, 0.3, (n, 5))

        return np.argmax(sc, axis=1)  # Pick highest-scoring action

    def _execute_movement(self, actions):
        """
        Move organisms based on their chosen actions.
        Clamp to grid boundaries. Deduct movement energy cost.
        """
        N = self.cfg.grid_size
        m = actions == 1; self.rows[m] = np.maximum(0, self.rows[m] - 1)          # Up
        m = actions == 2; self.rows[m] = np.minimum(N - 1, self.rows[m] + 1)      # Down
        m = actions == 3; self.cols[m] = np.minimum(N - 1, self.cols[m] + 1)      # Right
        m = actions == 4; self.cols[m] = np.maximum(0, self.cols[m] - 1)           # Left
        self.energy[actions > 0] -= self.cfg.energy_movement_cost  # Moving costs 0.2 energy

    # ══════════════════════════════════════════════════════════════════════════
    # HORIZONTAL GENE TRANSFER (THE BIG NEW MECHANIC)
    # ══════════════════════════════════════════════════════════════════════════

    def _horizontal_transfer(self):
        """
        Living organisms attempt to absorb genetic fragments from dead organisms.
        
        This is the signature mechanic of Phase 2. Here's the step-by-step:
        
        1. RECEPTIVITY CHECK: Each organism has a "receptivity" gene (gene 12).
           Higher receptivity = more likely to attempt absorption.
           The gene is passed through a sigmoid (0 to 1 probability).
        
        2. AVAILABILITY CHECK: There must be enough fragment material at the
           organism's cell (fragment_weight > 0.1). No fragments = nothing to absorb.
        
        3. RANDOM CHANCE: Even if receptive and fragments are available,
           there's still a random roll. receptivity IS the probability.
        
        4. SELECTIVITY FILTER: The organism has a "selectivity" gene (gene 13).
           High selectivity = only absorbs fragments that are genetically SIMILAR
           to itself. Low selectivity = will absorb anything.
           This prevents organisms from absorbing wildly different DNA that
           might be harmful. We measure similarity as the average distance
           between the organism's genome and the fragment genome.
        
        5. BLENDING: If all checks pass, the organism's genome becomes:
              85% its own genome + 15% the absorbed fragment
           This is a gentle blend — enough to introduce new traits but not
           so much that it overwrites the organism's identity.
        
        6. TRACKING: The organism's transfer_count is incremented, and the
           global total_transfers counter goes up. This lets us monitor
           how much horizontal transfer is happening.
        """
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        # ── Extract the 2 transfer genes for each organism ───────────────────
        # These are the LAST 2 genes in the genome (indices 12 and 13).
        tp = self.genomes[:, c.movement_params_size + c.photo_params_size:]

        # Gene 12: Receptivity — sigmoid converts to probability (0 to 1)
        # High gene value → probability near 1.0 (very receptive to foreign DNA)
        # Low gene value → probability near 0.0 (resistant to absorbing DNA)
        receptivity = 1.0 / (1.0 + np.exp(-tp[:, 0]))

        # Gene 13: Selectivity — absolute value (always positive)
        # High selectivity = picky (only absorbs similar DNA)
        # Low selectivity = permissive (absorbs diverse DNA)
        selectivity = np.abs(tp[:, 1])

        # ── Check which organisms attempt absorption ─────────────────────────
        # Two conditions must be met:
        #   1. There are enough fragments at this cell (weight > 0.1)
        #   2. A random roll beats the organism's receptivity probability
        local_fw = self.fragment_weight[self.rows, self.cols]  # Fragment weight at each organism
        attempts = (local_fw > 0.1) & (self.rng.random(n) < receptivity)
        aidx = np.where(attempts)[0]  # Indices of organisms attempting transfer
        if len(aidx) == 0:
            return  # Nobody is attempting

        # ── Selectivity filter: is the fragment similar enough? ──────────────
        # Grab the fragment genome at each attempting organism's cell
        local_frags = self.fragment_pool[self.rows[aidx], self.cols[aidx]]

        # Calculate "genetic distance" = root-mean-square difference between
        # the organism's genome and the fragment genome.
        # Small distance = very similar DNA. Large distance = very different.
        dists = np.sqrt(np.mean((self.genomes[aidx] - local_frags) ** 2, axis=1))

        # The acceptance threshold depends on selectivity:
        #   Low selectivity (0) → threshold = 2.0 / 1.0 = 2.0 (accepts distant DNA)
        #   High selectivity (3) → threshold = 2.0 / 4.0 = 0.5 (only accepts very similar DNA)
        thresh = 2.0 / (1.0 + selectivity[aidx])

        # Keep only organisms whose fragment distance is below their threshold
        tidx = aidx[dists < thresh]  # Final list of organisms that will absorb
        if len(tidx) == 0:
            return  # All attempts were rejected by selectivity filter

        # ── Perform the genome blending ──────────────────────────────────────
        # new_genome = 85% own genome + 15% fragment genome
        blend = c.transfer_blend_rate  # 0.15
        frags = self.fragment_pool[self.rows[tidx], self.cols[tidx]]
        self.genomes[tidx] = (1.0 - blend) * self.genomes[tidx] + blend * frags

        # ── Track the transfers ──────────────────────────────────────────────
        self.transfer_count[tidx] += 1       # Per-organism counter
        self.total_transfers += len(tidx)    # Global running total

    # ══════════════════════════════════════════════════════════════════════════
    # REPRODUCTION (now uses _append_organisms)
    # ══════════════════════════════════════════════════════════════════════════

    def _reproduce(self):
        """
        Organisms with enough energy (>= 80) and age (>= 8) reproduce asexually.
        
        This is much cleaner than Phase 1 thanks to _append_organisms:
        we just provide a dict of the required values, and the registry
        handles the rest. Arrays with defaults (age, transfer_count) are
        automatically filled with 0 for the babies.
        """
        c = self.cfg

        # Find organisms eligible to reproduce
        can = (self.energy >= c.energy_reproduction_threshold) & (self.age >= c.min_reproduction_age)
        pidx = np.where(can)[0]   # Indices of parents
        nb = len(pidx)            # Number of births
        if nb == 0:
            return

        # Parents pay the reproduction cost
        self.energy[pidx] -= c.energy_reproduction_cost

        # Generate unique IDs for the babies
        child_ids = np.arange(self.next_id, self.next_id + nb, dtype=np.int64)
        self.next_id += nb

        # Use _append_organisms to add all babies at once.
        # Only need to specify arrays without defaults (or with non-default values).
        # "age" and "transfer_count" default to 0 automatically!
        self._append_organisms({
            # Baby positions: parent position + random offset, clamped to grid
            "rows": np.clip(
                self.rows[pidx] + self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, nb),
                0, c.grid_size - 1),
            "cols": np.clip(
                self.cols[pidx] + self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, nb),
                0, c.grid_size - 1),
            "energy": np.full(nb, c.energy_initial),                  # 40 energy each
            "generation": self.generation[pidx] + 1,                  # Parent generation + 1
            "ids": child_ids,                                          # Unique IDs
            "parent_ids": self.ids[:self.pop][pidx],                  # Record parent's ID
            # Baby genome = parent genome + small random mutations
            "genomes": self.genomes[pidx] + self.rng.normal(0, c.mutation_rate, (nb, c.genome_size)),
        })
        # Note: age defaults to 0, transfer_count defaults to 0 — no need to specify!

    # ══════════════════════════════════════════════════════════════════════════
    # DEATH & DECOMPOSITION (now uses _filter_organisms and _get_dead_data)
    # ══════════════════════════════════════════════════════════════════════════

    def _kill_and_decompose(self):
        """
        Remove dead organisms and process their remains:
          1. Deposit energy as decomposition matter (nutrients for survivors)
          2. Deposit genome as fragment pool (DNA for horizontal transfer)
          3. Remove dead organisms from all arrays
        
        The fragment deposition is the big new piece. When multiple organisms
        die on the same cell, their genomes are averaged (weighted by how
        many died there) and blended with any existing fragments at that cell.
        """
        c = self.cfg

        # Determine who's alive: has energy AND hasn't exceeded max age
        alive = (self.energy > 0) & (self.age < c.max_age)
        dead = ~alive

        if dead.any():
            # ── Extract dead organism data before removing them ──────────────
            dd = self._get_dead_data(dead)  # Dict of all arrays for dead organisms
            dr = dd["rows"]                  # Dead organisms' row positions
            dc = dd["cols"]                  # Dead organisms' column positions
            de = np.maximum(0, dd["energy"]) # Dead organisms' remaining energy (min 0)
            dg = dd["genomes"]               # Dead organisms' genomes

            # ── Deposit decomposition matter (same as Phase 1) ───────────────
            # 40% of remaining energy + 0.5 base → decomposition grid
            np.add.at(self.decomposition, (dr, dc), de * c.nutrient_from_decomp + 0.5)

            # ══════════════════════════════════════════════════════════════════
            # DEPOSIT GENOME FRAGMENTS (NEW IN PHASE 2)
            # ══════════════════════════════════════════════════════════════════
            # Multiple organisms might die on the same cell. We need to:
            #   1. Group dead organisms by cell
            #   2. Average their genomes within each cell
            #   3. Blend that average with any existing fragments at the cell
            #
            # This uses a weighted averaging scheme:
            #   If a cell already has fragment_weight = 3 (from previous deaths)
            #   and 2 new organisms die there, the new weight = 5.
            #   The new fragment genome = (3/5 * old_genome) + (2/5 * new_avg_genome)
            #   This gives more influence to whichever side has more "mass".

            # ── Step 1: Group dead organisms by cell ─────────────────────────
            # Convert (row, col) to a single cell ID for grouping.
            # cell_id = row * grid_size + col  (unique per cell)
            cell_ids = dr * c.grid_size + dc
            unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
            # unique_cells: list of distinct cells where at least one organism died
            # inverse: maps each dead organism to its position in unique_cells

            nc = len(unique_cells)  # Number of distinct cells with deaths

            # ── Step 2: Sum genomes and count organisms per cell ─────────────
            genome_sums = np.zeros((nc, c.genome_size))  # Sum of all genomes per cell
            counts = np.zeros(nc)                         # Count of deaths per cell
            np.add.at(genome_sums, inverse, dg)           # Add each dead genome to its cell's sum
            np.add.at(counts, inverse, 1.0)               # Count deaths per cell

            # Convert cell IDs back to (row, col) coordinates
            ur = unique_cells // c.grid_size    # Row of each unique cell
            uc = unique_cells % c.grid_size     # Column of each unique cell

            # ── Step 3: Blend new fragments with existing ones ───────────────
            w0 = self.fragment_weight[ur, uc]   # Existing fragment weight at each cell
            wt = w0 + counts                    # New total weight = old + new deaths

            avg_new = genome_sums / counts[:, None]  # Average genome of newly dead organisms
            blend = counts / wt                      # How much weight the new deaths have
            # blend is high when many organisms die on a cell with few existing fragments,
            # and low when few die on a cell already heavy with fragments.

            # Weighted average: old genome * (old_weight/total) + new genome * (new_weight/total)
            self.fragment_pool[ur, uc] = (
                self.fragment_pool[ur, uc] * (1.0 - blend[:, None]) + avg_new * blend[:, None])

            # Update the fragment weights
            self.fragment_weight[ur, uc] = wt

        # ── Remove dead organisms from the population ────────────────────────
        # Uses the registry-based filter — one call handles all arrays.
        self._filter_organisms(alive)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN SIMULATION LOOP (one timestep)
    # ══════════════════════════════════════════════════════════════════════════

    def update(self):
        """
        Execute one timestep of the simulation.
        
        Order of operations:
          1. Count density (organisms per cell)
          2. Sense (each organism reads its local environment)
          3. Photosynthesis (gain energy from sunlight)
          4. Toxic damage (lose energy in polluted areas)
          5. Movement (decide direction and move)
          6. Aging + maintenance cost
          7. Horizontal transfer (every 5 steps — absorb dead organisms' DNA)
          8. Reproduction (if enough energy and old enough)
          9. Death + decomposition (remove dead, deposit remains + DNA fragments)
          10. Environment update (diffusion, decay, nutrients, fragment pool)
        
        Step 7 (horizontal transfer) is the key addition from Phase 1.
        It only runs every 5 steps to save computation.
        """
        # If everyone is dead, just update environment (watch toxicity fade)
        if self.pop == 0:
            self._update_environment()
            self.timestep += 1
            self._record_stats()
            return

        self._update_density()                                              # 1. Count crowding
        readings = self._sense_local()                                      # 2. Sense environment
        self.energy = np.minimum(                                           # 3. Photosynthesis
            self.energy + self._photosynthesize(readings), self.cfg.energy_max)
        self._apply_toxic_damage(readings)                                  # 4. Toxic damage
        self._execute_movement(self._decide_movement(readings))             # 5. Move

        self.age += 1                                                       # 6a. Age +1
        self.energy -= self.cfg.energy_maintenance_cost                     # 6b. Maintenance cost

        # 7. Horizontal transfer — only every 5 steps (saves computation,
        #    and makes transfer a periodic event rather than constant pressure)
        if self.timestep % self.cfg.transfer_check_interval == 0:
            self._horizontal_transfer()

        self._reproduce()                                                   # 8. Reproduce
        self._kill_and_decompose()                                          # 9. Die + deposit DNA
        self._update_environment()                                          # 10. Environment update

        self.timestep += 1
        self._record_stats()

    # ══════════════════════════════════════════════════════════════════════════
    # STATS & SNAPSHOTS
    # ══════════════════════════════════════════════════════════════════════════

    def _record_stats(self):
        """
        Record key metrics for this timestep.
        Same as Phase 1, plus new stats for horizontal gene transfer:
          - frag_mean/frag_max: How much genetic material is in the fragment pool
          - orgs_with_transfers: How many living organisms have absorbed DNA at least once
          - avg_transfer_count: Average number of absorptions per organism
          - total_transfers: Cumulative transfers across all time
        """
        p = self.pop
        c = self.cfg

        if p > 0:
            lt = self.toxic[self.rows, self.cols]  # Toxicity at each organism's cell

            # Count organisms in each toxicity zone
            in_low = int(np.sum(lt < c.toxic_threshold_low))                    # Safe (< 0.3)
            in_med = int(np.sum((lt >= c.toxic_threshold_low) & (lt < c.toxic_threshold_medium)))  # Stressed
            in_high = int(np.sum(lt >= c.toxic_threshold_medium))               # Danger (>= 0.8)

            # Transfer stats
            xferd = int(np.sum(self.transfer_count > 0))  # How many organisms have absorbed DNA
            avg_tc = float(self.transfer_count.mean())     # Average absorptions per organism
        else:
            in_low = in_med = in_high = xferd = 0
            avg_tc = 0.0

        self.stats_history.append({
            "t": self.timestep,
            "pop": p,
            "avg_energy": float(self.energy.mean()) if p > 0 else 0,
            "max_gen": int(self.generation.max()) if p > 0 else 0,
            "toxic_mean": float(self.toxic.mean()),
            "toxic_max": float(self.toxic.max()),
            "decomp_mean": float(self.decomposition.mean()),
            "frag_mean": float(self.fragment_weight.mean()),     # Avg fragment density (NEW)
            "frag_max": float(self.fragment_weight.max()),       # Peak fragment density (NEW)
            "in_low_toxic": in_low,
            "in_med_toxic": in_med,
            "in_high_toxic": in_high,
            "orgs_with_transfers": xferd,                        # Organisms that absorbed DNA (NEW)
            "avg_transfer_count": round(avg_tc, 2),              # Avg absorptions per organism (NEW)
            "total_transfers": self.total_transfers,             # Cumulative transfers ever (NEW)
        })

    def save_snapshot(self, output_dir):
        """
        Save a JSON snapshot of current organisms + stats.
        Samples up to 500 organisms if population is larger (for file size).
        Now includes transfer_count per organism.
        """
        os.makedirs(output_dir, exist_ok=True)
        p = self.pop

        # Sample 500 random organisms if population is large
        idx = self.rng.choice(p, min(p, 500), replace=False) if p > 500 else np.arange(p)

        # Build organism list with all relevant fields
        orgs = [{
            "id": int(self.ids[i]),
            "row": int(self.rows[i]),
            "col": int(self.cols[i]),
            "energy": round(float(self.energy[i]), 2),
            "age": int(self.age[i]),
            "generation": int(self.generation[i]),
            "genome": self.genomes[i].tolist(),
            "transfer_count": int(self.transfer_count[i])   # NEW: how many times absorbed DNA
        } for i in idx]

        s = self.stats_history[-1] if self.stats_history else {}
        with open(os.path.join(output_dir, f"snapshot_{self.timestep:06d}.json"), 'w') as f:
            json.dump({"timestep": self.timestep, "population": p, "organisms": orgs, "stats": s}, f)

    def save_env(self, output_dir):
        """
        Save environment grids as compressed numpy file.
        Now includes fragment_weight grid (showing where DNA fragments are concentrated).
        """
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(output_dir, f"env_{self.timestep:06d}.npz"),
            toxic=self.toxic.astype(np.float32),
            decomposition=self.decomposition.astype(np.float32),
            density=self.density.astype(np.int16),
            fragment_weight=self.fragment_weight.astype(np.float32)  # NEW: DNA fragment heatmap
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(cfg=None):
    """
    Run the full simulation from start to finish.
    
    Same structure as Phase 1, but the progress output now includes
    horizontal transfer stats:
      - frag: mean/max fragment pool weight
      - xfer: number of organisms that have absorbed DNA
      - cum: cumulative total transfers across all time
    """
    cfg = cfg or Config()
    world = World(cfg)

    # Print header
    print(f"The Shimmering Field — Phase 2 Step 1 (Refactored)")
    print(f"Grid: {cfg.grid_size}x{cfg.grid_size}  |  Pop: {cfg.initial_population}  |  Genome: {cfg.genome_size}")
    print(f"{'─' * 95}")

    start = time.time()

    # ── Main loop: 10,000 timesteps ──────────────────────────────────────────
    for t in range(cfg.total_timesteps):
        world.update()  # Execute one timestep

        # Print progress every 100 steps
        if world.timestep % cfg.snapshot_interval == 0:
            world.save_snapshot(cfg.output_dir)
            s = world.stats_history[-1]
            el = time.time() - start

            # Status line includes new transfer metrics
            print(f"  t={s['t']:5d}  |  pop={s['pop']:5d}  |  e={s['avg_energy']:5.1f}  |  "
                  f"gen={s['max_gen']:4d}  |  tox={s['toxic_mean']:.3f}  |  "
                  f"frag={s['frag_mean']:.2f}/{s['frag_max']:.1f}  |  "              # Fragment stats
                  f"xfer={s['orgs_with_transfers']:4d} cum={s['total_transfers']:6d}  |  {el:.1f}s")  # Transfer stats

        # Save full environment grids every 500 steps
        if world.timestep % 500 == 0:
            world.save_env(cfg.output_dir)

        # Handle extinction
        if world.pop == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            break

    # ── Wrap up ──────────────────────────────────────────────────────────────
    el = time.time() - start
    print(f"{'─' * 95}")
    print(f"Done in {el:.1f}s  |  Pop: {world.pop}  |  Transfers: {world.total_transfers}")

    # Save summary JSON with full config + stats history
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
# Run with: python phase2_step1.py
if __name__ == "__main__":
    run_simulation()
