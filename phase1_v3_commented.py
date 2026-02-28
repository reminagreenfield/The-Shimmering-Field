"""
The Shimmering Field — Phase 1 v3: Tuned Ecology
==================================================

This is an artificial life simulation. Imagine a 256x256 grid (like a big
chessboard) where tiny digital organisms live, eat sunlight, move around,
reproduce, and die. The twist: every organism produces toxic waste as a
byproduct of living, and that waste builds up in the environment. If too
many organisms crowd together, the toxicity kills them off — creating a
natural population balance without any hard-coded population cap.

Think of it like a virtual petri dish where evolution happens in real time.

Key changes from v2:
  - Toxic accumulation now damages organisms directly (energy drain + death)
  - Three toxic thresholds per design doc: low/medium/high
  - Hard population cap removed — carrying capacity is ecological
  - Zone-based chemical landscape: some zones absorb toxic, others amplify
  - Faster generational turnover (shorter lifespan, lower reproduction threshold)
  - Nutrient competition is local and finite — photosynthesis depletes local light capture
"""

# ─── IMPORTS ───────────────────────────────────────────────────────────────────
# numpy: A math library that lets us do fast operations on big arrays of numbers.
#         Instead of looping over 10,000 organisms one by one, we can process
#         them all at once using array math. This is critical for performance.
# json:   For saving simulation data as human-readable .json files.
# os:     For creating folders and working with file paths.
# time:   For measuring how long the simulation takes to run.
import numpy as np
import json
import os
import time


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# This class holds every tunable number in the simulation. By changing these
# values, you can make organisms live longer, toxicity spread faster, the
# world bigger, etc. Think of it as the "settings menu" for the simulation.

class Config:
    # ── Grid Settings ──────────────────────────────────────────────────────────
    # The world is a square grid. 256x256 = 65,536 individual cells.
    # Each cell can hold environment values (light, toxicity, nutrients)
    # and any number of organisms can stand on the same cell.
    grid_size = 256
    
    # ── Light Gradient ─────────────────────────────────────────────────────────
    # Light comes from the "top" of the grid and fades toward the "bottom".
    # Row 0 (top) gets full light (1.0), row 255 (bottom) gets almost none (0.05).
    # This creates a natural incentive for organisms to move upward,
    # but also means everyone crowds at the top — creating toxic buildup there.
    light_max = 1.0       # Brightness at the top row
    light_min = 0.05      # Brightness at the bottom row (dim, but not zero)
    
    # ── Zone-Based Chemical Landscape ──────────────────────────────────────────
    # The grid is divided into a checkerboard of 8x8 = 64 zones.
    # Each zone has a random "multiplier" that affects how toxicity behaves there:
    #   - Multiplier < 1.0: This zone ABSORBS toxicity (a safe haven / "refugia")
    #   - Multiplier > 1.0: This zone AMPLIFIES toxicity (a danger zone)
    # This creates interesting spatial patterns — organisms might evolve to
    # seek out the safe zones and avoid the dangerous ones.
    zone_count = 8        # 8 zones per axis = 64 total zones on the grid
    
    # ── Toxicity Settings ──────────────────────────────────────────────────────
    # Toxicity is THE central mechanic of this simulation. Here's how it works:
    #   1. Every living organism produces a small amount of toxic waste each step
    #   2. Toxicity slowly spreads to neighboring cells (diffusion)
    #   3. Toxicity slowly fades over time (decay)
    #   4. If toxicity gets too high, organisms take damage and die
    #
    # This creates a natural feedback loop:
    #   More organisms → more waste → higher toxicity → organisms die →
    #   fewer organisms → less waste → toxicity drops → organisms thrive again
    #
    # This is what creates a natural "carrying capacity" without any hard limit.
    
    toxic_decay_rate = 0.01           # How fast toxicity fades each step (1% base rate).
                                      # Slow decay means toxic lingers in the environment.
    
    toxic_diffusion_rate = 0.06       # How fast toxicity spreads to neighboring cells.
                                      # Higher = toxicity evens out across the grid faster.
    
    toxic_production_rate = 0.015     # How much toxic waste ONE organism produces per step.
                                      # Small per organism, but adds up fast with hundreds.
    
    # ── Three Toxic Thresholds ─────────────────────────────────────────────────
    # These define three "zones" of toxicity severity:
    #
    #   [0.0 ──── 0.3) = LOW:    Safe. Life is normal. No penalties.
    #   [0.3 ──── 0.8) = MEDIUM: Stressful. Photosynthesis becomes less effective.
    #   [0.8 ──── 1.5) = HIGH:   Dangerous. Organisms take direct energy damage.
    #   [1.5+)         = CRISIS: Organisms die rapidly. Ecosystem collapse.
    
    toxic_threshold_low = 0.3         # Below this: everything is fine
    toxic_threshold_medium = 0.8      # Above this: organisms start taking damage
    toxic_threshold_high = 1.5        # Above this: organisms die fast, crisis mode
    
    # ── Toxic Damage Values ────────────────────────────────────────────────────
    toxic_damage_medium = 1.5         # Energy drained per step per unit of excess toxicity
                                      # above the medium threshold. Like a slow poison.
    
    toxic_damage_high = 5.0           # Energy drained per step per unit of excess toxicity
                                      # above the high threshold. Like rapid poisoning.
    
    toxic_photo_penalty = 1.0         # How much toxicity reduces photosynthesis efficiency.
                                      # 1.0 means each unit of toxicity roughly halves
                                      # the energy an organism can harvest from light.
    
    # ── Nutrient Settings ──────────────────────────────────────────────────────
    # Nutrients are a background resource that slowly regenerates.
    # When organisms die, their bodies decompose and release nutrients back
    # into the soil. This creates a nutrient cycle:
    #   organisms die → decomposition → nutrients released → helps survivors
    
    nutrient_base_rate = 0.002        # Tiny trickle of nutrients added each step
                                      # (like rain or natural processes)
    
    nutrient_from_decomp = 0.4        # When an organism dies, 40% of its remaining
                                      # energy becomes nutrients for others
    
    nutrient_max = 3.0                # Nutrients are capped at this value per cell
    
    # ── Organism Settings ──────────────────────────────────────────────────────
    # Each organism is essentially a little bag of energy with a genome.
    # It gains energy from photosynthesis and loses energy from:
    #   - Existing (maintenance cost each step)
    #   - Moving (small cost per step moved)
    #   - Toxic damage (when in polluted areas)
    #   - Reproducing (big cost to create offspring)
    # When energy hits 0, the organism dies.
    
    initial_population = 100          # How many organisms we start with
    energy_initial = 40.0             # Starting energy for each organism (and each baby)
    energy_max = 150.0                # Energy can't exceed this (prevents infinite hoarding)
    energy_reproduction_threshold = 80.0  # Must have at least this much energy to reproduce
    energy_reproduction_cost = 40.0   # Energy spent to create one offspring
    energy_maintenance_cost = 0.6     # Energy cost just for being alive each step
    energy_movement_cost = 0.2        # Extra energy cost for moving (staying still is free)
    photosynthesis_base = 3.0         # Base energy gained from photosynthesis per step
                                      # (modified by light, toxicity, genome, crowding, etc.)
    
    # ── Genome Settings ────────────────────────────────────────────────────────
    # Each organism has a "genome" — a small array of floating-point numbers
    # that control its behavior. Think of these as the organism's "personality":
    #
    #   - movement_params (8 numbers): Control how the organism decides where to move.
    #     Things like: "Do I follow light?", "Do I avoid toxicity?",
    #     "Do I stay put when nutrients are good?", "How random am I?"
    #
    #   - photo_params (4 numbers): Control photosynthesis behavior.
    #     Things like: "How efficient am I at converting light?",
    #     "How tolerant am I of toxicity?", "Can I use dim light?",
    #     "How much energy can I store?"
    #
    # Total genome = 12 numbers per organism.
    # These get passed from parent to child with small random mutations,
    # so over many generations, evolution happens naturally.
    
    movement_params_size = 8          # 8 genes controlling movement decisions
    photo_params_size = 4             # 4 genes controlling photosynthesis
    genome_size = movement_params_size + photo_params_size  # = 12 total genes
    
    # ── Generational Settings ──────────────────────────────────────────────────
    mutation_rate = 0.08              # How much each gene can randomly change between
                                      # parent and child. Higher = faster evolution but
                                      # also more "noise" (random bad mutations).
    
    min_reproduction_age = 8          # Must be at least 8 steps old to reproduce.
                                      # Prevents instant chain-reproduction.
    
    offspring_distance = 5            # Babies appear within 5 cells of their parent.
                                      # Not too close (competition) not too far (isolation).
    
    max_age = 200                     # Organisms die of old age after 200 steps.
                                      # Shorter lifespan = faster generational turnover
                                      # = faster evolution.
    
    # ── Sensing ────────────────────────────────────────────────────────────────
    sensing_range = 3                 # How far an organism can "see" (not currently used
                                      # in this version — sensing is only immediate neighbors)
    
    # ── Simulation Settings ────────────────────────────────────────────────────
    total_timesteps = 10000           # Run the simulation for 10,000 steps
    snapshot_interval = 100           # Save a snapshot every 100 steps (for analysis later)
    output_dir = "output_v3"          # Folder where all output files go
    random_seed = 42                  # Fixed seed so the simulation is reproducible.
                                      # Same seed = exact same results every time.


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD
# ═══════════════════════════════════════════════════════════════════════════════
# The World class IS the simulation. It holds:
#   - The environment (light, toxicity, nutrients for every cell on the grid)
#   - All organisms (their positions, energy, age, genomes, etc.)
#   - All the logic for what happens each time step
#
# The key idea: organisms and environment are stored as parallel numpy arrays.
# For example, if there are 500 organisms:
#   self.rows    = array of 500 row positions
#   self.cols    = array of 500 column positions
#   self.energy  = array of 500 energy values
#   self.genomes = 500 x 12 matrix of genomes
# Organism #7's data is at index 7 in ALL of these arrays.

class World:
    def __init__(self, cfg=None):
        """
        Initialize the world: create the environment grids, place the initial
        organisms, and set everything to its starting state.
        """
        # Use provided config, or create a default one
        self.cfg = cfg or Config()
        c = self.cfg  # shorthand so we don't type "self.cfg" everywhere
        
        # Random number generator with a fixed seed for reproducibility.
        # Using numpy's modern RNG instead of the older np.random functions.
        self.rng = np.random.default_rng(c.random_seed)
        
        # Current time step (starts at 0, increments by 1 each update)
        self.timestep = 0
        
        # Counter for assigning unique IDs to organisms
        self.next_id = 0
        
        # Shorthand for grid size
        N = c.grid_size  # 256
        
        # ══════════════════════════════════════════════════════════════════════
        # ENVIRONMENT GRIDS
        # ══════════════════════════════════════════════════════════════════════
        # Each of these is a 256x256 2D array. Every cell in the grid has
        # its own value for light, toxicity, nutrients, etc.
        
        # LIGHT GRID: A gradient from bright (top) to dim (bottom).
        # np.linspace creates an array like [1.0, 0.996, 0.992, ..., 0.05]
        # The [:, None] * np.ones trick broadcasts this 1D column into a full
        # 2D grid where every column is identical (light only varies by row).
        self.light = np.linspace(c.light_max, c.light_min, N)[:, None] * np.ones((1, N))
        
        # TOXICITY GRID: Starts nearly empty — just tiny random amounts (0 to 0.005).
        # This will build up as organisms produce waste.
        self.toxic = self.rng.uniform(0.0, 0.005, (N, N))
        
        # NUTRIENT GRID: Starts with small random amounts (0.02 to 0.08).
        # Will be fed by decomposing dead organisms.
        self.nutrients = self.rng.uniform(0.02, 0.08, (N, N))
        
        # DECOMPOSITION GRID: Tracks organic matter from dead organisms.
        # Starts empty. When organisms die, their energy gets deposited here,
        # and it slowly converts into nutrients.
        self.decomposition = np.zeros((N, N))
        
        # DENSITY GRID: How many organisms are on each cell.
        # Recalculated every step. Used for crowding/competition calculations.
        self.density = np.zeros((N, N), dtype=np.int32)
        
        # ══════════════════════════════════════════════════════════════════════
        # ZONE MAP: Spatial variation in how toxicity behaves
        # ══════════════════════════════════════════════════════════════════════
        # Divide the 256x256 grid into an 8x8 grid of zones (each zone is 32x32 cells).
        # Each zone gets a random multiplier that affects toxicity:
        #   0.3 = strong absorber (toxicity decays fast here — very safe)
        #   0.5 = moderate absorber
        #   0.7 = mild absorber
        #   1.0 = neutral (normal toxicity behavior) — appears twice so it's more common
        #   1.2 = mild amplifier
        #   1.5 = moderate amplifier
        #   2.0 = strong amplifier (toxicity builds up fast here — very dangerous)
        
        zone_size = N // c.zone_count  # 256 / 8 = 32 cells per zone
        self.zone_map = np.ones((N, N))  # Start with all 1.0 (neutral)
        
        # Loop over each zone and assign a random multiplier
        for zi in range(c.zone_count):       # zi = zone row index (0-7)
            for zj in range(c.zone_count):   # zj = zone column index (0-7)
                # Calculate the pixel boundaries of this zone
                r0, r1 = zi * zone_size, (zi + 1) * zone_size  # row start, row end
                c0, c1 = zj * zone_size, (zj + 1) * zone_size  # col start, col end
                
                # Randomly pick one of these multiplier values for this zone.
                # Note: 1.0 appears twice, making neutral zones more likely.
                val = self.rng.choice([0.3, 0.5, 0.7, 1.0, 1.0, 1.2, 1.5, 2.0])
                
                # Fill this entire zone with the chosen multiplier
                self.zone_map[r0:r1, c0:c1] = val
        
        # Smooth the zone boundaries so there aren't sharp edges.
        # This makes the zones blend into each other gradually (more realistic).
        # uniform_filter replaces each cell with the average of its 8x8 neighborhood.
        from scipy.ndimage import uniform_filter
        self.zone_map = uniform_filter(self.zone_map, size=8)
        
        # ══════════════════════════════════════════════════════════════════════
        # ORGANISMS (stored as parallel arrays)
        # ══════════════════════════════════════════════════════════════════════
        # Instead of creating 100 "Organism" objects, we store all organism
        # data in parallel arrays. This is MUCH faster because numpy can
        # process all 100 (or 10,000) organisms in one batch operation.
        #
        # Example with 3 organisms:
        #   self.rows    = [42, 100, 200]     → organism 0 is at row 42, etc.
        #   self.cols    = [10,  50, 180]     → organism 0 is at column 10, etc.
        #   self.energy  = [40,  40,  40]     → all start with 40 energy
        #   self.genomes = [[...12 numbers...], [...], [...]]  → each has 12 genes
        
        pop = c.initial_population  # 100 organisms to start
        
        # Place organisms at random positions on the grid
        self.rows = self.rng.integers(0, N, size=pop)   # Random row for each organism
        self.cols = self.rng.integers(0, N, size=pop)   # Random column for each organism
        
        # All organisms start with the same energy
        self.energy = np.full(pop, c.energy_initial)    # Array of 100 values, all 40.0
        
        # All organisms start at age 0
        self.age = np.zeros(pop, dtype=np.int32)
        
        # All initial organisms are "generation 0" (the founders).
        # Their children will be generation 1, grandchildren generation 2, etc.
        self.generation = np.zeros(pop, dtype=np.int32)
        
        # Unique ID for each organism (0 through 99 for the initial batch)
        self.ids = np.arange(pop, dtype=np.int64)
        
        # Parent IDs: -1 means "no parent" (these are the founding organisms)
        self.parent_ids = np.full(pop, -1, dtype=np.int64)
        
        # Next available ID for new organisms
        self.next_id = pop  # Will be 100, then 101, 102, etc. as babies are born
        
        # GENOMES: Random starting genomes for each organism.
        # Each genome is 12 floating-point numbers drawn from a normal distribution
        # centered at 0 with standard deviation 0.5.
        # Shape: (100, 12) — 100 organisms, each with 12 genes.
        self.genomes = self.rng.normal(0, 0.5, (pop, c.genome_size))
        
        # ══════════════════════════════════════════════════════════════════════
        # STATISTICS TRACKING
        # ══════════════════════════════════════════════════════════════════════
        # A list that will accumulate one dictionary of stats per timestep.
        # Used for plotting population curves, toxicity trends, etc. after the run.
        self.stats_history = []
    
    # ══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT UPDATE
    # ══════════════════════════════════════════════════════════════════════════
    
    def _update_environment(self):
        """
        Update all the environment grids for one time step.
        This handles: toxicity spreading & decaying, organisms producing waste,
        nutrient regeneration, and decomposition of dead organisms' remains.
        """
        c = self.cfg
        
        # ── STEP 1: Toxic Diffusion ──────────────────────────────────────────
        # Toxicity "spreads" to neighboring cells, like a drop of ink in water.
        # We calculate the average toxicity of each cell's 4 neighbors (up/down/left/right),
        # then blend the cell's value toward that average.
        #
        # Technical detail: np.pad adds a border of repeated edge values around the grid
        # so we don't get index errors when looking at neighbors of edge cells.
        
        k = c.toxic_diffusion_rate  # 0.06 — how fast diffusion happens
        p = np.pad(self.toxic, 1, mode='edge')  # Add 1-cell border (copies edge values)
        
        # Get the average of each cell's 4 neighbors:
        #   p[:-2, 1:-1] = the cell above each position
        #   p[2:, 1:-1]  = the cell below
        #   p[1:-1, :-2] = the cell to the left
        #   p[1:-1, 2:]  = the cell to the right
        neighbors = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        
        # Blend toward the neighbor average. If k=0.06, each cell moves 6% toward
        # the average of its neighbors. This gradually smooths out toxic hotspots.
        self.toxic += k * (neighbors - self.toxic)
        
        # ── STEP 2: Toxic Decay (modulated by zone map) ──────────────────────
        # Toxicity naturally fades over time (like pollution breaking down).
        # BUT the rate depends on the zone:
        #   - In absorbing zones (zone_map < 1): decay is FASTER (dividing by a small number)
        #   - In amplifying zones (zone_map > 1): decay is SLOWER (dividing by a large number)
        # np.maximum(0.3, ...) prevents division by very small numbers.
        
        effective_decay = c.toxic_decay_rate / np.maximum(0.3, self.zone_map)
        self.toxic *= (1.0 - effective_decay)  # Multiply by ~0.99 to remove ~1%
        
        # ── STEP 3: Organisms Produce Toxic Waste ────────────────────────────
        # Every living organism produces a small amount of toxicity at its location.
        # The amount is also multiplied by the zone map (amplifying zones make
        # organisms produce MORE effective waste).
        #
        # np.add.at is used because multiple organisms might be on the same cell,
        # and we need ALL their contributions to add up (regular indexing wouldn't).
        
        if len(self.rows) > 0:  # Only if there are living organisms
            base_output = np.full(len(self.rows), c.toxic_production_rate)  # 0.015 each
            zone_mult = self.zone_map[self.rows, self.cols]  # Zone multiplier at each organism's position
            np.add.at(self.toxic, (self.rows, self.cols), base_output * zone_mult)
        
        # ── STEP 4: Nutrient Regeneration ────────────────────────────────────
        # A tiny amount of nutrients is added everywhere each step.
        # Think of it as natural processes: minerals dissolving, rain, etc.
        self.nutrients += c.nutrient_base_rate  # +0.002 everywhere
        
        # ── STEP 5: Decomposition → Nutrients ────────────────────────────────
        # Dead organisms leave behind decomposing matter (see _kill_and_decompose).
        # Each step, 1.5% of that decomposing matter converts to usable nutrients.
        # The decomposition pile also slowly shrinks on its own (0.2% per step).
        
        transfer = self.decomposition * 0.015   # 1.5% of decomp becomes nutrients
        self.nutrients += transfer               # Add it to the nutrient grid
        self.decomposition -= transfer           # Remove it from the decomp grid
        self.decomposition *= 0.998              # Slow baseline decay of remaining decomp
        
        # ── STEP 6: Clamp Everything to Valid Ranges ─────────────────────────
        # Make sure no values go negative or exceed their maximum.
        # np.clip(array, min, max) forces all values into the range [min, max].
        
        np.clip(self.toxic, 0, 5.0, out=self.toxic)           # Toxicity: 0 to 5.0
        np.clip(self.nutrients, 0, c.nutrient_max, out=self.nutrients)  # Nutrients: 0 to 3.0
        np.clip(self.decomposition, 0, 10.0, out=self.decomposition)   # Decomp: 0 to 10.0
    
    def _update_density(self):
        """
        Count how many organisms are on each cell of the grid.
        Result is stored in self.density (a 256x256 grid of integers).
        
        This is used later to model competition: if 10 organisms are on the
        same cell, they each get less sunlight (they're blocking each other).
        """
        self.density[:] = 0  # Reset to zero everywhere
        if len(self.rows) > 0:
            # For each organism, add 1 to the density at its (row, col) position.
            # np.add.at handles the case where multiple organisms share a cell.
            np.add.at(self.density, (self.rows, self.cols), 1)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SENSING: How organisms perceive their environment
    # ══════════════════════════════════════════════════════════════════════════
    
    def _sense_local(self):
        """
        Each organism "reads" its local environment. It can sense:
          1. Light level at its current cell
          2. Toxicity at its current cell
          3. Nutrients at its current cell
          4. Crowd density at its current cell (how many organisms are here)
          5-6. Light gradient (is there more light up/down/left/right?)
          7-8. Toxic gradient (is there more toxicity up/down/left/right?)
        
        These 8 sensor readings are what the organism's genome uses to decide
        what to do (move, stay, etc.). Think of it as the organism's "senses".
        
        Returns a matrix of shape (num_organisms, 8) — 8 readings per organism.
        """
        rows, cols = self.rows, self.cols
        N = self.cfg.grid_size
        
        # Read the value at each organism's exact position
        local_light = self.light[rows, cols]           # How bright is it here?
        local_toxic = self.toxic[rows, cols]            # How toxic is it here?
        local_nutrients = self.nutrients[rows, cols]     # How many nutrients are here?
        local_density = self.density[rows, cols].astype(np.float64)  # How crowded is it here?
        
        # Calculate gradient = "which direction is there MORE of this?"
        # For each organism, we look at the cell above vs. below, and right vs. left.
        # np.clip ensures we don't go off the edge of the grid.
        
        r_up = np.clip(rows - 1, 0, N - 1)   # Row above (clamped to grid edge)
        r_dn = np.clip(rows + 1, 0, N - 1)   # Row below
        c_lt = np.clip(cols - 1, 0, N - 1)    # Column to the left
        c_rt = np.clip(cols + 1, 0, N - 1)    # Column to the right
        
        # Light gradient: positive means "more light upward" (row - 1 direction)
        light_grad_y = self.light[r_up, cols] - self.light[r_dn, cols]
        # Light gradient: positive means "more light to the right"
        light_grad_x = self.light[rows, c_rt] - self.light[rows, c_lt]
        
        # Toxic gradient: positive means "more toxicity upward"
        toxic_grad_y = self.toxic[r_up, cols] - self.toxic[r_dn, cols]
        # Toxic gradient: positive means "more toxicity to the right"
        toxic_grad_x = self.toxic[rows, c_rt] - self.toxic[rows, c_lt]
        
        # Stack all 8 readings into one big matrix.
        # Row i of this matrix = all 8 readings for organism i.
        return np.column_stack([
            local_light, local_toxic, local_nutrients, local_density,
            light_grad_y, light_grad_x, toxic_grad_y, toxic_grad_x
        ])
    
    # ══════════════════════════════════════════════════════════════════════════
    # PHOTOSYNTHESIS: How organisms gain energy
    # ══════════════════════════════════════════════════════════════════════════
    
    def _photosynthesize(self, readings):
        """
        Calculate how much energy each organism gains from photosynthesis this step.
        
        The energy gained depends on FIVE factors, all multiplied together:
          1. Base efficiency    — determined by the organism's genome (gene 0)
          2. Toxic penalty      — high toxicity reduces energy gain (gene 1 = tolerance)
          3. Light availability — more light = more energy (gene 2 = sensitivity)
          4. Storage capacity   — genome-based cap on energy intake (gene 3)
          5. Crowding penalty   — sharing a cell = sharing the light
        
        This is where evolution happens for photosynthesis: organisms with better
        photo_params genes will harvest more energy and be more likely to survive
        and reproduce, passing those genes to their offspring.
        
        Args:
            readings: The 8-column sensor matrix from _sense_local()
        
        Returns:
            Array of energy gains, one per organism.
        """
        c = self.cfg
        
        # Extract the 4 photosynthesis genes for each organism.
        # self.genomes has shape (num_organisms, 12).
        # Columns 0-7 are movement genes, columns 8-11 are photo genes.
        photo = self.genomes[:, c.movement_params_size:]  # Shape: (num_organisms, 4)
        
        # Extract relevant sensor readings
        local_light = readings[:, 0]    # Light at this cell
        local_toxic = readings[:, 1]    # Toxicity at this cell
        local_density = readings[:, 3]  # Crowding at this cell
        
        # ── Factor 1: Base Efficiency ────────────────────────────────────────
        # Gene photo[:, 0] controls base efficiency.
        # np.tanh squishes any value to the range (-1, 1).
        # So efficiency ranges from photosynthesis_base * 0.7 to * 1.3.
        # Organisms with higher gene values are naturally better at photosynthesis.
        efficiency = c.photosynthesis_base * (1.0 + 0.3 * np.tanh(photo[:, 0]))
        
        # ── Factor 2: Toxic Penalty ──────────────────────────────────────────
        # Gene photo[:, 1] controls toxic tolerance.
        # Higher tolerance gene → organism is more resistant to toxicity's effect
        # on photosynthesis. But nobody is fully immune.
        # The penalty ranges from 0.0 (completely blocked) to 1.0 (no effect).
        tolerance = np.maximum(0.1, 1.0 + 0.5 * np.tanh(photo[:, 1]))
        toxic_penalty = np.maximum(0.0, 1.0 - local_toxic * c.toxic_photo_penalty / tolerance)
        
        # ── Factor 3: Light Sensitivity ──────────────────────────────────────
        # Gene photo[:, 2] controls how well the organism uses available light.
        # light_exp determines the curve shape: lower exponent = better at using dim light.
        # An organism at the bottom of the grid (dim light) benefits from a low exponent.
        light_exp = np.maximum(0.3, 1.0 - 0.3 * np.tanh(photo[:, 2]))
        light_mult = np.power(np.maximum(local_light, 0.01), light_exp)
        
        # ── Factor 4: Storage Capacity ───────────────────────────────────────
        # Gene photo[:, 3] controls how much of the generated energy the organism
        # can actually absorb. Uses a sigmoid function (0.5 to 1.0 range).
        # Think of it as "how big is this organism's energy storage tank?"
        storage = 0.5 + 0.5 / (1.0 + np.exp(-photo[:, 3]))
        
        # ── Factor 5: Density Competition ────────────────────────────────────
        # If multiple organisms are on the same cell, they share the sunlight.
        # More crowded = less energy per organism. This prevents infinite growth
        # even in well-lit areas — eventually organisms start stealing each
        # other's light.
        share = 1.0 / np.maximum(1.0, local_density * 0.5)
        
        # Multiply all five factors together for the final energy gain.
        # If ANY factor is 0, the total is 0 (e.g., total toxic shutdown).
        return efficiency * toxic_penalty * light_mult * storage * share
    
    # ══════════════════════════════════════════════════════════════════════════
    # TOXIC DAMAGE: The key mechanic that controls population
    # ══════════════════════════════════════════════════════════════════════════
    
    def _apply_toxic_damage(self, readings):
        """
        Apply direct energy damage to organisms based on local toxicity.
        
        This is THE most important mechanic in the simulation. Without this,
        organisms would multiply without limit (since there's no hard population
        cap). Here's how it creates natural carrying capacity:
        
          1. More organisms → more toxic waste produced
          2. More waste → toxicity rises above thresholds
          3. Above medium threshold (0.8): organisms start losing energy
          4. Above high threshold (1.5): organisms die rapidly
          5. Deaths → fewer organisms → less waste → toxicity drops
          6. Lower toxicity → survivors thrive → population grows again
        
        This negative feedback loop keeps the population oscillating around
        a sustainable level.
        
        Args:
            readings: The 8-column sensor matrix from _sense_local()
        """
        c = self.cfg
        local_toxic = readings[:, 1]  # Toxicity at each organism's position
        
        # Start with zero damage for everyone
        damage = np.zeros(len(self.rows))
        
        # ── Medium Threshold Damage ──────────────────────────────────────────
        # If toxicity at this cell is above 0.8, the organism takes moderate damage.
        # Damage is proportional to HOW MUCH above the threshold it is.
        # Example: toxicity = 1.0, threshold = 0.8, excess = 0.2
        #          damage = 0.2 * 1.5 = 0.3 energy lost this step
        
        medium_mask = local_toxic > c.toxic_threshold_medium  # Which organisms are above 0.8?
        if medium_mask.any():  # Only compute if at least one organism is affected
            excess = local_toxic[medium_mask] - c.toxic_threshold_medium  # How much over?
            damage[medium_mask] += excess * c.toxic_damage_medium  # 1.5 damage per unit excess
        
        # ── High Threshold Damage (STACKS with medium damage) ────────────────
        # If toxicity is above 1.5, the organism takes ADDITIONAL severe damage
        # on top of the medium damage. This makes very toxic areas lethal.
        # Example: toxicity = 2.0, high threshold = 1.5, excess = 0.5
        #          additional damage = 0.5 * 5.0 = 2.5 energy lost
        #          (plus the medium damage from above)
        
        high_mask = local_toxic > c.toxic_threshold_high  # Which organisms are above 1.5?
        if high_mask.any():
            excess = local_toxic[high_mask] - c.toxic_threshold_high
            damage[high_mask] += excess * c.toxic_damage_high  # 5.0 damage per unit excess
        
        # Subtract the damage from each organism's energy.
        # If this drops energy to 0 or below, the organism will die in _kill_and_decompose().
        self.energy -= damage
    
    # ══════════════════════════════════════════════════════════════════════════
    # MOVEMENT: How organisms decide where to go
    # ══════════════════════════════════════════════════════════════════════════
    
    def _decide_movement(self, readings):
        """
        Each organism uses its genome to decide whether to stay still or
        move in one of 4 directions (up, down, left, right).
        
        The decision works like a simple neural network:
          1. Read sensor inputs (light, toxicity, gradients, etc.)
          2. Multiply inputs by genome weights (the movement genes)
          3. The direction with the highest "score" wins
          4. A bit of randomness is added (exploration noise)
        
        The 5 possible actions:
          0 = Stay still (no movement cost)
          1 = Move up    (row - 1)
          2 = Move down  (row + 1)
          3 = Move right (col + 1)
          4 = Move left  (col - 1)
        
        Over generations, evolution tunes these genome weights so organisms
        learn to move toward light and away from toxicity.
        
        Args:
            readings: The 8-column sensor matrix from _sense_local()
        
        Returns:
            Array of action choices (0-4), one per organism.
        """
        c = self.cfg
        n = len(self.rows)  # Number of living organisms
        if n == 0:
            return np.array([], dtype=np.int32)  # Nobody to move
        
        # Extract the 8 movement genes for each organism.
        # Columns 0-7 of the genome.
        move_p = self.genomes[:, :c.movement_params_size]  # Shape: (n, 8)
        
        # Scores for each organism's 5 possible actions.
        # Higher score = more likely to choose that action.
        scores = np.zeros((n, 5))  # Shape: (num_organisms, 5_actions)
        
        # Pull out the sensor readings we'll use
        local_nutrients = readings[:, 2]   # Nutrients here (good — stay!)
        local_density = readings[:, 3]     # Crowding here (bad — leave!)
        light_grad_y = readings[:, 4]      # Light gradient vertical
        light_grad_x = readings[:, 5]      # Light gradient horizontal
        toxic_grad_y = readings[:, 6]      # Toxic gradient vertical
        toxic_grad_x = readings[:, 7]      # Toxic gradient horizontal
        
        # ── Score for STAYING (action 0) ─────────────────────────────────────
        # Influenced by:
        #   move_p[:, 5]: Base "laziness" gene — how much does this organism like staying?
        #   move_p[:, 2] * nutrients: Stays more if there are good nutrients here
        #   move_p[:, 1] * density: Wants to leave if it's too crowded
        scores[:, 0] = (move_p[:, 5] 
                        + move_p[:, 2] * local_nutrients 
                        - move_p[:, 1] * np.minimum(local_density, 10) * 0.1)
        
        # ── Score for moving UP (action 1) ───────────────────────────────────
        # Moves toward more light (positive light gradient = more light above)
        # Moves away from toxicity (negative toxic gradient = less toxicity above)
        # move_p[:, 6] = "light-seeking" gene weight
        # move_p[:, 7] = "toxic-avoiding" gene weight
        scores[:, 1] = move_p[:, 6] * light_grad_y - move_p[:, 7] * toxic_grad_y
        
        # ── Score for moving DOWN (action 2) ─────────────────────────────────
        # Opposite of up: negative gradients favor downward movement
        scores[:, 2] = -move_p[:, 6] * light_grad_y + move_p[:, 7] * toxic_grad_y
        
        # ── Score for moving RIGHT (action 3) ────────────────────────────────
        scores[:, 3] = move_p[:, 6] * light_grad_x - move_p[:, 7] * toxic_grad_x
        
        # ── Score for moving LEFT (action 4) ─────────────────────────────────
        scores[:, 4] = -move_p[:, 6] * light_grad_x + move_p[:, 7] * toxic_grad_x
        
        # ── Add Exploration Noise ────────────────────────────────────────────
        # Gene move_p[:, 4] controls how "random" this organism's movement is.
        # High noise gene = organism explores more, makes unexpected moves.
        # Low noise gene = organism is more deterministic, follows gradients strictly.
        # This noise is crucial for evolution — without randomness, organisms
        # couldn't discover new strategies.
        scores += move_p[:, 4:5] * self.rng.normal(0, 0.3, (n, 5))
        
        # Pick the action with the highest score for each organism.
        # np.argmax returns the index (0-4) of the maximum value in each row.
        return np.argmax(scores, axis=1)
    
    def _execute_movement(self, actions):
        """
        Actually move the organisms based on their chosen actions.
        
        Actions:
          0 = Stay (do nothing)
          1 = Up (row -= 1)
          2 = Down (row += 1)
          3 = Right (col += 1)
          4 = Left (col -= 1)
        
        Movement is clamped to grid boundaries (organisms can't walk off the edge).
        Moving costs a small amount of energy; staying still is free.
        
        Args:
            actions: Array of action choices (0-4), one per organism.
        """
        c = self.cfg
        N = c.grid_size  # 256
        
        # Move UP: decrease row (but not below 0)
        mask = actions == 1
        self.rows[mask] = np.maximum(0, self.rows[mask] - 1)
        
        # Move DOWN: increase row (but not above 255)
        mask = actions == 2
        self.rows[mask] = np.minimum(N - 1, self.rows[mask] + 1)
        
        # Move RIGHT: increase column (but not above 255)
        mask = actions == 3
        self.cols[mask] = np.minimum(N - 1, self.cols[mask] + 1)
        
        # Move LEFT: decrease column (but not below 0)
        mask = actions == 4
        self.cols[mask] = np.maximum(0, self.cols[mask] - 1)
        
        # Deduct movement energy cost from all organisms that moved.
        # Staying still (action 0) costs nothing.
        moved = actions > 0
        self.energy[moved] -= c.energy_movement_cost  # -0.2 energy per move
    
    # ══════════════════════════════════════════════════════════════════════════
    # REPRODUCTION: How organisms create offspring
    # ══════════════════════════════════════════════════════════════════════════
    
    def _reproduce(self):
        """
        Organisms with enough energy and age can reproduce asexually.
        
        Requirements to reproduce:
          - Energy >= 80 (reproduction threshold)
          - Age >= 8 (minimum reproduction age)
        
        What happens:
          1. Parent loses 40 energy (reproduction cost)
          2. A baby appears nearby (within 5 cells of parent)
          3. Baby gets the parent's genome with small random mutations
          4. Baby starts with 40 energy and age 0
          5. Baby's generation = parent's generation + 1
        
        This is where EVOLUTION happens:
          - Successful organisms (those that survive long enough and accumulate
            enough energy) get to reproduce
          - Their children inherit the parent's genome with slight changes
          - Over many generations, the population evolves toward genomes that
            are better at surviving in this environment
        """
        c = self.cfg
        
        # Find which organisms meet BOTH requirements to reproduce
        can_reproduce = (
            (self.energy >= c.energy_reproduction_threshold) &  # Has enough energy (>= 80)
            (self.age >= c.min_reproduction_age)                 # Is old enough (>= 8 steps)
        )
        
        # Get the indices of organisms that can reproduce
        parent_idx = np.where(can_reproduce)[0]
        n_births = len(parent_idx)
        if n_births == 0:
            return  # Nobody is ready to reproduce
        
        # Parents pay the reproduction cost
        self.energy[parent_idx] -= c.energy_reproduction_cost  # -40 energy
        
        # ── Place the babies near their parents ──────────────────────────────
        # Random offset of -5 to +5 cells in each direction
        offsets_r = self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, n_births)
        offsets_c = self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, n_births)
        
        # Baby position = parent position + offset, clamped to grid boundaries
        child_rows = np.clip(self.rows[parent_idx] + offsets_r, 0, c.grid_size - 1)
        child_cols = np.clip(self.cols[parent_idx] + offsets_c, 0, c.grid_size - 1)
        
        # ── Create the baby genomes with mutations ───────────────────────────
        # Start with a copy of the parent's genome, then add small random noise.
        # mutation_rate = 0.08, so each gene changes by a random amount drawn
        # from a normal distribution with standard deviation 0.08.
        # Most mutations are tiny, but occasionally a gene shifts significantly.
        child_genomes = (self.genomes[parent_idx] 
                         + self.rng.normal(0, c.mutation_rate, (n_births, c.genome_size)))
        
        # ── Set up all the other baby attributes ─────────────────────────────
        child_energy = np.full(n_births, c.energy_initial)          # 40 energy each
        child_age = np.zeros(n_births, dtype=np.int32)              # Age 0 (newborn)
        child_gen = self.generation[parent_idx] + 1                 # One generation beyond parent
        child_ids = np.arange(self.next_id, self.next_id + n_births, dtype=np.int64)  # Unique IDs
        child_parent_ids = self.ids[parent_idx]                     # Record who the parent was
        self.next_id += n_births  # Advance the ID counter
        
        # ── Add babies to the population ─────────────────────────────────────
        # Append the new organisms to the end of every parallel array.
        # After this, the arrays are longer by n_births elements.
        self.rows = np.concatenate([self.rows, child_rows])
        self.cols = np.concatenate([self.cols, child_cols])
        self.energy = np.concatenate([self.energy, child_energy])
        self.age = np.concatenate([self.age, child_age])
        self.generation = np.concatenate([self.generation, child_gen])
        self.genomes = np.concatenate([self.genomes, child_genomes])
        self.ids = np.concatenate([self.ids, child_ids])
        self.parent_ids = np.concatenate([self.parent_ids, child_parent_ids])
    
    # ══════════════════════════════════════════════════════════════════════════
    # DEATH & DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _kill_and_decompose(self):
        """
        Remove dead organisms and convert their remains into decomposition matter.
        
        An organism dies if:
          - Its energy drops to 0 or below (starvation, toxic damage, etc.)
          - Its age exceeds max_age (200 steps — death of old age)
        
        When an organism dies:
          1. Its remaining energy (if any) is deposited as decomposition matter
             at its cell location, plus a small bonus (+0.5)
          2. The decomposition matter will slowly convert to nutrients over time
             (handled in _update_environment), feeding the survivors
          3. The dead organism is removed from all arrays
        
        This creates a nutrient cycle: death feeds the ecosystem.
        """
        c = self.cfg
        
        # Determine who is alive: has energy AND is under the age limit
        alive = (self.energy > 0) & (self.age < c.max_age)
        dead = ~alive  # The inverse — everyone who is NOT alive
        
        # If there are dead organisms, deposit their remains
        if dead.any():
            dead_rows = self.rows[dead]
            dead_cols = self.cols[dead]
            dead_energy = np.maximum(0, self.energy[dead])  # Can't deposit negative energy
            
            # Each dead organism deposits 40% of its remaining energy + 0.5 base
            # as decomposition matter at its location
            deposits = dead_energy * c.nutrient_from_decomp + 0.5
            np.add.at(self.decomposition, (dead_rows, dead_cols), deposits)
        
        # Remove dead organisms by keeping only the alive ones.
        # This "filters" every parallel array to only include living organisms.
        # After this, if 50 out of 500 died, all arrays shrink from 500 to 450 elements.
        self.rows = self.rows[alive]
        self.cols = self.cols[alive]
        self.energy = self.energy[alive]
        self.age = self.age[alive]
        self.generation = self.generation[alive]
        self.genomes = self.genomes[alive]
        self.ids = self.ids[alive]
        self.parent_ids = self.parent_ids[alive]
    
    # ══════════════════════════════════════════════════════════════════════════
    # MAIN SIMULATION LOOP (one time step)
    # ══════════════════════════════════════════════════════════════════════════
    
    def update(self):
        """
        Execute ONE time step of the simulation.
        This is called 10,000 times during a full run.
        
        The order matters! Here's the sequence:
        
          1. COUNT DENSITY    — How crowded is each cell?
          2. SENSE            — Each organism reads its local environment
          3. PHOTOSYNTHESIS   — Organisms gain energy from sunlight
          4. TOXIC DAMAGE     — Organisms in polluted areas lose energy
          5. MOVEMENT         — Organisms decide where to move and move there
          6. AGING            — Everyone gets 1 step older, pays maintenance cost
          7. REPRODUCTION     — Organisms with enough energy create offspring
          8. DEATH            — Organisms with no energy or max age are removed
          9. ENVIRONMENT      — Update toxicity, nutrients, decomposition
          10. RECORD STATS    — Log population, energy, toxicity, etc.
        
        If the population reaches 0 (extinction), we skip organism steps
        and only update the environment (to watch toxicity dissipate).
        """
        # Edge case: if everyone is dead, just update the environment
        if len(self.rows) == 0:
            self._update_environment()
            self.timestep += 1
            self._record_stats()
            return
        
        # 1. Count how many organisms are on each cell
        self._update_density()
        
        # 2. Each organism senses its local environment (8 sensor readings)
        readings = self._sense_local()
        
        # 3. Organisms convert sunlight into energy (modified by genes, toxicity, crowding)
        energy_gain = self._photosynthesize(readings)
        self.energy = np.minimum(self.energy + energy_gain, self.cfg.energy_max)
        # ^ np.minimum caps energy at 150 so organisms can't hoard infinite energy
        
        # 4. THE KEY MECHANIC: Organisms in toxic areas take direct energy damage
        self._apply_toxic_damage(readings)
        
        # 5. Each organism decides which direction to move (based on genome + senses)
        #    and then actually moves there
        actions = self._decide_movement(readings)
        self._execute_movement(actions)
        
        # 6. Every organism ages by 1 step and pays a maintenance cost just for being alive
        self.age += 1
        self.energy -= self.cfg.energy_maintenance_cost  # -0.6 energy per step
        
        # 7. Organisms with enough energy reproduce (creating mutated offspring nearby)
        self._reproduce()
        
        # 8. Remove dead organisms (energy <= 0 or too old) and deposit their remains
        self._kill_and_decompose()
        
        # 9. Update the environment: toxic diffusion/decay, nutrient regen, decomposition
        self._update_environment()
        
        # Advance the clock
        self.timestep += 1
        
        # Record statistics for this timestep
        self._record_stats()
    
    def _record_stats(self):
        """
        Record a snapshot of key metrics for this timestep.
        These stats are saved at the end and can be used to plot
        population curves, toxicity trends, evolutionary progress, etc.
        """
        pop = len(self.rows)  # Current population count
        
        # Count how many organisms are in each toxicity zone
        # This tells us: are organisms finding safe areas, or are they stuck in toxic ones?
        if pop > 0:
            local_toxic = self.toxic[self.rows, self.cols]  # Toxicity at each organism's cell
            
            # Safe zone: toxicity below 0.3
            in_low = int(np.sum(local_toxic < self.cfg.toxic_threshold_low))
            
            # Stressed zone: toxicity between 0.3 and 0.8
            in_med = int(np.sum((local_toxic >= self.cfg.toxic_threshold_low) & 
                               (local_toxic < self.cfg.toxic_threshold_medium)))
            
            # Danger zone: toxicity above 0.8
            in_high = int(np.sum(local_toxic >= self.cfg.toxic_threshold_medium))
        else:
            in_low = in_med = in_high = 0
        
        # Append this timestep's stats to the history list
        self.stats_history.append({
            "t": self.timestep,                                               # Current time step
            "pop": pop,                                                        # Total population
            "avg_energy": float(self.energy.mean()) if pop > 0 else 0,        # Average energy
            "min_energy": float(self.energy.min()) if pop > 0 else 0,         # Lowest energy
            "max_gen": int(self.generation.max()) if pop > 0 else 0,          # Highest generation #
            "avg_age": float(self.age.mean()) if pop > 0 else 0,              # Average age
            "toxic_mean": float(self.toxic.mean()),                            # Avg toxicity across grid
            "toxic_max": float(self.toxic.max()),                              # Peak toxicity anywhere
            "toxic_p95": float(np.percentile(self.toxic, 95)),                # 95th percentile toxic
            "decomp_mean": float(self.decomposition.mean()),                   # Avg decomposition
            "nutrient_mean": float(self.nutrients.mean()),                     # Avg nutrients
            "in_low_toxic": in_low,                                            # # in safe zone
            "in_med_toxic": in_med,                                            # # in stressed zone
            "in_high_toxic": in_high,                                          # # in danger zone
        })
    
    # ══════════════════════════════════════════════════════════════════════════
    # SNAPSHOT SAVING: Saving simulation state to files
    # ══════════════════════════════════════════════════════════════════════════
    
    def snapshot(self):
        """
        Create a JSON-serializable dictionary capturing the current state
        of the simulation. This is saved to disk periodically so we can
        analyze or visualize the simulation later.
        
        To keep file sizes manageable, if there are more than 500 organisms,
        we randomly sample 500 of them (we don't need all 10,000 in the snapshot).
        """
        pop = len(self.rows)
        
        # If population is large, randomly sample 500 organisms to save
        if pop > 500:
            idx = self.rng.choice(pop, 500, replace=False)  # Pick 500 random indices
        else:
            idx = np.arange(pop)  # Save all of them
        
        # Build a list of organism dictionaries (one per sampled organism)
        organisms = []
        for i in idx:
            organisms.append({
                "id": int(self.ids[i]),                      # Unique ID
                "row": int(self.rows[i]),                    # Position (row)
                "col": int(self.cols[i]),                    # Position (column)
                "energy": round(float(self.energy[i]), 2),   # Current energy (2 decimal places)
                "age": int(self.age[i]),                     # Current age in steps
                "generation": int(self.generation[i]),       # Which generation (0 = founder)
                "genome": self.genomes[i].tolist(),          # All 12 genes as a list
            })
        
        # Get the latest stats entry
        s = self.stats_history[-1] if self.stats_history else {}
        
        # Return the complete snapshot dictionary
        return {
            "timestep": self.timestep,
            "population": pop,
            "sampled": len(organisms),      # How many organisms are in this snapshot
            "organisms": organisms,
            "environment": {
                "toxic_mean": s.get("toxic_mean", 0),
                "toxic_max": s.get("toxic_max", 0),
                "toxic_p95": s.get("toxic_p95", 0),
                "nutrient_mean": s.get("nutrient_mean", 0),
                "decomp_mean": s.get("decomp_mean", 0),
            },
            "stats": s,
        }
    
    def save_snapshot(self, output_dir):
        """
        Save a JSON snapshot of the current simulation state to a file.
        File is named like: snapshot_000100.json (for timestep 100).
        """
        os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist
        data = self.snapshot()
        path = os.path.join(output_dir, f"snapshot_{self.timestep:06d}.json")
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def save_env_snapshot(self, output_dir):
        """
        Save the full environment grids (toxicity, nutrients, density, etc.)
        as a compressed numpy file (.npz). This is for detailed visualization.
        
        These are saved less frequently than organism snapshots because the
        full 256x256 grids are bigger than a list of organism positions.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"env_{self.timestep:06d}.npz")
        np.savez_compressed(path,
            toxic=self.toxic.astype(np.float32),             # Toxicity grid
            decomposition=self.decomposition.astype(np.float32),  # Decomposition grid
            nutrients=self.nutrients.astype(np.float32),      # Nutrient grid
            density=self.density.astype(np.int16),            # Organism density grid
            zone_map=self.zone_map.astype(np.float32),        # Zone multiplier map
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(cfg=None):
    """
    Run the full simulation from start to finish.
    
    This function:
      1. Creates the world with the given config
      2. Runs update() for 10,000 timesteps
      3. Saves snapshots at regular intervals
      4. Prints progress to the console
      5. Handles extinction gracefully (keeps running environment for 200 more steps)
      6. Saves a final summary JSON with all config + stats history
    
    Args:
        cfg: Optional Config object. If None, uses default settings.
    
    Returns:
        The World object after the simulation completes.
    """
    cfg = cfg or Config()
    world = World(cfg)
    
    # Print header with simulation parameters
    print(f"The Shimmering Field — Phase 1 v3 (tuned)")
    print(f"Grid: {cfg.grid_size}x{cfg.grid_size}  |  Pop: {cfg.initial_population}")
    print(f"Toxic thresholds: low={cfg.toxic_threshold_low} med={cfg.toxic_threshold_medium} high={cfg.toxic_threshold_high}")
    print(f"Running {cfg.total_timesteps} timesteps...")
    print(f"{'─' * 80}")
    
    start = time.time()  # Start the stopwatch
    env_save_interval = 500  # Save full environment grids every 500 steps (less often than organism snapshots)
    
    # ══════════════════════════════════════════════════════════════════════════
    # THE MAIN LOOP: run for 10,000 timesteps
    # ══════════════════════════════════════════════════════════════════════════
    for t in range(cfg.total_timesteps):
        # Execute one time step (sense → photosynthesize → damage → move → reproduce → die → environment)
        world.update()
        
        # ── Save snapshot and print progress every 100 steps ─────────────────
        if world.timestep % cfg.snapshot_interval == 0:
            world.save_snapshot(cfg.output_dir)
            s = world.stats_history[-1]   # Get the latest stats
            elapsed = time.time() - start  # How long has the simulation been running?
            
            # Format the toxic zone distribution for display
            toxic_zone = f"safe={s['in_low_toxic']:4d} med={s['in_med_toxic']:4d} high={s['in_high_toxic']:4d}"
            
            # Print a status line showing key metrics:
            #   t = timestep, pop = population, e = avg energy, gen = max generation,
            #   tox = mean/max toxicity, zone breakdown, elapsed time
            print(
                f"  t={s['t']:5d}  |  pop={s['pop']:5d}  |  "
                f"e={s['avg_energy']:5.1f}  |  gen={s['max_gen']:4d}  |  "
                f"tox={s['toxic_mean']:.3f}/{s['toxic_max']:.2f}  |  "
                f"{toxic_zone}  |  {elapsed:.1f}s"
            )
        
        # ── Save full environment grids every 500 steps ──────────────────────
        if world.timestep % env_save_interval == 0:
            world.save_env_snapshot(cfg.output_dir)
        
        # ── Handle extinction ────────────────────────────────────────────────
        # If every organism has died, the simulation doesn't just stop.
        # We continue running the environment for 200 more steps so we can
        # watch what happens to toxicity and nutrients after all life is gone
        # (they should gradually decay and normalize).
        if len(world.rows) == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            for _ in range(200):
                world._update_environment()
                world.timestep += 1
                world._record_stats()
            break  # Exit the main loop
    
    # ══════════════════════════════════════════════════════════════════════════
    # WRAP UP: Print final summary and save results
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start
    print(f"{'─' * 80}")
    print(f"Complete in {elapsed:.1f}s  |  Final pop: {len(world.rows)}")
    
    # Save a comprehensive summary file containing:
    #   - All config values (so we know exactly what settings produced these results)
    #   - The complete stats history (one entry per timestep — for plotting)
    os.makedirs(cfg.output_dir, exist_ok=True)
    summary = {
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
        "stats_history": world.stats_history,
    }
    path = os.path.join(cfg.output_dir, "run_summary.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {path}")
    
    return world


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
# This is the standard Python pattern for "run this if the file is executed
# directly (not imported as a module)". 
# Running `python phase1_v3.py` will call run_simulation() with default config.

if __name__ == "__main__":
    run_simulation()
