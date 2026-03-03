"""
The Shimmering Field — Phase 1 v2: Minimal Viable Ecology
==========================================================
Vectorized with NumPy. Density-dependent resource competition
creates natural carrying capacity.

Single world, 256x256 grid, photosynthetic + movement modules,
vertical inheritance with mutation, no viral system.
"""

import numpy as np
import json
import os
import time

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

class Config:
    # Grid
    grid_size = 256
    
    # Light gradient
    light_max = 1.0
    light_min = 0.1
    
    # Toxicity
    toxic_decay_rate = 0.02
    toxic_diffusion_rate = 0.05
    toxic_production_rate = 0.008
    
    # Nutrients
    nutrient_base_rate = 0.003
    nutrient_from_decomp = 0.3
    
    # Organisms
    initial_population = 100
    max_population = 8000           # hard cap safety valve
    energy_initial = 50.0
    energy_max = 200.0
    energy_reproduction_threshold = 120.0
    energy_reproduction_cost = 60.0
    energy_maintenance_cost = 0.8
    energy_movement_cost = 0.3
    photosynthesis_base = 2.5       # base energy from photosynthesis at max light
    toxic_photo_penalty = 0.5       # how much toxic load reduces photosynthesis
    
    # Genome sizes
    movement_params_size = 8
    photo_params_size = 4
    genome_size = movement_params_size + photo_params_size  # 12 total
    
    # Reproduction
    mutation_rate = 0.05
    min_reproduction_age = 15
    offspring_distance = 4
    
    # Death
    max_age = 400
    
    # Sensing
    sensing_range = 3
    
    # Simulation
    total_timesteps = 5000
    snapshot_interval = 50
    output_dir = "output"
    random_seed = 42


# ─────────────────────────────────────────────
# Vectorized World
# ─────────────────────────────────────────────

class World:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.rng = np.random.default_rng(c.random_seed)
        self.timestep = 0
        self.next_id = 0
        
        N = c.grid_size
        
        # ── Environment grids ──
        self.light = np.linspace(c.light_max, c.light_min, N)[:, None] * np.ones((1, N))
        self.toxic = self.rng.uniform(0.0, 0.01, (N, N))
        self.nutrients = self.rng.uniform(0.01, 0.05, (N, N))
        self.decomposition = np.zeros((N, N))
        
        # Density map — updated each step
        self.density = np.zeros((N, N), dtype=np.int32)
        
        # ── Organism state arrays ──
        # Positions, energy, age, genome all stored as parallel arrays
        pop = c.initial_population
        self.rows = self.rng.integers(0, N, size=pop)
        self.cols = self.rng.integers(0, N, size=pop)
        self.energy = np.full(pop, c.energy_initial)
        self.age = np.zeros(pop, dtype=np.int32)
        self.generation = np.zeros(pop, dtype=np.int32)
        self.ids = np.arange(pop, dtype=np.int64)
        self.parent_ids = np.full(pop, -1, dtype=np.int64)
        self.next_id = pop
        
        # Genome: [movement_params(8) | photo_params(4)]
        self.genomes = self.rng.normal(0, 0.5, (pop, c.genome_size))
        
        # ── Stats ──
        self.stats_history = []
    
    # ── Environment update ──
    
    def _update_environment(self):
        c = self.cfg
        
        # Toxic diffusion (average of 4-neighbors, weighted blend)
        k = c.toxic_diffusion_rate
        p = np.pad(self.toxic, 1, mode='edge')
        neighbors = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        self.toxic += k * (neighbors - self.toxic)
        
        # Toxic decay
        self.toxic *= (1.0 - c.toxic_decay_rate)
        
        # Organism toxic byproducts — deposit onto grid
        np.add.at(self.toxic, (self.rows, self.cols), c.toxic_production_rate)
        
        # Nutrient regeneration
        self.nutrients += c.nutrient_base_rate
        
        # Decomposition slowly becomes nutrients
        transfer = self.decomposition * 0.01
        self.nutrients += transfer
        self.decomposition -= transfer
        
        # Clamp
        np.clip(self.toxic, 0, 10.0, out=self.toxic)
        np.clip(self.nutrients, 0, 5.0, out=self.nutrients)
        np.clip(self.decomposition, 0, 10.0, out=self.decomposition)
    
    def _update_density(self):
        """Count organisms per cell."""
        self.density[:] = 0
        np.add.at(self.density, (self.rows, self.cols), 1)
    
    # ── Sensing (vectorized) ──
    
    def _sense_local(self):
        """Return per-organism local readings and gradients."""
        rows, cols = self.rows, self.cols
        N = self.cfg.grid_size
        
        # Local values at organism positions
        local_light = self.light[rows, cols]
        local_toxic = self.toxic[rows, cols]
        local_nutrients = self.nutrients[rows, cols]
        local_density = self.density[rows, cols].astype(np.float64)
        
        # Gradients (clamped at boundaries)
        r_up = np.clip(rows - 1, 0, N - 1)
        r_dn = np.clip(rows + 1, 0, N - 1)
        c_lt = np.clip(cols - 1, 0, N - 1)
        c_rt = np.clip(cols + 1, 0, N - 1)
        
        light_grad_y = self.light[r_up, cols] - self.light[r_dn, cols]
        light_grad_x = self.light[rows, c_rt] - self.light[rows, c_lt]
        toxic_grad_y = self.toxic[r_up, cols] - self.toxic[r_dn, cols]
        toxic_grad_x = self.toxic[rows, c_rt] - self.toxic[rows, c_lt]
        
        # Stack: (n_organisms, 8)
        readings = np.column_stack([
            local_light, local_toxic, local_nutrients, local_density,
            light_grad_y, light_grad_x, toxic_grad_y, toxic_grad_x
        ])
        return readings
    
    # ── Photosynthesis (vectorized) ──
    
    def _photosynthesize(self, readings):
        """Compute energy gained per organism. Density-dependent."""
        c = self.cfg
        photo = self.genomes[:, c.movement_params_size:]  # last 4 params
        
        local_light = readings[:, 0]
        local_toxic = readings[:, 1]
        local_density = readings[:, 3]
        
        # Base efficiency modulated by genome
        efficiency = c.photosynthesis_base * (1.0 + 0.3 * np.tanh(photo[:, 0]))
        
        # Toxic penalty, reduced by tolerance parameter
        tolerance = np.maximum(0.1, 1.0 + 0.5 * np.tanh(photo[:, 1]))
        toxic_penalty = np.maximum(0.0, 1.0 - local_toxic * c.toxic_photo_penalty / tolerance)
        
        # Light sensitivity
        light_exp = np.maximum(0.3, 1.0 - 0.3 * np.tanh(photo[:, 2]))
        light_mult = np.power(local_light, light_exp)
        
        # Storage efficiency
        storage = 0.5 + 0.5 / (1.0 + np.exp(-photo[:, 3]))
        
        # DENSITY COMPETITION: energy shared among organisms in same cell
        # This is the key carrying capacity mechanism
        share = 1.0 / np.maximum(1.0, local_density)
        
        return efficiency * toxic_penalty * light_mult * storage * share
    
    # ── Movement (vectorized) ──
    
    def _decide_movement(self, readings):
        """Return movement directions for all organisms. 0=stay,1=up,2=down,3=right,4=left"""
        c = self.cfg
        n = len(self.rows)
        move_p = self.genomes[:, :c.movement_params_size]  # first 8 params
        
        # Scores for 5 directions: (n, 5)
        scores = np.zeros((n, 5))
        
        light_grad_y = readings[:, 4]
        light_grad_x = readings[:, 5]
        toxic_grad_y = readings[:, 6]
        toxic_grad_x = readings[:, 7]
        local_nutrients = readings[:, 2]
        local_density = readings[:, 3]
        
        # param indices:
        # 0: nutrient attraction
        # 1: density avoidance
        # 2: nutrient stay bonus
        # 3: (unused / future)
        # 4: random movement weight
        # 5: stay tendency
        # 6: light-seeking strength
        # 7: toxic-avoidance strength
        
        # Stay
        scores[:, 0] = move_p[:, 5] + move_p[:, 2] * local_nutrients - move_p[:, 1] * local_density * 0.1
        
        # Up (row - 1)
        scores[:, 1] = move_p[:, 6] * light_grad_y + move_p[:, 7] * (-toxic_grad_y)
        # Down (row + 1)
        scores[:, 2] = -move_p[:, 6] * light_grad_y - move_p[:, 7] * (-toxic_grad_y)
        # Right (col + 1)
        scores[:, 3] = move_p[:, 6] * light_grad_x + move_p[:, 7] * (-toxic_grad_x)
        # Left (col - 1)
        scores[:, 4] = -move_p[:, 6] * light_grad_x - move_p[:, 7] * (-toxic_grad_x)
        
        # Random perturbation
        scores += move_p[:, 4:5] * self.rng.normal(0, 0.3, (n, 5))
        
        return np.argmax(scores, axis=1)
    
    def _execute_movement(self, actions):
        """Apply movement actions."""
        c = self.cfg
        N = c.grid_size
        
        # Movement deltas
        moved = actions > 0
        
        # Up
        mask = actions == 1
        self.rows[mask] = np.maximum(0, self.rows[mask] - 1)
        
        # Down
        mask = actions == 2
        self.rows[mask] = np.minimum(N - 1, self.rows[mask] + 1)
        
        # Right
        mask = actions == 3
        self.cols[mask] = np.minimum(N - 1, self.cols[mask] + 1)
        
        # Left
        mask = actions == 4
        self.cols[mask] = np.maximum(0, self.cols[mask] - 1)
        
        # Movement energy cost
        self.energy[moved] -= c.energy_movement_cost
    
    # ── Reproduction (vectorized) ──
    
    def _reproduce(self):
        """Reproduce eligible organisms. Returns arrays for offspring."""
        c = self.cfg
        
        can_reproduce = (
            (self.energy >= c.energy_reproduction_threshold) &
            (self.age >= c.min_reproduction_age)
        )
        
        # Population cap
        current_pop = len(self.rows)
        n_eligible = can_reproduce.sum()
        if current_pop + n_eligible > c.max_population:
            # Randomly select subset that can reproduce
            eligible_indices = np.where(can_reproduce)[0]
            budget = max(0, c.max_population - current_pop)
            if budget > 0 and len(eligible_indices) > budget:
                chosen = self.rng.choice(eligible_indices, budget, replace=False)
                can_reproduce[:] = False
                can_reproduce[chosen] = True
            elif budget == 0:
                can_reproduce[:] = False
        
        parent_idx = np.where(can_reproduce)[0]
        n_births = len(parent_idx)
        
        if n_births == 0:
            return
        
        # Parents pay cost
        self.energy[parent_idx] -= c.energy_reproduction_cost
        
        # Offspring positions: nearby parent
        offsets_r = self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, n_births)
        offsets_c = self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, n_births)
        child_rows = np.clip(self.rows[parent_idx] + offsets_r, 0, c.grid_size - 1)
        child_cols = np.clip(self.cols[parent_idx] + offsets_c, 0, c.grid_size - 1)
        
        # Offspring genomes: mutated copies
        child_genomes = self.genomes[parent_idx] + self.rng.normal(0, c.mutation_rate, (n_births, c.genome_size))
        
        # Offspring metadata
        child_energy = np.full(n_births, c.energy_initial)
        child_age = np.zeros(n_births, dtype=np.int32)
        child_gen = self.generation[parent_idx] + 1
        child_ids = np.arange(self.next_id, self.next_id + n_births, dtype=np.int64)
        child_parent_ids = self.ids[parent_idx]
        self.next_id += n_births
        
        # Append to population
        self.rows = np.concatenate([self.rows, child_rows])
        self.cols = np.concatenate([self.cols, child_cols])
        self.energy = np.concatenate([self.energy, child_energy])
        self.age = np.concatenate([self.age, child_age])
        self.generation = np.concatenate([self.generation, child_gen])
        self.genomes = np.concatenate([self.genomes, child_genomes])
        self.ids = np.concatenate([self.ids, child_ids])
        self.parent_ids = np.concatenate([self.parent_ids, child_parent_ids])
    
    # ── Death ──
    
    def _kill_and_decompose(self):
        """Remove dead organisms, deposit into decomposition layer."""
        c = self.cfg
        
        alive = (self.energy > 0) & (self.age < c.max_age)
        dead = ~alive
        
        if dead.any():
            # Deposit into decomposition layer
            dead_rows = self.rows[dead]
            dead_cols = self.cols[dead]
            dead_energy = np.maximum(0, self.energy[dead])
            deposits = dead_energy * c.nutrient_from_decomp + 1.0
            np.add.at(self.decomposition, (dead_rows, dead_cols), deposits)
        
        # Filter to alive only
        self.rows = self.rows[alive]
        self.cols = self.cols[alive]
        self.energy = self.energy[alive]
        self.age = self.age[alive]
        self.generation = self.generation[alive]
        self.genomes = self.genomes[alive]
        self.ids = self.ids[alive]
        self.parent_ids = self.parent_ids[alive]
    
    # ── Main update ──
    
    def update(self):
        c = self.cfg
        
        # 1. Update density map
        self._update_density()
        
        # 2. Sense
        readings = self._sense_local()
        
        # 3. Photosynthesis
        energy_gain = self._photosynthesize(readings)
        self.energy = np.minimum(self.energy + energy_gain, c.energy_max)
        
        # 4. Movement
        actions = self._decide_movement(readings)
        self._execute_movement(actions)
        
        # 5. Age and maintenance
        self.age += 1
        self.energy -= c.energy_maintenance_cost
        
        # 6. Reproduction
        self._reproduce()
        
        # 7. Death and decomposition
        self._kill_and_decompose()
        
        # 8. Environment update
        self._update_environment()
        
        self.timestep += 1
        
        # Record stats
        pop = len(self.rows)
        self.stats_history.append({
            "t": self.timestep,
            "pop": pop,
            "avg_energy": float(self.energy.mean()) if pop > 0 else 0,
            "max_gen": int(self.generation.max()) if pop > 0 else 0,
            "toxic_mean": float(self.toxic.mean()),
            "toxic_max": float(self.toxic.max()),
            "decomp_mean": float(self.decomposition.mean()),
            "nutrient_mean": float(self.nutrients.mean()),
        })
    
    # ── Snapshots ──
    
    def snapshot(self):
        pop = len(self.rows)
        
        # Sample organisms for snapshot (cap at 500 to keep file small)
        if pop > 500:
            idx = self.rng.choice(pop, 500, replace=False)
        else:
            idx = np.arange(pop)
        
        organisms = []
        for i in idx:
            organisms.append({
                "id": int(self.ids[i]),
                "row": int(self.rows[i]),
                "col": int(self.cols[i]),
                "energy": round(float(self.energy[i]), 2),
                "age": int(self.age[i]),
                "generation": int(self.generation[i]),
                "genome": self.genomes[i].tolist(),
            })
        
        s = self.stats_history[-1] if self.stats_history else {}
        
        return {
            "timestep": self.timestep,
            "population": pop,
            "sampled": len(organisms),
            "organisms": organisms,
            "environment": {
                "toxic_mean": s.get("toxic_mean", 0),
                "toxic_max": s.get("toxic_max", 0),
                "nutrient_mean": s.get("nutrient_mean", 0),
                "decomp_mean": s.get("decomp_mean", 0),
            },
            "stats": {
                "avg_energy": s.get("avg_energy", 0),
                "max_generation": s.get("max_gen", 0),
            }
        }
    
    def save_snapshot(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        data = self.snapshot()
        path = os.path.join(output_dir, f"snapshot_{self.timestep:06d}.json")
        with open(path, 'w') as f:
            json.dump(data, f)
        return path


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

def run_simulation(cfg=None):
    cfg = cfg or Config()
    
    world = World(cfg)
    
    print(f"The Shimmering Field — Phase 1 v2 (vectorized)")
    print(f"Grid: {cfg.grid_size}x{cfg.grid_size}  |  Initial pop: {cfg.initial_population}  |  Max pop: {cfg.max_population}")
    print(f"Running {cfg.total_timesteps} timesteps...")
    print(f"{'─' * 72}")
    
    start = time.time()
    
    for t in range(cfg.total_timesteps):
        world.update()
        
        if world.timestep % cfg.snapshot_interval == 0:
            world.save_snapshot(cfg.output_dir)
            s = world.stats_history[-1]
            elapsed = time.time() - start
            print(
                f"  t={s['t']:5d}  |  pop={s['pop']:5d}  |  "
                f"energy={s['avg_energy']:6.1f}  |  gen={s['max_gen']:4d}  |  "
                f"toxic={s['toxic_mean']:.4f}  |  decomp={s['decomp_mean']:.4f}  |  "
                f"{elapsed:.1f}s"
            )
        
        if len(world.rows) == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            break
    
    elapsed = time.time() - start
    print(f"{'─' * 72}")
    print(f"Complete in {elapsed:.1f}s  |  Final pop: {len(world.rows)}")
    
    # Save summary
    os.makedirs(cfg.output_dir, exist_ok=True)
    summary = {
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
        "stats_history": world.stats_history,
    }
    path = os.path.join(cfg.output_dir, "run_summary.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {path}")
    print(f"Snapshots: {cfg.output_dir}/snapshot_*.json")
    
    return world


if __name__ == "__main__":
    run_simulation()
