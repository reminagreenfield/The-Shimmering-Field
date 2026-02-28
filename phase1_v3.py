"""
The Shimmering Field — Phase 1 v3: Tuned Ecology
==================================================
Key changes from v2:
  - Toxic accumulation now damages organisms directly (energy drain + death)
  - Three toxic thresholds per design doc: low/medium/high
  - Hard population cap removed — carrying capacity is ecological
  - Zone-based chemical landscape: some zones absorb toxic, others amplify
  - Faster generational turnover (shorter lifespan, lower reproduction threshold)
  - Nutrient competition is local and finite — photosynthesis depletes local light capture
"""

import numpy as np
import json
import os
import time


class Config:
    # Grid
    grid_size = 256
    
    # Light gradient
    light_max = 1.0
    light_min = 0.05
    
    # ── Zone-based chemical landscape ──
    # Grid divided into zones with different toxic absorption/amplification
    zone_count = 8  # 8x8 zones across grid
    
    # ── Toxicity — now the central driver ──
    toxic_decay_rate = 0.01           # slower decay — toxic persists
    toxic_diffusion_rate = 0.06       # spreads meaningfully
    toxic_production_rate = 0.015     # metabolic byproduct per organism
    
    # Three thresholds (per design doc)
    toxic_threshold_low = 0.3         # below this: normal ecology
    toxic_threshold_medium = 0.8      # above this: organisms take damage
    toxic_threshold_high = 1.5        # above this: rapid death, system crisis
    
    # Toxic damage
    toxic_damage_medium = 1.5         # energy drain per step above medium threshold
    toxic_damage_high = 5.0           # energy drain per step above high threshold
    toxic_photo_penalty = 1.0         # photosynthesis efficiency reduction per unit toxic
    
    # ── Nutrients ──
    nutrient_base_rate = 0.002
    nutrient_from_decomp = 0.4
    nutrient_max = 3.0
    
    # ── Organisms ──
    initial_population = 100
    energy_initial = 40.0
    energy_max = 150.0
    energy_reproduction_threshold = 80.0
    energy_reproduction_cost = 40.0
    energy_maintenance_cost = 0.6
    energy_movement_cost = 0.2
    photosynthesis_base = 3.0
    
    # Genome
    movement_params_size = 8
    photo_params_size = 4
    genome_size = movement_params_size + photo_params_size
    
    # ── Faster generations ──
    mutation_rate = 0.08
    min_reproduction_age = 8
    offspring_distance = 5
    max_age = 200                     # shorter lifespan = faster turnover
    
    # Sensing
    sensing_range = 3
    
    # Simulation
    total_timesteps = 10000
    snapshot_interval = 100
    output_dir = "output_v3"
    random_seed = 42


class World:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.rng = np.random.default_rng(c.random_seed)
        self.timestep = 0
        self.next_id = 0
        N = c.grid_size
        
        # ── Environment ──
        self.light = np.linspace(c.light_max, c.light_min, N)[:, None] * np.ones((1, N))
        self.toxic = self.rng.uniform(0.0, 0.005, (N, N))
        self.nutrients = self.rng.uniform(0.02, 0.08, (N, N))
        self.decomposition = np.zeros((N, N))
        self.density = np.zeros((N, N), dtype=np.int32)
        
        # ── Zone map: toxic absorption/amplification multiplier ──
        # Values < 1 = zone absorbs toxic (refugia)
        # Values > 1 = zone amplifies toxic (danger zones)
        # Creates spatial heterogeneity per design doc
        zone_size = N // c.zone_count
        self.zone_map = np.ones((N, N))
        for zi in range(c.zone_count):
            for zj in range(c.zone_count):
                r0, r1 = zi * zone_size, (zi + 1) * zone_size
                c0, c1 = zj * zone_size, (zj + 1) * zone_size
                # Mix of absorbing and amplifying zones
                val = self.rng.choice([0.3, 0.5, 0.7, 1.0, 1.0, 1.2, 1.5, 2.0])
                self.zone_map[r0:r1, c0:c1] = val
        
        # Smooth zone boundaries slightly
        from scipy.ndimage import uniform_filter
        self.zone_map = uniform_filter(self.zone_map, size=8)
        
        # ── Organisms ──
        pop = c.initial_population
        self.rows = self.rng.integers(0, N, size=pop)
        self.cols = self.rng.integers(0, N, size=pop)
        self.energy = np.full(pop, c.energy_initial)
        self.age = np.zeros(pop, dtype=np.int32)
        self.generation = np.zeros(pop, dtype=np.int32)
        self.ids = np.arange(pop, dtype=np.int64)
        self.parent_ids = np.full(pop, -1, dtype=np.int64)
        self.next_id = pop
        self.genomes = self.rng.normal(0, 0.5, (pop, c.genome_size))
        
        # ── Stats ──
        self.stats_history = []
    
    # ── Environment ──
    
    def _update_environment(self):
        c = self.cfg
        
        # Toxic diffusion
        k = c.toxic_diffusion_rate
        p = np.pad(self.toxic, 1, mode='edge')
        neighbors = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        self.toxic += k * (neighbors - self.toxic)
        
        # Toxic decay — modulated by zone map
        # Absorbing zones (low multiplier) decay faster
        # Amplifying zones (high multiplier) decay slower
        effective_decay = c.toxic_decay_rate / np.maximum(0.3, self.zone_map)
        self.toxic *= (1.0 - effective_decay)
        
        # Organism toxic byproducts — amplified by zone map
        if len(self.rows) > 0:
            base_output = np.full(len(self.rows), c.toxic_production_rate)
            zone_mult = self.zone_map[self.rows, self.cols]
            np.add.at(self.toxic, (self.rows, self.cols), base_output * zone_mult)
        
        # Nutrient regeneration
        self.nutrients += c.nutrient_base_rate
        
        # Decomposition → nutrients
        transfer = self.decomposition * 0.015
        self.nutrients += transfer
        self.decomposition -= transfer
        self.decomposition *= 0.998  # slow baseline decay
        
        # Clamp
        np.clip(self.toxic, 0, 5.0, out=self.toxic)
        np.clip(self.nutrients, 0, c.nutrient_max, out=self.nutrients)
        np.clip(self.decomposition, 0, 10.0, out=self.decomposition)
    
    def _update_density(self):
        self.density[:] = 0
        if len(self.rows) > 0:
            np.add.at(self.density, (self.rows, self.cols), 1)
    
    # ── Sensing ──
    
    def _sense_local(self):
        rows, cols = self.rows, self.cols
        N = self.cfg.grid_size
        
        local_light = self.light[rows, cols]
        local_toxic = self.toxic[rows, cols]
        local_nutrients = self.nutrients[rows, cols]
        local_density = self.density[rows, cols].astype(np.float64)
        
        r_up = np.clip(rows - 1, 0, N - 1)
        r_dn = np.clip(rows + 1, 0, N - 1)
        c_lt = np.clip(cols - 1, 0, N - 1)
        c_rt = np.clip(cols + 1, 0, N - 1)
        
        light_grad_y = self.light[r_up, cols] - self.light[r_dn, cols]
        light_grad_x = self.light[rows, c_rt] - self.light[rows, c_lt]
        toxic_grad_y = self.toxic[r_up, cols] - self.toxic[r_dn, cols]
        toxic_grad_x = self.toxic[rows, c_rt] - self.toxic[rows, c_lt]
        
        return np.column_stack([
            local_light, local_toxic, local_nutrients, local_density,
            light_grad_y, light_grad_x, toxic_grad_y, toxic_grad_x
        ])
    
    # ── Photosynthesis ──
    
    def _photosynthesize(self, readings):
        c = self.cfg
        photo = self.genomes[:, c.movement_params_size:]
        
        local_light = readings[:, 0]
        local_toxic = readings[:, 1]
        local_density = readings[:, 3]
        
        # Base efficiency
        efficiency = c.photosynthesis_base * (1.0 + 0.3 * np.tanh(photo[:, 0]))
        
        # Toxic penalty — real reduction
        tolerance = np.maximum(0.1, 1.0 + 0.5 * np.tanh(photo[:, 1]))
        toxic_penalty = np.maximum(0.0, 1.0 - local_toxic * c.toxic_photo_penalty / tolerance)
        
        # Light sensitivity
        light_exp = np.maximum(0.3, 1.0 - 0.3 * np.tanh(photo[:, 2]))
        light_mult = np.power(np.maximum(local_light, 0.01), light_exp)
        
        # Storage
        storage = 0.5 + 0.5 / (1.0 + np.exp(-photo[:, 3]))
        
        # Density competition — shared resources
        share = 1.0 / np.maximum(1.0, local_density * 0.5)
        
        return efficiency * toxic_penalty * light_mult * storage * share
    
    # ── Toxic damage — the key new mechanic ──
    
    def _apply_toxic_damage(self, readings):
        """Direct energy damage from toxic exposure. This is what creates carrying capacity."""
        c = self.cfg
        local_toxic = readings[:, 1]
        
        damage = np.zeros(len(self.rows))
        
        # Medium threshold: organism takes damage
        medium_mask = local_toxic > c.toxic_threshold_medium
        if medium_mask.any():
            excess = local_toxic[medium_mask] - c.toxic_threshold_medium
            damage[medium_mask] += excess * c.toxic_damage_medium
        
        # High threshold: severe damage
        high_mask = local_toxic > c.toxic_threshold_high
        if high_mask.any():
            excess = local_toxic[high_mask] - c.toxic_threshold_high
            damage[high_mask] += excess * c.toxic_damage_high
        
        self.energy -= damage
    
    # ── Movement ──
    
    def _decide_movement(self, readings):
        c = self.cfg
        n = len(self.rows)
        if n == 0:
            return np.array([], dtype=np.int32)
        
        move_p = self.genomes[:, :c.movement_params_size]
        scores = np.zeros((n, 5))
        
        local_nutrients = readings[:, 2]
        local_density = readings[:, 3]
        light_grad_y = readings[:, 4]
        light_grad_x = readings[:, 5]
        toxic_grad_y = readings[:, 6]
        toxic_grad_x = readings[:, 7]
        
        # Stay: influenced by local conditions
        scores[:, 0] = (move_p[:, 5] 
                        + move_p[:, 2] * local_nutrients 
                        - move_p[:, 1] * np.minimum(local_density, 10) * 0.1)
        
        # Up (row - 1): toward more light, away from toxic
        scores[:, 1] = move_p[:, 6] * light_grad_y - move_p[:, 7] * toxic_grad_y
        # Down
        scores[:, 2] = -move_p[:, 6] * light_grad_y + move_p[:, 7] * toxic_grad_y
        # Right
        scores[:, 3] = move_p[:, 6] * light_grad_x - move_p[:, 7] * toxic_grad_x
        # Left
        scores[:, 4] = -move_p[:, 6] * light_grad_x + move_p[:, 7] * toxic_grad_x
        
        # Noise
        scores += move_p[:, 4:5] * self.rng.normal(0, 0.3, (n, 5))
        
        return np.argmax(scores, axis=1)
    
    def _execute_movement(self, actions):
        c = self.cfg
        N = c.grid_size
        
        mask = actions == 1
        self.rows[mask] = np.maximum(0, self.rows[mask] - 1)
        mask = actions == 2
        self.rows[mask] = np.minimum(N - 1, self.rows[mask] + 1)
        mask = actions == 3
        self.cols[mask] = np.minimum(N - 1, self.cols[mask] + 1)
        mask = actions == 4
        self.cols[mask] = np.maximum(0, self.cols[mask] - 1)
        
        moved = actions > 0
        self.energy[moved] -= c.energy_movement_cost
    
    # ── Reproduction ──
    
    def _reproduce(self):
        c = self.cfg
        
        can_reproduce = (
            (self.energy >= c.energy_reproduction_threshold) &
            (self.age >= c.min_reproduction_age)
        )
        
        parent_idx = np.where(can_reproduce)[0]
        n_births = len(parent_idx)
        if n_births == 0:
            return
        
        self.energy[parent_idx] -= c.energy_reproduction_cost
        
        offsets_r = self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, n_births)
        offsets_c = self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, n_births)
        child_rows = np.clip(self.rows[parent_idx] + offsets_r, 0, c.grid_size - 1)
        child_cols = np.clip(self.cols[parent_idx] + offsets_c, 0, c.grid_size - 1)
        
        child_genomes = (self.genomes[parent_idx] 
                         + self.rng.normal(0, c.mutation_rate, (n_births, c.genome_size)))
        
        child_energy = np.full(n_births, c.energy_initial)
        child_age = np.zeros(n_births, dtype=np.int32)
        child_gen = self.generation[parent_idx] + 1
        child_ids = np.arange(self.next_id, self.next_id + n_births, dtype=np.int64)
        child_parent_ids = self.ids[parent_idx]
        self.next_id += n_births
        
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
        c = self.cfg
        alive = (self.energy > 0) & (self.age < c.max_age)
        dead = ~alive
        
        if dead.any():
            dead_rows = self.rows[dead]
            dead_cols = self.cols[dead]
            dead_energy = np.maximum(0, self.energy[dead])
            deposits = dead_energy * c.nutrient_from_decomp + 0.5
            np.add.at(self.decomposition, (dead_rows, dead_cols), deposits)
        
        self.rows = self.rows[alive]
        self.cols = self.cols[alive]
        self.energy = self.energy[alive]
        self.age = self.age[alive]
        self.generation = self.generation[alive]
        self.genomes = self.genomes[alive]
        self.ids = self.ids[alive]
        self.parent_ids = self.parent_ids[alive]
    
    # ── Main loop ──
    
    def update(self):
        if len(self.rows) == 0:
            self._update_environment()
            self.timestep += 1
            self._record_stats()
            return
        
        # 1. Density
        self._update_density()
        
        # 2. Sense
        readings = self._sense_local()
        
        # 3. Photosynthesis
        energy_gain = self._photosynthesize(readings)
        self.energy = np.minimum(self.energy + energy_gain, self.cfg.energy_max)
        
        # 4. TOXIC DAMAGE — the key mechanic
        self._apply_toxic_damage(readings)
        
        # 5. Movement
        actions = self._decide_movement(readings)
        self._execute_movement(actions)
        
        # 6. Age + maintenance
        self.age += 1
        self.energy -= self.cfg.energy_maintenance_cost
        
        # 7. Reproduction
        self._reproduce()
        
        # 8. Death
        self._kill_and_decompose()
        
        # 9. Environment
        self._update_environment()
        
        self.timestep += 1
        self._record_stats()
    
    def _record_stats(self):
        pop = len(self.rows)
        
        # Count organisms in different toxic zones
        if pop > 0:
            local_toxic = self.toxic[self.rows, self.cols]
            in_low = int(np.sum(local_toxic < self.cfg.toxic_threshold_low))
            in_med = int(np.sum((local_toxic >= self.cfg.toxic_threshold_low) & 
                               (local_toxic < self.cfg.toxic_threshold_medium)))
            in_high = int(np.sum(local_toxic >= self.cfg.toxic_threshold_medium))
        else:
            in_low = in_med = in_high = 0
        
        self.stats_history.append({
            "t": self.timestep,
            "pop": pop,
            "avg_energy": float(self.energy.mean()) if pop > 0 else 0,
            "min_energy": float(self.energy.min()) if pop > 0 else 0,
            "max_gen": int(self.generation.max()) if pop > 0 else 0,
            "avg_age": float(self.age.mean()) if pop > 0 else 0,
            "toxic_mean": float(self.toxic.mean()),
            "toxic_max": float(self.toxic.max()),
            "toxic_p95": float(np.percentile(self.toxic, 95)),
            "decomp_mean": float(self.decomposition.mean()),
            "nutrient_mean": float(self.nutrients.mean()),
            "in_low_toxic": in_low,
            "in_med_toxic": in_med,
            "in_high_toxic": in_high,
        })
    
    # ── Snapshots ──
    
    def snapshot(self):
        pop = len(self.rows)
        
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
                "toxic_p95": s.get("toxic_p95", 0),
                "nutrient_mean": s.get("nutrient_mean", 0),
                "decomp_mean": s.get("decomp_mean", 0),
            },
            "stats": s,
        }
    
    def save_snapshot(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        data = self.snapshot()
        path = os.path.join(output_dir, f"snapshot_{self.timestep:06d}.json")
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def save_env_snapshot(self, output_dir):
        """Save full environment grids for visualization."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"env_{self.timestep:06d}.npz")
        np.savez_compressed(path,
            toxic=self.toxic.astype(np.float32),
            decomposition=self.decomposition.astype(np.float32),
            nutrients=self.nutrients.astype(np.float32),
            density=self.density.astype(np.int16),
            zone_map=self.zone_map.astype(np.float32),
        )


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

def run_simulation(cfg=None):
    cfg = cfg or Config()
    world = World(cfg)
    
    print(f"The Shimmering Field — Phase 1 v3 (tuned)")
    print(f"Grid: {cfg.grid_size}x{cfg.grid_size}  |  Pop: {cfg.initial_population}")
    print(f"Toxic thresholds: low={cfg.toxic_threshold_low} med={cfg.toxic_threshold_medium} high={cfg.toxic_threshold_high}")
    print(f"Running {cfg.total_timesteps} timesteps...")
    print(f"{'─' * 80}")
    
    start = time.time()
    env_save_interval = 500  # save environment grids less often
    
    for t in range(cfg.total_timesteps):
        world.update()
        
        if world.timestep % cfg.snapshot_interval == 0:
            world.save_snapshot(cfg.output_dir)
            s = world.stats_history[-1]
            elapsed = time.time() - start
            
            toxic_zone = f"safe={s['in_low_toxic']:4d} med={s['in_med_toxic']:4d} high={s['in_high_toxic']:4d}"
            print(
                f"  t={s['t']:5d}  |  pop={s['pop']:5d}  |  "
                f"e={s['avg_energy']:5.1f}  |  gen={s['max_gen']:4d}  |  "
                f"tox={s['toxic_mean']:.3f}/{s['toxic_max']:.2f}  |  "
                f"{toxic_zone}  |  {elapsed:.1f}s"
            )
        
        if world.timestep % env_save_interval == 0:
            world.save_env_snapshot(cfg.output_dir)
        
        if len(world.rows) == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            # Keep running environment for a bit to see decomposition
            for _ in range(200):
                world._update_environment()
                world.timestep += 1
                world._record_stats()
            break
    
    elapsed = time.time() - start
    print(f"{'─' * 80}")
    print(f"Complete in {elapsed:.1f}s  |  Final pop: {len(world.rows)}")
    
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


if __name__ == "__main__":
    run_simulation()
