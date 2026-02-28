"""
The Shimmering Field — Phase 2 Step 1 (Refactored)
====================================================
Organism state managed via array registry — adding new per-organism
arrays for Step 2+ requires only registering them, not touching
_reproduce/_kill_and_decompose.

Mechanics: tuned ecology + horizontal transfer from decomposition layer.
"""

import numpy as np
import json
import os
import time
from scipy.ndimage import uniform_filter


class Config:
    grid_size = 128
    light_max = 1.0
    light_min = 0.05
    zone_count = 8

    toxic_decay_rate = 0.01
    toxic_diffusion_rate = 0.06
    toxic_production_rate = 0.015
    toxic_threshold_low = 0.3
    toxic_threshold_medium = 0.8
    toxic_threshold_high = 1.5
    toxic_damage_medium = 1.5
    toxic_damage_high = 5.0
    toxic_photo_penalty = 1.0

    nutrient_base_rate = 0.002
    nutrient_from_decomp = 0.4
    nutrient_max = 3.0

    initial_population = 80
    energy_initial = 40.0
    energy_max = 150.0
    energy_reproduction_threshold = 80.0
    energy_reproduction_cost = 40.0
    energy_maintenance_cost = 0.6
    energy_movement_cost = 0.2
    photosynthesis_base = 3.0

    movement_params_size = 8
    photo_params_size = 4
    transfer_params_size = 2
    genome_size = movement_params_size + photo_params_size + transfer_params_size  # 14

    mutation_rate = 0.08
    min_reproduction_age = 8
    offspring_distance = 5
    max_age = 200
    sensing_range = 3

    transfer_check_interval = 5
    transfer_blend_rate = 0.15
    decomp_fragment_decay = 0.005
    decomp_fragment_diffusion = 0.02

    total_timesteps = 10000
    snapshot_interval = 100
    output_dir = "output_p2s1r"
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

        zone_size = N // c.zone_count
        self.zone_map = np.ones((N, N))
        for zi in range(c.zone_count):
            for zj in range(c.zone_count):
                r0, r1 = zi * zone_size, (zi + 1) * zone_size
                c0, c1 = zj * zone_size, (zj + 1) * zone_size
                self.zone_map[r0:r1, c0:c1] = self.rng.choice(
                    [0.3, 0.5, 0.7, 1.0, 1.0, 1.2, 1.5, 2.0])
        self.zone_map = uniform_filter(self.zone_map, size=8)

        self.fragment_pool = np.zeros((N, N, c.genome_size))
        self.fragment_weight = np.zeros((N, N))

        # ── Organism array registry ──
        # Each entry: (attr_name, dtype, default_value_or_None)
        # default=None means it must be explicitly provided at birth
        self._org_arrays = [
            ("rows",           np.int64,   None),
            ("cols",           np.int64,   None),
            ("energy",         np.float64, None),
            ("age",            np.int32,   0),
            ("generation",     np.int32,   None),
            ("ids",            np.int64,   None),
            ("parent_ids",     np.int64,   None),
            ("genomes",        np.float64, None),   # 2D: (n, genome_size)
            ("transfer_count", np.int32,   0),
        ]

        # ── Initialize population ──
        pop = c.initial_population
        self.rows = self.rng.integers(0, N, size=pop).astype(np.int64)
        self.cols = self.rng.integers(0, N, size=pop).astype(np.int64)
        self.energy = np.full(pop, c.energy_initial)
        self.age = np.zeros(pop, dtype=np.int32)
        self.generation = np.zeros(pop, dtype=np.int32)
        self.ids = np.arange(pop, dtype=np.int64)
        self.parent_ids = np.full(pop, -1, dtype=np.int64)
        self.genomes = self.rng.normal(0, 0.5, (pop, c.genome_size))
        self.transfer_count = np.zeros(pop, dtype=np.int32)
        self.next_id = pop

        self.total_transfers = 0
        self.stats_history = []

    # ══════════════════════════════════════
    # Organism array helpers
    # ══════════════════════════════════════

    @property
    def pop(self):
        return len(self.rows)

    def _filter_organisms(self, mask):
        """Keep only organisms where mask is True."""
        for name, _, _ in self._org_arrays:
            setattr(self, name, getattr(self, name)[mask])

    def _append_organisms(self, new_data: dict):
        """Append new organisms. new_data maps attr_name -> array of values."""
        for name, dtype, default in self._org_arrays:
            existing = getattr(self, name)
            if name in new_data:
                addition = np.asarray(new_data[name], dtype=existing.dtype)
            elif default is not None:
                n = len(next(iter(new_data.values())))
                addition = np.full(n, default, dtype=existing.dtype)
                if existing.ndim > 1:
                    addition = np.zeros((n, *existing.shape[1:]), dtype=existing.dtype) + default
            else:
                raise ValueError(f"Must provide '{name}' in new_data (no default)")
            setattr(self, name, np.concatenate([existing, addition]))

    def _get_dead_data(self, dead_mask):
        """Return dict of arrays for dead organisms."""
        return {name: getattr(self, name)[dead_mask] for name, _, _ in self._org_arrays}

    # ══════════════════════════════════════
    # Environment
    # ══════════════════════════════════════

    def _update_environment(self):
        c = self.cfg
        k = c.toxic_diffusion_rate
        p = np.pad(self.toxic, 1, mode='edge')
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        self.toxic += k * (nb - self.toxic)
        self.toxic *= (1.0 - c.toxic_decay_rate / np.maximum(0.3, self.zone_map))

        if self.pop > 0:
            np.add.at(self.toxic, (self.rows, self.cols),
                      c.toxic_production_rate * self.zone_map[self.rows, self.cols])

        self.nutrients += c.nutrient_base_rate
        xfer = self.decomposition * 0.015
        self.nutrients += xfer
        self.decomposition -= xfer
        self.decomposition *= 0.998

        self.fragment_weight *= (1.0 - c.decomp_fragment_decay)
        if self.timestep % 10 == 0 and self.fragment_weight.max() > 0.001:
            k2 = c.decomp_fragment_diffusion * 10
            pw = np.pad(self.fragment_weight, 1, mode='edge')
            wn = (pw[:-2, 1:-1] + pw[2:, 1:-1] + pw[1:-1, :-2] + pw[1:-1, 2:]) / 4.0
            self.fragment_weight += k2 * (wn - self.fragment_weight)
            self.fragment_weight = np.maximum(self.fragment_weight, 0.0)
            pp = np.pad(self.fragment_pool, ((1, 1), (1, 1), (0, 0)), mode='edge')
            gn = (pp[:-2, 1:-1, :] + pp[2:, 1:-1, :] + pp[1:-1, :-2, :] + pp[1:-1, 2:, :]) / 4.0
            self.fragment_pool += k2 * (gn - self.fragment_pool)

        np.clip(self.toxic, 0, 5.0, out=self.toxic)
        np.clip(self.nutrients, 0, c.nutrient_max, out=self.nutrients)
        np.clip(self.decomposition, 0, 10.0, out=self.decomposition)

    def _update_density(self):
        self.density[:] = 0
        if self.pop > 0:
            np.add.at(self.density, (self.rows, self.cols), 1)

    # ══════════════════════════════════════
    # Sensing
    # ══════════════════════════════════════

    def _sense_local(self):
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

    # ══════════════════════════════════════
    # Photosynthesis, toxic damage, movement
    # ══════════════════════════════════════

    def _photosynthesize(self, readings):
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
        c = self.cfg
        lt = readings[:, 1]
        dmg = np.zeros(self.pop)
        m = lt > c.toxic_threshold_medium
        if m.any(): dmg[m] += (lt[m] - c.toxic_threshold_medium) * c.toxic_damage_medium
        h = lt > c.toxic_threshold_high
        if h.any(): dmg[h] += (lt[h] - c.toxic_threshold_high) * c.toxic_damage_high
        self.energy -= dmg

    def _decide_movement(self, readings):
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
        N = self.cfg.grid_size
        m = actions == 1; self.rows[m] = np.maximum(0, self.rows[m] - 1)
        m = actions == 2; self.rows[m] = np.minimum(N - 1, self.rows[m] + 1)
        m = actions == 3; self.cols[m] = np.minimum(N - 1, self.cols[m] + 1)
        m = actions == 4; self.cols[m] = np.maximum(0, self.cols[m] - 1)
        self.energy[actions > 0] -= self.cfg.energy_movement_cost

    # ══════════════════════════════════════
    # Horizontal transfer
    # ══════════════════════════════════════

    def _horizontal_transfer(self):
        c = self.cfg
        n = self.pop
        if n == 0: return
        tp = self.genomes[:, c.movement_params_size + c.photo_params_size:]
        receptivity = 1.0 / (1.0 + np.exp(-tp[:, 0]))
        selectivity = np.abs(tp[:, 1])
        local_fw = self.fragment_weight[self.rows, self.cols]
        attempts = (local_fw > 0.1) & (self.rng.random(n) < receptivity)
        aidx = np.where(attempts)[0]
        if len(aidx) == 0: return
        local_frags = self.fragment_pool[self.rows[aidx], self.cols[aidx]]
        dists = np.sqrt(np.mean((self.genomes[aidx] - local_frags) ** 2, axis=1))
        thresh = 2.0 / (1.0 + selectivity[aidx])
        tidx = aidx[dists < thresh]
        if len(tidx) == 0: return
        blend = c.transfer_blend_rate
        frags = self.fragment_pool[self.rows[tidx], self.cols[tidx]]
        self.genomes[tidx] = (1.0 - blend) * self.genomes[tidx] + blend * frags
        self.transfer_count[tidx] += 1
        self.total_transfers += len(tidx)

    # ══════════════════════════════════════
    # Reproduction (uses _append_organisms)
    # ══════════════════════════════════════

    def _reproduce(self):
        c = self.cfg
        can = (self.energy >= c.energy_reproduction_threshold) & (self.age >= c.min_reproduction_age)
        pidx = np.where(can)[0]
        nb = len(pidx)
        if nb == 0: return

        self.energy[pidx] -= c.energy_reproduction_cost

        child_ids = np.arange(self.next_id, self.next_id + nb, dtype=np.int64)
        self.next_id += nb

        self._append_organisms({
            "rows": np.clip(self.rows[pidx] + self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, nb), 0, c.grid_size - 1),
            "cols": np.clip(self.cols[pidx] + self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, nb), 0, c.grid_size - 1),
            "energy": np.full(nb, c.energy_initial),
            "generation": self.generation[pidx] + 1,
            "ids": child_ids,
            "parent_ids": self.ids[:self.pop][pidx],  # parent ids from before append
            "genomes": self.genomes[pidx] + self.rng.normal(0, c.mutation_rate, (nb, c.genome_size)),
        })
        # age and transfer_count get their defaults (0) automatically

    # ══════════════════════════════════════
    # Death (uses _filter_organisms)
    # ══════════════════════════════════════

    def _kill_and_decompose(self):
        c = self.cfg
        alive = (self.energy > 0) & (self.age < c.max_age)
        dead = ~alive

        if dead.any():
            dd = self._get_dead_data(dead)
            dr, dc, de, dg = dd["rows"], dd["cols"], np.maximum(0, dd["energy"]), dd["genomes"]
            np.add.at(self.decomposition, (dr, dc), de * c.nutrient_from_decomp + 0.5)

            # Batch fragment deposition grouped by cell
            cell_ids = dr * c.grid_size + dc
            unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
            nc = len(unique_cells)
            genome_sums = np.zeros((nc, c.genome_size))
            counts = np.zeros(nc)
            np.add.at(genome_sums, inverse, dg)
            np.add.at(counts, inverse, 1.0)
            ur = unique_cells // c.grid_size
            uc = unique_cells % c.grid_size
            w0 = self.fragment_weight[ur, uc]
            wt = w0 + counts
            avg_new = genome_sums / counts[:, None]
            blend = counts / wt
            self.fragment_pool[ur, uc] = (
                self.fragment_pool[ur, uc] * (1.0 - blend[:, None]) + avg_new * blend[:, None])
            self.fragment_weight[ur, uc] = wt

        self._filter_organisms(alive)

    # ══════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════

    def update(self):
        if self.pop == 0:
            self._update_environment()
            self.timestep += 1
            self._record_stats()
            return
        self._update_density()
        readings = self._sense_local()
        self.energy = np.minimum(self.energy + self._photosynthesize(readings), self.cfg.energy_max)
        self._apply_toxic_damage(readings)
        self._execute_movement(self._decide_movement(readings))
        self.age += 1
        self.energy -= self.cfg.energy_maintenance_cost
        if self.timestep % self.cfg.transfer_check_interval == 0:
            self._horizontal_transfer()
        self._reproduce()
        self._kill_and_decompose()
        self._update_environment()
        self.timestep += 1
        self._record_stats()

    # ══════════════════════════════════════
    # Stats & snapshots
    # ══════════════════════════════════════

    def _record_stats(self):
        p = self.pop
        c = self.cfg
        if p > 0:
            lt = self.toxic[self.rows, self.cols]
            in_low = int(np.sum(lt < c.toxic_threshold_low))
            in_med = int(np.sum((lt >= c.toxic_threshold_low) & (lt < c.toxic_threshold_medium)))
            in_high = int(np.sum(lt >= c.toxic_threshold_medium))
            xferd = int(np.sum(self.transfer_count > 0))
            avg_tc = float(self.transfer_count.mean())
        else:
            in_low = in_med = in_high = xferd = 0; avg_tc = 0.0
        self.stats_history.append({
            "t": self.timestep, "pop": p,
            "avg_energy": float(self.energy.mean()) if p > 0 else 0,
            "max_gen": int(self.generation.max()) if p > 0 else 0,
            "toxic_mean": float(self.toxic.mean()),
            "toxic_max": float(self.toxic.max()),
            "decomp_mean": float(self.decomposition.mean()),
            "frag_mean": float(self.fragment_weight.mean()),
            "frag_max": float(self.fragment_weight.max()),
            "in_low_toxic": in_low, "in_med_toxic": in_med, "in_high_toxic": in_high,
            "orgs_with_transfers": xferd,
            "avg_transfer_count": round(avg_tc, 2),
            "total_transfers": self.total_transfers,
        })

    def save_snapshot(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        p = self.pop
        idx = self.rng.choice(p, min(p, 500), replace=False) if p > 500 else np.arange(p)
        orgs = [{"id": int(self.ids[i]), "row": int(self.rows[i]), "col": int(self.cols[i]),
                 "energy": round(float(self.energy[i]), 2), "age": int(self.age[i]),
                 "generation": int(self.generation[i]), "genome": self.genomes[i].tolist(),
                 "transfer_count": int(self.transfer_count[i])} for i in idx]
        s = self.stats_history[-1] if self.stats_history else {}
        with open(os.path.join(output_dir, f"snapshot_{self.timestep:06d}.json"), 'w') as f:
            json.dump({"timestep": self.timestep, "population": p, "organisms": orgs, "stats": s}, f)

    def save_env(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(os.path.join(output_dir, f"env_{self.timestep:06d}.npz"),
            toxic=self.toxic.astype(np.float32), decomposition=self.decomposition.astype(np.float32),
            density=self.density.astype(np.int16), fragment_weight=self.fragment_weight.astype(np.float32))


def run_simulation(cfg=None):
    cfg = cfg or Config()
    world = World(cfg)
    print(f"The Shimmering Field — Phase 2 Step 1 (Refactored)")
    print(f"Grid: {cfg.grid_size}x{cfg.grid_size}  |  Pop: {cfg.initial_population}  |  Genome: {cfg.genome_size}")
    print(f"{'─' * 95}")
    start = time.time()
    for t in range(cfg.total_timesteps):
        world.update()
        if world.timestep % cfg.snapshot_interval == 0:
            world.save_snapshot(cfg.output_dir)
            s = world.stats_history[-1]
            el = time.time() - start
            print(f"  t={s['t']:5d}  |  pop={s['pop']:5d}  |  e={s['avg_energy']:5.1f}  |  "
                  f"gen={s['max_gen']:4d}  |  tox={s['toxic_mean']:.3f}  |  "
                  f"frag={s['frag_mean']:.2f}/{s['frag_max']:.1f}  |  "
                  f"xfer={s['orgs_with_transfers']:4d} cum={s['total_transfers']:6d}  |  {el:.1f}s")
        if world.timestep % 500 == 0:
            world.save_env(cfg.output_dir)
        if world.pop == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            break
    el = time.time() - start
    print(f"{'─' * 95}")
    print(f"Done in {el:.1f}s  |  Pop: {world.pop}  |  Transfers: {world.total_transfers}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_summary.json"), 'w') as f:
        json.dump({"config": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                    "stats_history": world.stats_history}, f, indent=2)
    return world

if __name__ == "__main__":
    run_simulation()
