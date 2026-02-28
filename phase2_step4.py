"""
The Shimmering Field — Phase 2 Step 4: Viral Archaeology
==========================================================
Adds stratified temporal layers to the decomposition/fragment system.

New mechanics:
  - Three strata: recent, intermediate, ancient
  - Dead organisms deposit genome fragments into RECENT stratum only
  - Material sediments over time: recent → intermediate → ancient
  - Access gated by local toxic concentration (design doc thresholds):
      Low toxic  (<0.3): recent stratum only
      Medium     (0.3-0.8): recent + intermediate
      High       (>0.8): recent + intermediate + ancient
  - Deeper strata = more divergent material, higher blend rate (bigger jumps)
  - Ancient genomes are "fossils" from organisms that lived under different conditions

Built on Step 2 (lytic/lysogenic viral system).
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

    # Genome layout: movement(8) + photo(4) + transfer(2) + viral(2) = 16
    movement_params_size = 8
    photo_params_size = 4
    transfer_params_size = 2
    viral_params_size = 2       # [viral_resistance, lysogenic_suppression]
    genome_size = movement_params_size + photo_params_size + transfer_params_size + viral_params_size

    mutation_rate = 0.08
    min_reproduction_age = 8
    offspring_distance = 5
    max_age = 200
    sensing_range = 3

    # Horizontal transfer
    transfer_check_interval = 5
    transfer_blend_rate_recent = 0.10       # blend rate from recent stratum
    transfer_blend_rate_intermediate = 0.18 # bigger jumps from intermediate
    transfer_blend_rate_ancient = 0.30      # biggest jumps from ancient (fossil DNA)
    decomp_fragment_decay = 0.005
    decomp_fragment_diffusion = 0.02

    # Stratified archaeology (NEW)
    sedimentation_rate_recent = 0.005       # fraction of recent → intermediate per step
    sedimentation_rate_intermediate = 0.002 # fraction of intermediate → ancient per step
    ancient_decay_rate = 0.001              # ancient material decays very slowly
    # Toxic thresholds for stratum access (from design doc)
    stratum_access_medium = 0.3             # toxic > this = can access intermediate
    stratum_access_high = 0.8               # toxic > this = can access ancient

    # ── Viral system (NEW) ──
    viral_decay_rate = 0.01            # free particles decay per step (was 0.03)
    viral_diffusion_rate = 0.08       # particles spread to neighbors
    viral_infection_rate = 0.3        # probability of infection per step (was 0.15)
    viral_lytic_damage = 2.0          # energy drain per step from lytic infection
    viral_lytic_growth = 0.1          # viral_load increases per step when infected
    viral_burst_threshold = 1.0       # viral_load at which organism bursts
    viral_burst_amount = 8.0          # particles released on lytic death (was 3.0)
    viral_burst_radius = 3            # how far burst particles spread
    lysogenic_probability = 0.4       # fraction of new infections that go lysogenic
    lysogenic_activation_toxic = 0.6  # toxic level above which lysogenic activates
    lysogenic_blend_rate = 0.1        # how much lysogenic material affects expressed genome
    lysogenic_inheritance = 0.8       # fraction of lysogenic material passed to offspring
    viral_check_interval = 3          # run viral dynamics every N steps

    total_timesteps = 10000
    snapshot_interval = 100
    output_dir = "output_p2s4"
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

        # Stratified genome fragment pools (replaces single fragment_pool)
        # Each stratum: genome pool (N, N, genome_size) + weight (N, N)
        self.strata_pool = {
            "recent":       np.zeros((N, N, c.genome_size)),
            "intermediate": np.zeros((N, N, c.genome_size)),
            "ancient":      np.zeros((N, N, c.genome_size)),
        }
        self.strata_weight = {
            "recent":       np.zeros((N, N)),
            "intermediate": np.zeros((N, N)),
            "ancient":      np.zeros((N, N)),
        }

        # NEW: Free viral particles on grid — seeded with initial hotspots
        self.viral_particles = np.zeros((N, N))
        # Seed 5-10 viral hotspots to bootstrap the system
        n_seeds = 8
        for _ in range(n_seeds):
            sr = self.rng.integers(0, N)
            sc_ = self.rng.integers(0, N)
            radius = 5
            r0, r1 = max(0, sr - radius), min(N, sr + radius + 1)
            c0_, c1_ = max(0, sc_ - radius), min(N, sc_ + radius + 1)
            self.viral_particles[r0:r1, c0_:c1_] += self.rng.uniform(0.5, 2.0)
        self.viral_genome_pool = np.zeros((N, N, c.genome_size))
        self.viral_genome_weight = np.zeros((N, N))
        # Seed viral genome pool at hotspot locations with random genomes
        self.viral_genome_pool[self.viral_particles > 0.1] = self.rng.normal(0, 0.5, 
            (int((self.viral_particles > 0.1).sum()), c.genome_size))
        self.viral_genome_weight[self.viral_particles > 0.1] = 1.0

        # ── Organism arrays ──
        self._org_arrays = [
            ("rows",              np.int64,   None),
            ("cols",              np.int64,   None),
            ("energy",            np.float64, None),
            ("age",               np.int32,   0),
            ("generation",        np.int32,   None),
            ("ids",               np.int64,   None),
            ("parent_ids",        np.int64,   None),
            ("genomes",           np.float64, None),      # 2D
            ("transfer_count",    np.int32,   0),
            # NEW: viral state
            ("viral_load",        np.float64, 0.0),       # lytic infection intensity
            ("lysogenic_strength",np.float64, None),      # how much lysogenic material present
            ("lysogenic_genome",  np.float64, None),      # 2D: blended genome of integrated material
        ]

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
        self.viral_load = np.zeros(pop, dtype=np.float64)
        self.lysogenic_strength = np.zeros(pop, dtype=np.float64)
        self.lysogenic_genome = np.zeros((pop, c.genome_size), dtype=np.float64)
        self.next_id = pop

        self.total_transfers = 0
        self.transfers_by_stratum = {"recent": 0, "intermediate": 0, "ancient": 0}
        self.total_lytic_deaths = 0
        self.total_lysogenic_integrations = 0
        self.total_lysogenic_activations = 0
        self.stats_history = []

    # ── Organism array helpers ──

    @property
    def pop(self):
        return len(self.rows)

    def _filter_organisms(self, mask):
        for name, _, _ in self._org_arrays:
            setattr(self, name, getattr(self, name)[mask])

    def _append_organisms(self, new_data: dict):
        for name, dtype, default in self._org_arrays:
            existing = getattr(self, name)
            if name in new_data:
                addition = np.asarray(new_data[name], dtype=existing.dtype)
            elif default is not None:
                n = len(next(iter(new_data.values())))
                if existing.ndim > 1:
                    addition = np.full((n, *existing.shape[1:]), default, dtype=existing.dtype)
                else:
                    addition = np.full(n, default, dtype=existing.dtype)
            else:
                raise ValueError(f"Must provide '{name}' in new_data (no default)")
            setattr(self, name, np.concatenate([existing, addition]))

    def _get_dead_data(self, dead_mask):
        return {name: getattr(self, name)[dead_mask] for name, _, _ in self._org_arrays}

    # ── Environment ──

    def _update_environment(self):
        c = self.cfg
        # Toxic diffusion + decay
        k = c.toxic_diffusion_rate
        p = np.pad(self.toxic, 1, mode='edge')
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        self.toxic += k * (nb - self.toxic)
        self.toxic *= (1.0 - c.toxic_decay_rate / np.maximum(0.3, self.zone_map))
        if self.pop > 0:
            np.add.at(self.toxic, (self.rows, self.cols),
                      c.toxic_production_rate * self.zone_map[self.rows, self.cols])

        # Nutrients
        self.nutrients += c.nutrient_base_rate
        xfer = self.decomposition * 0.015
        self.nutrients += xfer
        self.decomposition -= xfer
        self.decomposition *= 0.998

        # Fragment pool
        # ── Stratified sedimentation and diffusion ──
        # Recent → intermediate
        sed_r = c.sedimentation_rate_recent
        xfer_w = self.strata_weight["recent"] * sed_r
        self.strata_weight["recent"] -= xfer_w
        old_iw = self.strata_weight["intermediate"]
        new_iw = old_iw + xfer_w
        safe_iw = np.maximum(new_iw, 1e-8)
        blend_i = xfer_w / safe_iw
        self.strata_pool["intermediate"] = (
            self.strata_pool["intermediate"] * (1.0 - blend_i[:, :, None])
            + self.strata_pool["recent"] * blend_i[:, :, None])
        self.strata_weight["intermediate"] = new_iw

        # Intermediate → ancient
        sed_i = c.sedimentation_rate_intermediate
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

        # Decay per stratum (recent fastest, ancient slowest)
        self.strata_weight["recent"] *= (1.0 - c.decomp_fragment_decay)
        self.strata_weight["intermediate"] *= (1.0 - c.decomp_fragment_decay * 0.5)
        self.strata_weight["ancient"] *= (1.0 - c.ancient_decay_rate)

        # Diffuse all strata periodically
        if self.timestep % 10 == 0:
            k2 = c.decomp_fragment_diffusion * 10
            for sname in ("recent", "intermediate", "ancient"):
                sw = self.strata_weight[sname]
                if sw.max() < 0.001:
                    continue
                pw = np.pad(sw, 1, mode='edge')
                wn = (pw[:-2, 1:-1] + pw[2:, 1:-1] + pw[1:-1, :-2] + pw[1:-1, 2:]) / 4.0
                self.strata_weight[sname] += k2 * (wn - sw)
                self.strata_weight[sname] = np.maximum(self.strata_weight[sname], 0.0)
                sp = self.strata_pool[sname]
                pp = np.pad(sp, ((1, 1), (1, 1), (0, 0)), mode='edge')
                gn = (pp[:-2, 1:-1, :] + pp[2:, 1:-1, :] + pp[1:-1, :-2, :] + pp[1:-1, 2:, :]) / 4.0
                self.strata_pool[sname] += k2 * (gn - sp)

        # NEW: Viral particle diffusion + decay
        self.viral_particles *= (1.0 - c.viral_decay_rate)
        kv = c.viral_diffusion_rate
        pv = np.pad(self.viral_particles, 1, mode='edge')
        vn = (pv[:-2, 1:-1] + pv[2:, 1:-1] + pv[1:-1, :-2] + pv[1:-1, 2:]) / 4.0
        self.viral_particles += kv * (vn - self.viral_particles)
        self.viral_particles = np.maximum(self.viral_particles, 0.0)
        # Viral genome pool follows particles
        if self.timestep % 10 == 0 and self.viral_genome_weight.max() > 0.001:
            self.viral_genome_weight *= (1.0 - c.viral_decay_rate * 10)
            self.viral_genome_weight = np.maximum(self.viral_genome_weight, 0.0)
            pp2 = np.pad(self.viral_genome_pool, ((1, 1), (1, 1), (0, 0)), mode='edge')
            gn2 = (pp2[:-2, 1:-1, :] + pp2[2:, 1:-1, :] + pp2[1:-1, :-2, :] + pp2[1:-1, 2:, :]) / 4.0
            self.viral_genome_pool += kv * (gn2 - self.viral_genome_pool)

        # Clamp
        np.clip(self.toxic, 0, 5.0, out=self.toxic)
        np.clip(self.nutrients, 0, c.nutrient_max, out=self.nutrients)
        np.clip(self.decomposition, 0, 10.0, out=self.decomposition)
        np.clip(self.viral_particles, 0, 20.0, out=self.viral_particles)

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
            self.light[rows, cols], self.toxic[rows, cols],
            self.nutrients[rows, cols], self.density[rows, cols].astype(np.float64),
            self.light[ru, cols] - self.light[rd, cols],
            self.light[rows, cr] - self.light[rows, cl],
            self.toxic[ru, cols] - self.toxic[rd, cols],
            self.toxic[rows, cr] - self.toxic[rows, cl],
        ])

    # ── Photosynthesis, toxic damage, movement ──

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

    # ── Horizontal transfer ──

    def _horizontal_transfer(self):
        """Horizontal transfer from stratified decomp layers, gated by local toxic."""
        c = self.cfg
        n = self.pop
        if n == 0: return
        tp = self.genomes[:, c.movement_params_size + c.photo_params_size:
                          c.movement_params_size + c.photo_params_size + c.transfer_params_size]
        receptivity = 1.0 / (1.0 + np.exp(-tp[:, 0]))
        selectivity = np.abs(tp[:, 1])
        local_toxic = self.toxic[self.rows, self.cols]

        # Determine which strata each organism can access based on local toxic
        can_recent = np.ones(n, dtype=bool)  # always
        can_intermediate = local_toxic >= c.stratum_access_medium
        can_ancient = local_toxic >= c.stratum_access_high

        # Process each stratum
        strata_config = [
            ("recent",       can_recent,       c.transfer_blend_rate_recent),
            ("intermediate", can_intermediate,  c.transfer_blend_rate_intermediate),
            ("ancient",      can_ancient,       c.transfer_blend_rate_ancient),
        ]

        for sname, access_mask, blend_rate in strata_config:
            sw = self.strata_weight[sname]
            local_sw = sw[self.rows, self.cols]

            # Must have access, local material, and pass receptivity roll
            eligible = access_mask & (local_sw > 0.1) & (self.rng.random(n) < receptivity)
            eidx = np.where(eligible)[0]
            if len(eidx) == 0:
                continue

            local_frags = self.strata_pool[sname][self.rows[eidx], self.cols[eidx]]
            dists = np.sqrt(np.mean((self.genomes[eidx] - local_frags) ** 2, axis=1))
            # Deeper strata have relaxed selectivity (accept more divergent material)
            depth_factor = {"recent": 1.0, "intermediate": 1.5, "ancient": 2.5}[sname]
            thresh = (2.0 * depth_factor) / (1.0 + selectivity[eidx])
            tidx = eidx[dists < thresh]
            if len(tidx) == 0:
                continue

            frags = self.strata_pool[sname][self.rows[tidx], self.cols[tidx]]
            self.genomes[tidx] = (1.0 - blend_rate) * self.genomes[tidx] + blend_rate * frags
            self.transfer_count[tidx] += 1
            self.total_transfers += len(tidx)
            self.transfers_by_stratum[sname] += len(tidx)

    # ══════════════════════════════════════
    # NEW: Viral dynamics
    # ══════════════════════════════════════

    def _viral_dynamics(self):
        """Lytic infection, lysogenic activation, and lytic damage."""
        c = self.cfg
        n = self.pop
        if n == 0: return

        # Viral genome params: resistance and suppression
        vp_start = c.movement_params_size + c.photo_params_size + c.transfer_params_size
        viral_resistance = 1.0 / (1.0 + np.exp(-self.genomes[:, vp_start]))       # 0-1
        lysogenic_suppression = 1.0 / (1.0 + np.exp(-self.genomes[:, vp_start + 1]))  # 0-1

        local_viral = self.viral_particles[self.rows, self.cols]
        local_toxic = self.toxic[self.rows, self.cols]

        # ── 1. New infections from free viral particles ──
        uninfected = self.viral_load == 0
        exposure = local_viral > 0.05
        candidates = uninfected & exposure

        if candidates.any():
            cidx = np.where(candidates)[0]
            # Infection probability: higher particles + lower resistance = more likely
            inf_prob = c.viral_infection_rate * np.minimum(local_viral[cidx], 5.0) / 5.0
            inf_prob *= (1.0 - viral_resistance[cidx] * 0.8)  # resistance reduces but doesn't eliminate
            rolls = self.rng.random(len(cidx))
            infected = cidx[rolls < inf_prob]

            if len(infected) > 0:
                # Split into lytic vs lysogenic
                lyso_rolls = self.rng.random(len(infected))
                goes_lytic = infected[lyso_rolls >= c.lysogenic_probability]
                goes_lysogenic = infected[lyso_rolls < c.lysogenic_probability]

                # Lytic: set viral_load to start infection
                if len(goes_lytic) > 0:
                    self.viral_load[goes_lytic] = 0.1

                # Lysogenic: integrate viral genome material silently
                if len(goes_lysogenic) > 0:
                    vg_weight = self.viral_genome_weight[self.rows[goes_lysogenic], self.cols[goes_lysogenic]]
                    has_material = vg_weight > 0.01
                    lyso_with_material = goes_lysogenic[has_material]
                    if len(lyso_with_material) > 0:
                        viral_genomes = self.viral_genome_pool[
                            self.rows[lyso_with_material], self.cols[lyso_with_material]]
                        # Blend into lysogenic storage
                        old_str = self.lysogenic_strength[lyso_with_material]
                        new_str = old_str + 0.3
                        blend = 0.3 / np.maximum(new_str, 0.01)
                        self.lysogenic_genome[lyso_with_material] = (
                            self.lysogenic_genome[lyso_with_material] * (1.0 - blend[:, None])
                            + viral_genomes * blend[:, None])
                        self.lysogenic_strength[lyso_with_material] = new_str
                        self.total_lysogenic_integrations += len(lyso_with_material)

        # ── 2. Lytic infection progression ──
        lytic = self.viral_load > 0
        if lytic.any():
            lidx = np.where(lytic)[0]
            self.viral_load[lidx] += c.viral_lytic_growth
            self.energy[lidx] -= c.viral_lytic_damage * self.viral_load[lidx]

        # ── 3. Lysogenic activation under toxic stress ──
        has_lysogenic = self.lysogenic_strength > 0.01
        stressed = local_toxic > c.lysogenic_activation_toxic
        not_lytic = self.viral_load == 0
        activation_candidates = has_lysogenic & stressed & not_lytic

        if activation_candidates.any():
            acidx = np.where(activation_candidates)[0]
            # Activation probability: higher toxic + lower suppression = more likely
            stress_level = (local_toxic[acidx] - c.lysogenic_activation_toxic) / c.lysogenic_activation_toxic
            act_prob = np.minimum(stress_level * 0.3, 0.5) * (1.0 - lysogenic_suppression[acidx] * 0.7)
            act_rolls = self.rng.random(len(acidx))
            activated = acidx[act_rolls < act_prob]

            if len(activated) > 0:
                # Lysogenic → lytic: blend lysogenic genome into expressed genome, start lytic
                blend = c.lysogenic_blend_rate * self.lysogenic_strength[activated]
                blend = np.minimum(blend, 0.4)  # cap blending
                self.genomes[activated] = (
                    self.genomes[activated] * (1.0 - blend[:, None])
                    + self.lysogenic_genome[activated] * blend[:, None])
                self.viral_load[activated] = 0.2  # start lytic from activation
                self.lysogenic_strength[activated] *= 0.3  # partially consumed
                self.total_lysogenic_activations += len(activated)

    # ── Reproduction ──

    def _reproduce(self):
        c = self.cfg
        # Lytically infected organisms cannot reproduce
        can = ((self.energy >= c.energy_reproduction_threshold) &
               (self.age >= c.min_reproduction_age) &
               (self.viral_load == 0))
        pidx = np.where(can)[0]
        nb = len(pidx)
        if nb == 0: return

        self.energy[pidx] -= c.energy_reproduction_cost
        child_ids = np.arange(self.next_id, self.next_id + nb, dtype=np.int64)
        self.next_id += nb

        # Lysogenic material passes to offspring (vertical transfer)
        child_lyso_genome = self.lysogenic_genome[pidx] * c.lysogenic_inheritance
        child_lyso_strength = self.lysogenic_strength[pidx] * c.lysogenic_inheritance

        self._append_organisms({
            "rows": np.clip(self.rows[pidx] + self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, nb), 0, c.grid_size - 1),
            "cols": np.clip(self.cols[pidx] + self.rng.integers(-c.offspring_distance, c.offspring_distance + 1, nb), 0, c.grid_size - 1),
            "energy": np.full(nb, c.energy_initial),
            "generation": self.generation[pidx] + 1,
            "ids": child_ids,
            "parent_ids": self.ids[:self.pop][pidx],
            "genomes": self.genomes[pidx] + self.rng.normal(0, c.mutation_rate, (nb, c.genome_size)),
            "lysogenic_genome": child_lyso_genome,
            "lysogenic_strength": child_lyso_strength,
        })

    # ── Death ──

    def _kill_and_decompose(self):
        c = self.cfg
        N = c.grid_size

        # Organisms burst when viral_load exceeds threshold
        bursting = self.viral_load >= c.viral_burst_threshold
        natural_death = (self.energy <= 0) | (self.age >= c.max_age)
        dead = bursting | natural_death

        if dead.any():
            dd = self._get_dead_data(dead)
            dr, dc = dd["rows"], dd["cols"]
            de = np.maximum(0, dd["energy"])
            dg = dd["genomes"]

            # Standard decomposition
            np.add.at(self.decomposition, (dr, dc), de * c.nutrient_from_decomp + 0.5)

            # Batch fragment deposition into RECENT stratum
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

            # NEW: Lytic burst — release viral particles carrying host genome
            burst_mask = dd["viral_load"] >= c.viral_burst_threshold
            if burst_mask.any():
                br = dd["rows"][burst_mask]
                bc = dd["cols"][burst_mask]
                bg = dd["genomes"][burst_mask]
                n_burst = len(br)
                self.total_lytic_deaths += n_burst

                # Spread particles in radius around burst location
                for i in range(n_burst):
                    r0 = max(0, br[i] - c.viral_burst_radius)
                    r1 = min(N, br[i] + c.viral_burst_radius + 1)
                    c0 = max(0, bc[i] - c.viral_burst_radius)
                    c1 = min(N, bc[i] + c.viral_burst_radius + 1)
                    area = (r1 - r0) * (c1 - c0)
                    self.viral_particles[r0:r1, c0:c1] += c.viral_burst_amount / area

                    # Deposit genome into viral genome pool
                    w_old = self.viral_genome_weight[r0:r1, c0:c1]
                    w_add = c.viral_burst_amount / area
                    w_new = w_old + w_add
                    blend_v = w_add / np.maximum(w_new, 1e-8)
                    self.viral_genome_pool[r0:r1, c0:c1] = (
                        self.viral_genome_pool[r0:r1, c0:c1] * (1.0 - blend_v[:, :, None])
                        + bg[i][None, None, :] * blend_v[:, :, None])
                    self.viral_genome_weight[r0:r1, c0:c1] = w_new

            # Spontaneous viral release: small fraction of natural deaths emit particles
            # This keeps the viral system alive even without active lytic cycles
            natural_only = (~burst_mask) if burst_mask.any() else np.ones(len(dr), dtype=bool)
            if natural_only.any():
                spont_roll = self.rng.random(int(natural_only.sum()))
                spont_mask_local = spont_roll < 0.15  # 15% of natural deaths
                if spont_mask_local.any():
                    nat_idx = np.where(natural_only)[0][spont_mask_local]
                    np.add.at(self.viral_particles,
                              (dd["rows"][nat_idx], dd["cols"][nat_idx]), 1.0)
                    np.add.at(self.viral_genome_weight,
                              (dd["rows"][nat_idx], dd["cols"][nat_idx]), 1.0)

        self._filter_organisms(~dead)

    # ── Main loop ──

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
        if self.timestep % self.cfg.viral_check_interval == 0:
            self._viral_dynamics()

        self._reproduce()
        self._kill_and_decompose()
        self._update_environment()
        self.timestep += 1
        self._record_stats()

    # ── Stats ──

    def _record_stats(self):
        p = self.pop
        c = self.cfg
        if p > 0:
            lt = self.toxic[self.rows, self.cols]
            in_low = int(np.sum(lt < c.toxic_threshold_low))
            in_med = int(np.sum((lt >= c.toxic_threshold_low) & (lt < c.toxic_threshold_medium)))
            in_high = int(np.sum(lt >= c.toxic_threshold_medium))
            xferd = int(np.sum(self.transfer_count > 0))
            n_lytic = int(np.sum(self.viral_load > 0))
            n_lysogenic = int(np.sum(self.lysogenic_strength > 0.01))
        else:
            in_low = in_med = in_high = xferd = n_lytic = n_lysogenic = 0
        self.stats_history.append({
            "t": self.timestep, "pop": p,
            "avg_energy": float(self.energy.mean()) if p > 0 else 0,
            "max_gen": int(self.generation.max()) if p > 0 else 0,
            "toxic_mean": float(self.toxic.mean()),
            "toxic_max": float(self.toxic.max()),
            "decomp_mean": float(self.decomposition.mean()),
            "strata_recent": float(self.strata_weight["recent"].mean()),
            "strata_intermediate": float(self.strata_weight["intermediate"].mean()),
            "strata_ancient": float(self.strata_weight["ancient"].mean()),
            "viral_mean": float(self.viral_particles.mean()),
            "viral_max": float(self.viral_particles.max()),
            "in_low_toxic": in_low, "in_med_toxic": in_med, "in_high_toxic": in_high,
            "orgs_with_transfers": xferd,
            "total_transfers": self.total_transfers,
            "xfer_recent": self.transfers_by_stratum["recent"],
            "xfer_intermediate": self.transfers_by_stratum["intermediate"],
            "xfer_ancient": self.transfers_by_stratum["ancient"],
            "n_lytic": n_lytic,
            "n_lysogenic": n_lysogenic,
            "total_lytic_deaths": self.total_lytic_deaths,
            "total_lyso_integrations": self.total_lysogenic_integrations,
            "total_lyso_activations": self.total_lysogenic_activations,
        })

    def save_snapshot(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        p = self.pop
        idx = self.rng.choice(p, min(p, 500), replace=False) if p > 500 else np.arange(p)
        orgs = [{"id": int(self.ids[i]), "row": int(self.rows[i]), "col": int(self.cols[i]),
                 "energy": round(float(self.energy[i]), 2), "age": int(self.age[i]),
                 "generation": int(self.generation[i]),
                 "viral_load": round(float(self.viral_load[i]), 3),
                 "lysogenic_strength": round(float(self.lysogenic_strength[i]), 3),
                 "transfer_count": int(self.transfer_count[i])} for i in idx]
        s = self.stats_history[-1] if self.stats_history else {}
        with open(os.path.join(output_dir, f"snapshot_{self.timestep:06d}.json"), 'w') as f:
            json.dump({"timestep": self.timestep, "population": p, "organisms": orgs, "stats": s}, f)

    def save_env(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(os.path.join(output_dir, f"env_{self.timestep:06d}.npz"),
            toxic=self.toxic.astype(np.float32), decomposition=self.decomposition.astype(np.float32),
            density=self.density.astype(np.int16),
            strata_recent_w=self.strata_weight["recent"].astype(np.float32),
            strata_intermediate_w=self.strata_weight["intermediate"].astype(np.float32),
            strata_ancient_w=self.strata_weight["ancient"].astype(np.float32),
            viral_particles=self.viral_particles.astype(np.float32))


def run_simulation(cfg=None):
    cfg = cfg or Config()
    world = World(cfg)
    print(f"The Shimmering Field — Phase 2 Step 4: Viral Archaeology")
    print(f"Grid: {cfg.grid_size}x{cfg.grid_size}  |  Pop: {cfg.initial_population}  |  Genome: {cfg.genome_size}")
    print(f"Strata access: recent=always  intermediate=tox>{cfg.stratum_access_medium}  ancient=tox>{cfg.stratum_access_high}")
    print(f"{'─' * 115}")
    start = time.time()
    for t in range(cfg.total_timesteps):
        world.update()
        if world.timestep % cfg.snapshot_interval == 0:
            world.save_snapshot(cfg.output_dir)
            s = world.stats_history[-1]
            el = time.time() - start
            print(f"  t={s['t']:5d}  |  pop={s['pop']:5d}  |  e={s['avg_energy']:5.1f}  |  "
                  f"gen={s['max_gen']:4d}  |  tox={s['toxic_mean']:.3f}  |  "
                  f"R={s['strata_recent']:.2f} I={s['strata_intermediate']:.2f} A={s['strata_ancient']:.2f}  |  "
                  f"xR={s['xfer_recent']:5d} xI={s['xfer_intermediate']:5d} xA={s['xfer_ancient']:5d}  |  "
                  f"lyt={s['n_lytic']:3d} lys={s['n_lysogenic']:4d}  |  {el:.1f}s")
        if world.timestep % 500 == 0:
            world.save_env(cfg.output_dir)
        if world.pop == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            break
    el = time.time() - start
    print(f"{'─' * 115}")
    tbs = world.transfers_by_stratum
    print(f"Done in {el:.1f}s  |  Pop: {world.pop}  |  Lytic deaths: {world.total_lytic_deaths}")
    print(f"Transfers — Recent: {tbs['recent']}  Intermediate: {tbs['intermediate']}  Ancient: {tbs['ancient']}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_summary.json"), 'w') as f:
        json.dump({"config": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                    "stats_history": world.stats_history}, f, indent=2)
    return world

if __name__ == "__main__":
    run_simulation()
