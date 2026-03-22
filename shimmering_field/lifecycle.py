"""Death, decomposition, organism initialization."""

import numpy as np

from .constants import *


class LifecycleMixin:
    """Death, decomposition, organism initialization."""

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

        # Default genome: PHOTO + TOXPROD (sessile producers — plants)
        self.module_present = np.zeros((pop, N_MODULES), dtype=bool)
        self.module_present[:, M_PHOTO] = True
        self.module_present[:, M_TOXPROD] = True

        # Mobile producers: only 10% start with MOVE (rare mobile plants)
        n_mobile_prod = int(pop * (1.0 - c.initial_sessile_fraction))
        if n_mobile_prod > 0:
            mobile_idx = self.rng.choice(pop, n_mobile_prod, replace=False)
            self.module_present[mobile_idx, M_MOVE] = True

        self.module_active = self.module_present.copy()

        self.weights = self.rng.normal(0, 0.5, (pop, TOTAL_WEIGHT_PARAMS))

        # Seed carnivores: CONSUME+MOVE+TOXPROD, no PHOTO, high aggression, high prey_selectivity
        # Only 10% of consumers — apex predators are RARE
        n_carn = int(pop * c.initial_consumer_fraction * 0.20)
        carn_idx = self.rng.choice(pop, n_carn, replace=False)
        self.module_present[carn_idx, M_PHOTO] = False
        self.module_present[carn_idx, M_CONSUME] = True
        self.module_present[carn_idx, M_MOVE] = True  # animals MUST move
        self.module_active[carn_idx] = self.module_present[carn_idx]
        # Spread across grid (not just center — they need to find prey)
        self.rows[carn_idx] = self.rng.integers(0, N, n_carn).astype(np.int64)
        self.cols[carn_idx] = self.rng.integers(0, N, n_carn).astype(np.int64)
        self.energy[carn_idx] = c.energy_initial * 3.0
        # Seed CONSUME weights: high aggression, low decomp_pref, HIGH prey_selectivity (carnivore)
        consume_off = int(MODULE_WEIGHT_OFFSETS[M_CONSUME])
        self.weights[carn_idx, consume_off + 3] = self.rng.uniform(1.0, 2.5, n_carn)   # aggression
        self.weights[carn_idx, consume_off + 2] = self.rng.uniform(-3.0, -1.5, n_carn)  # decomp_pref (low)
        self.weights[carn_idx, consume_off + 0] = self.rng.uniform(1.5, 4.0, n_carn)    # prey_selectivity (strongly high = carnivore)

        # Seed herbivores: CONSUME+MOVE+TOXPROD, no PHOTO, LOW prey_selectivity
        used_idx = set(carn_idx.tolist())
        remaining = np.array([i for i in range(pop) if i not in used_idx])
        n_herb = int(pop * c.initial_consumer_fraction * 0.55)  # 65% of consumers are herbivores
        herb_idx = self.rng.choice(remaining, n_herb, replace=False)
        self.module_present[herb_idx, M_PHOTO] = False
        self.module_present[herb_idx, M_CONSUME] = True
        self.module_present[herb_idx, M_MOVE] = True
        self.module_active[herb_idx] = self.module_present[herb_idx]
        self.rows[herb_idx] = self.rng.integers(0, N, n_herb).astype(np.int64)
        self.cols[herb_idx] = self.rng.integers(0, N, n_herb).astype(np.int64)
        self.energy[herb_idx] = c.energy_initial * 3.0
        self.weights[herb_idx, consume_off + 3] = self.rng.uniform(0.5, 1.5, n_herb)     # aggression (moderate)
        self.weights[herb_idx, consume_off + 2] = self.rng.uniform(-2.0, -0.5, n_herb)   # decomp_pref (low)
        self.weights[herb_idx, consume_off + 0] = self.rng.uniform(-4.0, -1.5, n_herb)   # prey_selectivity (strongly low = herbivore)

        # Seed detritivores: CONSUME+MOVE+DEFENSE+TOXPROD, no PHOTO, high decomp_pref
        # Spread across the ENTIRE grid — they follow death, not clusters
        used_idx.update(herb_idx.tolist())
        remaining = np.array([i for i in range(pop) if i not in used_idx])
        n_detr = int(pop * c.initial_detritivore_fraction)
        detr_idx = self.rng.choice(remaining, n_detr, replace=False)
        self.module_present[detr_idx, M_PHOTO] = False
        self.module_present[detr_idx, M_CONSUME] = True
        self.module_present[detr_idx, M_MOVE] = True  # animals MUST move
        self.module_present[detr_idx, M_DEFENSE] = True  # armored scavengers
        self.module_active[detr_idx] = self.module_present[detr_idx]
        self.rows[detr_idx] = self.rng.integers(0, N, n_detr).astype(np.int64)
        self.cols[detr_idx] = self.rng.integers(0, N, n_detr).astype(np.int64)
        self.energy[detr_idx] = c.energy_initial * 3.0
        self.weights[detr_idx, consume_off + 2] = self.rng.uniform(1.5, 3.0, n_detr)   # decomp_pref (high)
        self.weights[detr_idx, consume_off + 3] = self.rng.uniform(-2.0, -0.5, n_detr)  # aggression (low)
        # Seed DEFENSE weights for detritivores — shell focus (armored)
        defense_off = int(MODULE_WEIGHT_OFFSETS[M_DEFENSE])
        self.weights[detr_idx, defense_off + 0] = self.rng.uniform(1.0, 2.5, n_detr)  # shell (high)

        # Seed decomp broadly — death happens everywhere
        self.decomposition += self.rng.uniform(0.0, 0.3, (N, N))
        # Hot spots near center
        self.decomposition[mid-15:mid+15, mid-15:mid+15] += self.rng.uniform(0.5, 2.0, (30, 30))
        # Scatter death patches across grid
        for _ in range(20):
            dr, dc = self.rng.integers(5, N-5), self.rng.integers(5, N-5)
            self.decomposition[dr-3:dr+3, dc-3:dc+3] += self.rng.uniform(1.0, 4.0, (6, 6))

        # Seed omnivores: CONSUME+MOVE+TOXPROD — animals that eat both plants and animals
        # MID-RANGE prey_selectivity: not specialists, but versatile
        used_idx.update(detr_idx.tolist())
        remaining = np.array([i for i in range(pop) if i not in used_idx])
        n_omni = int(pop * c.initial_omnivore_fraction)
        if n_omni > 0 and len(remaining) >= n_omni:
            omni_idx = self.rng.choice(remaining, n_omni, replace=False)
            self.module_present[omni_idx, M_PHOTO] = False
            self.module_present[omni_idx, M_CONSUME] = True
            self.module_present[omni_idx, M_MOVE] = True
            self.module_active[omni_idx] = self.module_present[omni_idx]
            self.rows[omni_idx] = self.rng.integers(0, N, n_omni).astype(np.int64)
            self.cols[omni_idx] = self.rng.integers(0, N, n_omni).astype(np.int64)
            self.energy[omni_idx] = c.energy_initial * 2.5
            # Mid-range prey selectivity: 0.35-0.65 = omnivore band
            self.weights[omni_idx, consume_off + 0] = self.rng.uniform(-0.15, 0.15, n_omni)  # maps to ~0.46-0.54
            self.weights[omni_idx, consume_off + 3] = self.rng.uniform(0.5, 1.5, n_omni)    # moderate aggression
            self.weights[omni_idx, consume_off + 2] = self.rng.uniform(-2.0, -0.5, n_omni)  # low decomp_pref

        # Viral/HGT state
        self.transfer_count = np.zeros(pop, dtype=np.int32)
        self.viral_load = np.zeros(pop, dtype=np.float64)
        self.lysogenic_strength = np.zeros(pop, dtype=np.float64)
        self.lysogenic_genome = np.zeros((pop, TOTAL_WEIGHT_PARAMS), dtype=np.float64)

        # VRESIST: immune memory from surviving infections
        self.immune_experience = np.zeros(pop, dtype=np.float64)
        # SOCIAL: relationship accumulation with compatible neighbors
        self.relationship_score = np.zeros(pop, dtype=np.float64)
        # Endosymbiosis: merger history
        self.merger_count = np.zeros(pop, dtype=np.int32)
        # Capacity shedding: per-module usage tracking (starts at 1.0 for present modules)
        self.module_usage = np.zeros((pop, N_MODULES), dtype=np.float64)
        self.module_usage[self.module_present] = 1.0
        # Genomic incompatibility: stress from accumulated HGT
        self.genomic_stress = np.zeros(pop, dtype=np.float64)
        self.genomic_cascade_phase = np.zeros(pop, dtype=np.int32)  # 0=none, 1/2/3=cascade phases
        # Developmental dependency
        self.dev_progress = np.zeros(pop, dtype=np.float64)
        self.is_mature = np.zeros(pop, dtype=bool)
        # Initial population starts mature (founding organisms)
        self.is_mature[:] = True
        self.dev_progress[:] = 1.0

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

    # ── Capacity Shedding ──

