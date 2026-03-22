"""Reproduction, mutation, and reproductive manipulation."""

import numpy as np

from .constants import *


class ReproductionMixin:
    """Reproduction, mutation, and reproductive manipulation."""

    def _reproduce(self):
        c = self.cfg
        N = c.grid_size

        # Global carrying capacity: reproduction gets harder as population approaches cap
        # Only affects producers — consumers are already limited by prey availability
        pop_ratio = self.pop / c.carrying_capacity
        pop_pressure = max(0.0, (pop_ratio - 0.75) * 4.0)
        global_penalty = 1.0 + pop_pressure * c.carrying_capacity_penalty

        # Density-dependent threshold: harder to reproduce in crowded areas
        local_dens = self.density[self.rows, self.cols].astype(np.float64)
        effective_threshold = c.energy_reproduction_threshold + local_dens * c.repro_density_penalty

        # Carrying capacity pressures producers heavily, consumers lightly
        # Producers cause overcrowding; consumers are already limited by prey scarcity
        has_consume = self.module_present[:, M_CONSUME]
        cap_mult = np.where(has_consume, 1.0 + pop_pressure * c.carrying_capacity_penalty * 0.4,
                                          global_penalty)
        effective_threshold *= cap_mult

        # Mediator bonus: pollination service lowers reproduction threshold
        local_med = self.mediator_field[self.rows, self.cols]
        mediator_bonus = np.minimum(local_med, 3.0) * c.mediate_repro_bonus
        effective_threshold -= mediator_bonus

        # Collapse zone penalty: sigmoid reproduction suppression
        collapse_mod = self._collapse_reproduction_modifier()
        # Convert modifier (0-1) to threshold increase: low modifier = high threshold
        effective_threshold /= np.maximum(collapse_mod, 0.1)

        # Detritivores are r-selected: reproduce at lower threshold
        troph = self._trophic_roles()
        is_detritivore = (troph == 3)
        if is_detritivore.any():
            effective_threshold[is_detritivore] *= c.detritivore_repro_bonus

        can = ((self.energy >= effective_threshold) &
               (self.age >= c.min_reproduction_age) &
               (self.viral_load == 0))
        pidx = np.where(can)[0]
        nb = len(pidx)
        if nb == 0:
            return

        self.energy[pidx] -= c.energy_reproduction_cost

        # Reward mediators near reproduction events
        if nb > 0:
            has_mediate = self.module_active[:, M_MEDIATE]
            if has_mediate.any():
                mw = self._module_weights(M_MEDIATE)
                reward_sens = 1.0 / (1.0 + np.exp(-mw[:, 3]))  # reward_sensitivity
                local_repro_signal = np.zeros(self.pop)
                # Count reproductions at each mediator's location
                repro_cells = set(zip(self.rows[pidx].tolist(), self.cols[pidx].tolist()))
                for mi in np.where(has_mediate)[0]:
                    r, col = self.rows[mi], self.cols[mi]
                    # Check within mediate_radius
                    nearby_repros = sum(1 for rr, cc in repro_cells
                                       if abs(rr - r) <= c.mediate_radius
                                       and abs(cc - col) <= c.mediate_radius)
                    if nearby_repros > 0:
                        self.energy[mi] += (c.mediate_energy_reward * reward_sens[mi]
                                          * min(nearby_repros, 5))
        child_ids = np.arange(self.next_id, self.next_id + nb, dtype=np.int64)
        self.next_id += nb

        child_modules = self.module_present[pidx].copy()

        # Module gain
        gain_rolls = self.rng.random(nb)
        for i in np.where(gain_rolls < c.module_gain_rate)[0]:
            absent = [gm for gm in GAINABLE_MODULES if not child_modules[i, gm]]
            if absent:
                chosen = self.rng.choice(absent)
                # Sessile-mobile divergence: producers almost never evolve MOVE
                if chosen == M_MOVE and child_modules[i, M_PHOTO] and not child_modules[i, M_CONSUME]:
                    if self.rng.random() > c.move_gain_producer_mult:
                        # Reject this MOVE gain for a producer — pick something else
                        alt = [gm for gm in absent if gm != M_MOVE]
                        chosen = self.rng.choice(alt) if alt else chosen
                child_modules[i, chosen] = True

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
                # Sessile-mobile divergence: consumers almost never lose MOVE
                if m == M_MOVE and child_modules[i, M_CONSUME] and not child_modules[i, M_PHOTO]:
                    if self.rng.random() > c.move_lose_consumer_mult:
                        continue  # Skip — consumers keep MOVE
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

        child_weights = self.weights[pidx] + self.rng.normal(0, c.mutation_rate, (nb, TOTAL_WEIGHT_PARAMS))

        # ── Reproductive Manipulation (Wolbachia-style) ──
        # Lysogenic viral material in parents silently biases offspring
        parent_lyso = self.lysogenic_strength[pidx]
        manipulated = parent_lyso > c.repro_manip_threshold
        n_manipulated = int(manipulated.sum())

        if n_manipulated > 0:
            midx = np.where(manipulated)[0]
            parent_lyso_genome = self.lysogenic_genome[pidx[midx]]
            manip_strength = np.minimum(parent_lyso[midx], 2.0) / 2.0  # normalize to 0-1

            # Self-limiting: manipulation weakens at high population penetration
            pop_lyso_frac = (self.lysogenic_strength > c.repro_manip_threshold).sum() / max(self.pop, 1)
            if pop_lyso_frac > c.repro_manip_saturation:
                saturation_penalty = (pop_lyso_frac - c.repro_manip_saturation) / (1.0 - c.repro_manip_saturation)
                manip_strength *= (1.0 - saturation_penalty * 0.8)

            # 1. Trait bias: blend offspring weights toward lysogenic genome
            trait_blend = c.repro_manip_trait_bias * manip_strength
            child_weights[midx] = (child_weights[midx] * (1.0 - trait_blend[:, None])
                                  + parent_lyso_genome * trait_blend[:, None])

            # 2. Receptivity boost: make offspring more susceptible to HGT
            recept_idx = STANDALONE_OFFSET + SP_TRANSFER_RECEPTIVITY
            child_weights[midx, recept_idx] += c.repro_manip_receptivity_boost * manip_strength

            # 3. Viability filter: penalize offspring that diverge from lysogenic template
            divergence = np.sqrt(np.mean((child_weights[midx] - parent_lyso_genome) ** 2, axis=1))
            penalty_mask = divergence > c.repro_manip_divergence_thresh
            if penalty_mask.any():
                pen_idx = midx[penalty_mask]
                pen_strength = manip_strength[penalty_mask]
                energy_penalty = c.repro_manip_viability_cost * pen_strength * \
                    (divergence[penalty_mask] - c.repro_manip_divergence_thresh)
                # Store penalty to apply to offspring energy below
                # (offspring energy set to energy_initial, so we reduce it)
                child_energy = np.full(nb, c.energy_initial)
                child_energy[pen_idx] -= energy_penalty
                np.clip(child_energy, 5.0, c.energy_initial, out=child_energy)
            else:
                child_energy = np.full(nb, c.energy_initial)

            self.total_manipulated_births += n_manipulated
        else:
            child_energy = np.full(nb, c.energy_initial)

        self._append_organisms({
            "rows": np.clip(self.rows[pidx] + row_off, 0, N - 1),
            "cols": np.clip(self.cols[pidx] + col_off, 0, N - 1),
            "energy": child_energy,
            "age": np.zeros(nb, dtype=np.int32),
            "generation": self.generation[pidx] + 1,
            "ids": child_ids,
            "parent_ids": self.ids[pidx],
            "weights": child_weights,
            "module_present": child_modules,
            "module_active": child_modules.copy(),
            "transfer_count": np.zeros(nb, dtype=np.int32),
            "viral_load": np.zeros(nb, dtype=np.float64),
            "lysogenic_strength": self.lysogenic_strength[pidx] * c.lysogenic_inheritance,
            "lysogenic_genome": self.lysogenic_genome[pidx] * c.lysogenic_inheritance,
            "immune_experience": self.immune_experience[pidx] * 0.3,  # partial maternal immunity
            "relationship_score": np.zeros(nb, dtype=np.float64),     # fresh start
            "merger_count": self.merger_count[pidx].copy(),           # inherit merger history
            "module_usage": np.ones((nb, N_MODULES), dtype=np.float64) * child_modules,  # full usage for present modules
            "genomic_stress": self.genomic_stress[pidx] * 0.3,  # inherit partial stress (epigenetic memory)
            "genomic_cascade_phase": np.zeros(nb, dtype=np.int32),
            "dev_progress": np.zeros(nb, dtype=np.float64),
            "is_mature": np.zeros(nb, dtype=bool),
        })

    # ── Death and Decomposition ──

