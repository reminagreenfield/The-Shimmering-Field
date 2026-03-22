"""Predation with hard trophic rules and sessile vulnerability."""

import numpy as np

from .constants import *


class PredationMixin:
    """Predation with hard trophic rules and sessile vulnerability."""

    def _predation(self):
        """Predation with hard trophic rules:
          Herbivore -> can ONLY eat producers
          Carnivore -> can ONLY eat herbivores, carnivores, omnivores (NOT producers)
          Omnivore  -> can eat anything except detritivores
          Detritivore -> does NOT hunt (eats decomp)
        Sessile prey is more vulnerable (can't flee)."""
        c = self.cfg
        n = self.pop
        if n < 2:
            return 0

        N = c.grid_size
        has_consume = self.module_active[:, M_CONSUME]
        has_move = self.module_active[:, M_MOVE]
        has_defense = self.module_active[:, M_DEFENSE]
        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]

        # Compute trophic roles for all organisms
        troph = self._trophic_roles()
        # 0=producer, 1=herbivore, 2=carnivore, 3=detritivore, 4=omnivore

        # Predators: herbivores, carnivores, omnivores (NOT detritivores, NOT producers)
        predator_mask = (troph == 1) | (troph == 2) | (troph == 4)
        if not predator_mask.any():
            return 0

        uw = self._module_weights(M_CONSUME)
        aggression = 1.0 / (1.0 + np.exp(-uw[:, 3]))
        decomp_pref = 1.0 / (1.0 + np.exp(-uw[:, 2]))

        # Precompute DEFENSE stats
        shell_val = np.zeros(n)
        camo_val = np.zeros(n)
        counter_val = np.zeros(n)
        if has_defense.any():
            dw = self._module_weights(M_DEFENSE)
            shell_val = c.defense_shell_max * (1.0 / (1.0 + np.exp(-dw[:, 0]))) * has_defense
            camo_val = c.defense_camouflage_max * (1.0 / (1.0 + np.exp(-dw[:, 1]))) * has_defense
            counter_val = c.defense_counter_damage * np.tanh(np.maximum(0, dw[:, 3])) * has_defense

        # Behavioral hijacking suppresses defense
        if self.hijack_intensity.any():
            suppress = self.hijack_intensity * c.hijack_defense_suppress
            shell_val *= (1.0 - suppress)
            camo_val *= (1.0 - suppress)
            counter_val *= (1.0 - suppress)

        camo_rolls = self.rng.random(n)

        # Hunt success base — role-dependent
        hunt_skill = aggression * (1.0 - 0.5 * decomp_pref)
        base_prob = c.predation_base_success + hunt_skill * 0.3
        base_prob += np.where(has_move, 0.08, 0.0)
        # Specialist bonus: herbivores and carnivores are better at their niche
        is_specialist = (troph == 1) | (troph == 2)
        base_prob += np.where(is_specialist, 0.15, 0.0)
        # Mobile consumer bonus
        consumer_mobile = has_move & has_consume
        base_prob += np.where(consumer_mobile, c.consumer_mobility_hunt_bonus, 0.0)
        # Omnivore penalty: generalists are less efficient hunters than specialists
        omni_penalty = np.where(troph == 4, c.omnivore_hunt_penalty, 1.0)

        # Energy extraction: specialists get more per interaction
        frac_base = np.where(is_specialist,
                             c.predation_energy_fraction * (1.0 + c.consumer_specialist_bonus),
                             c.predation_energy_fraction)

        # Flat spatial index
        cell_key = self.rows * N + self.cols
        sort_idx = np.argsort(cell_key)
        sorted_keys = cell_key[sort_idx]
        unique_cells, cell_starts, cell_counts = np.unique(
            sorted_keys, return_index=True, return_counts=True)
        cell_start_map = np.full(N * N, -1, dtype=np.int32)
        cell_count_map = np.zeros(N * N, dtype=np.int32)
        cell_start_map[unique_cells] = cell_starts.astype(np.int32)
        cell_count_map[unique_cells] = cell_counts.astype(np.int32)

        kill_mask = np.zeros(n, dtype=bool)
        cell_kill_count = np.zeros(N * N, dtype=np.int32)
        satiated = np.zeros(n, dtype=bool)
        kills = 0

        predator_idx = np.where(predator_mask)[0]
        self.rng.shuffle(predator_idx)
        hunt_rolls = self.rng.random(len(predator_idx))
        sat_rolls = self.rng.random(len(predator_idx))

        for pi_idx, pi in enumerate(predator_idx):
            if kill_mask[pi] or satiated[pi]:
                continue

            pi_role = troph[pi]  # 1=herb, 2=carn, 4=omni
            pr, pc = int(self.rows[pi]), int(self.cols[pi])
            # Carnivores range further — apex predators cover large territories
            if pi_role == 2 and has_move[pi]:
                R = c.predation_hunt_radius_carnivore
            elif has_move[pi]:
                R = c.predation_hunt_radius_mobile
            else:
                R = c.predation_hunt_radius_base

            # Gather candidates with HARD TROPHIC FILTERING
            candidates = []
            r_lo, r_hi = max(0, pr - R), min(N - 1, pr + R)
            c_lo, c_hi = max(0, pc - R), min(N - 1, pc + R)
            for cr in range(r_lo, r_hi + 1):
                for cc_iter in range(c_lo, c_hi + 1):
                    ck = cr * N + cc_iter
                    if cell_kill_count[ck] >= c.predation_max_kills_per_cell:
                        continue
                    cs = cell_start_map[ck]
                    if cs < 0:
                        continue
                    cnt = cell_count_map[ck]
                    for si in range(cs, cs + cnt):
                        j = sort_idx[si]
                        if j == pi or kill_mask[j]:
                            continue
                        if camo_val[j] > 0 and camo_rolls[j] < camo_val[j]:
                            continue

                        # HARD TROPHIC RULES
                        j_role = troph[j]
                        if pi_role == 1:
                            # Herbivore: can ONLY eat producers
                            if j_role != 0:
                                continue
                        elif pi_role == 2:
                            # Carnivore: can ONLY eat animals (NOT producers)
                            if j_role == 0:
                                continue
                        elif pi_role == 4:
                            # Omnivore: can eat producers AND animals
                            pass

                        # Detritivores are unpalatable — predators usually skip them
                        if j_role == 3 and self.rng.random() < c.detritivore_predation_resist:
                            continue

                        candidates.append(j)

            if not candidates:
                continue

            # Target selection: weight by energy (prefer weaker prey)
            cands = np.array(candidates)
            target_e = np.maximum(self.energy[cands], 1.0)
            sel_weight = 1.0 / target_e
            sel_weight /= sel_weight.sum()
            ti = cands[self.rng.choice(len(cands), p=sel_weight)]

            # Success probability
            target_difficulty = np.clip(self.energy[ti] / c.energy_max, 0.1, 1.0)
            prob = base_prob[pi] - target_difficulty * 0.08
            # Omnivore generalist penalty
            prob *= omni_penalty[pi]

            # Sessile prey vulnerability: immobile prey cannot flee
            if not has_move[ti]:
                prob += c.sessile_prey_vulnerability

            # Type match bonuses for specialists
            ti_is_prod = (troph[ti] == 0)
            if pi_role == 1 and ti_is_prod:
                prob += c.herbivore_producer_bonus
            elif pi_role == 2 and not ti_is_prod:
                prob += c.carnivore_consumer_bonus

            prob -= shell_val[ti]
            prob = np.clip(prob, 0.01, 0.65)

            if hunt_rolls[pi_idx] < prob:
                # ── GRAZING vs LETHAL KILL ──
                # Grazing (non-lethal): herbivores and omnivores eating producers
                # Killing (lethal): carnivores eating animals, omnivores eating animals
                target_is_producer = (troph[ti] == 0)
                
                if target_is_producer and pi_role in (1, 4):
                    # GRAZING: take energy but don't kill the plant
                    # Can't graze a depleted plant — nothing left to eat
                    if self.energy[ti] < c.graze_min_prey_energy:
                        continue  # skip, plant too depleted
                    graze_mult = c.herbivore_energy_mult if pi_role == 1 else c.omnivore_graze_mult
                    grazed = max(0.0, self.energy[ti] - c.graze_min_prey_energy) * c.graze_energy_fraction * graze_mult
                    self.energy[pi] = min(self.energy[pi] + grazed, c.energy_max)
                    self.energy[ti] -= grazed
                else:
                    # LETHAL KILL: carnivore or omnivore eating an animal
                    if pi_role == 2:
                        energy_mult = c.carnivore_energy_mult
                    elif pi_role == 4:
                        energy_mult = c.omnivore_kill_mult
                    else:
                        energy_mult = 1.0
                    gained = max(0.0, self.energy[ti]) * frac_base[pi] * energy_mult
                    self.energy[pi] = min(self.energy[pi] + gained, c.energy_max)
                    self.energy[ti] = -1.0
                    kill_mask[ti] = True
                    cell_kill_count[int(self.rows[ti]) * N + int(self.cols[ti])] += 1
                    kills += 1
                
                if sat_rolls[pi_idx] < c.predator_satiation:
                    satiated[pi] = True
            else:
                if counter_val[ti] > 0:
                    self.energy[pi] -= counter_val[ti]

        self.total_predation_kills += kills
        return kills

    # ── Toxic Damage ──

    def _compute_hijack_intensity(self):
        """Compute per-organism hijack intensity from viral load.
        Returns array of 0-1 values. Called once per step, stored on self."""
        c = self.cfg
        n = self.pop
        self.hijack_intensity = np.zeros(n)
        if n == 0:
            return

        # Hijack active when viral_load in range [min, burst_threshold)
        active = (self.viral_load >= c.hijack_load_min) & (self.viral_load < c.viral_burst_threshold)
        if not active.any():
            return

        aidx = np.where(active)[0]
        load = self.viral_load[aidx]

        # Intensity: ramps from 0 at hijack_load_min to 1 at hijack_load_heavy
        raw = (load - c.hijack_load_min) / max(c.hijack_load_heavy - c.hijack_load_min, 0.01)
        intensity = np.clip(raw, 0.0, 1.0)

        # Toxic stress amplifies hijack (stressed hosts easier to control)
        local_toxic = self.toxic[self.rows[aidx], self.cols[aidx]]
        stress_mult = 1.0 + local_toxic * c.hijack_stress_amplifier
        intensity = np.minimum(intensity * stress_mult, 1.0)

        # VRESIST partially resists hijacking
        has_vresist = self.module_active[aidx, M_VRESIST]
        if has_vresist.any():
            vw = self._module_weights(M_VRESIST)
            breadth = 1.0 / (1.0 + np.exp(-vw[aidx, 2]))
            memory = np.minimum(self.immune_experience[aidx], 3.0) / 3.0
            resistance = (breadth * 0.5 + memory * 0.5) * c.hijack_vresist_suppress
            intensity[has_vresist] *= (1.0 - resistance[has_vresist])

        self.hijack_intensity[aidx] = intensity
        self.total_hijacked_steps += int(active.sum())

    # ── Endosymbiosis ──

