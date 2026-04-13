"""Endosymbiotic mergers, capacity shedding, genomic incompatibility."""

import numpy as np

from .constants import *


class EndosymbiosisMixin:
    """Endosymbiotic mergers, capacity shedding, genomic incompatibility."""

    def _attempt_endosymbiosis(self):
        """Check for endosymbiotic mergers between co-located compatible organisms.
        Uses spatial index to find pairs in the same cell."""
        c = self.cfg
        n = self.pop
        if n < 2:
            return 0
        N = c.grid_size

        # Eligible: high relationship, sufficient energy, no active infection
        # SOCIAL not required — relationship builds from co-location
        # (SOCIAL just accelerates relationship growth)
        eligible = ((self.relationship_score >= c.endo_relationship_threshold)
                   & (self.energy >= c.endo_energy_threshold)
                   & (self.viral_load == 0))
        if eligible.sum() < 2:
            return 0

        eidx = np.where(eligible)[0]

        # Local toxic must be in stress window
        local_toxic = self.toxic[self.rows[eidx], self.cols[eidx]]
        in_window = (local_toxic >= c.endo_toxic_min) & (local_toxic <= c.endo_toxic_max)
        eidx = eidx[in_window]
        if len(eidx) < 2:
            return 0

        # Group by cell
        cell_key = self.rows[eidx] * N + self.cols[eidx]
        sort_order = np.argsort(cell_key)
        sorted_keys = cell_key[sort_order]
        sorted_idx = eidx[sort_order]

        mergers_a = []  # index of organism A
        mergers_b = []  # index of organism B
        merged_set = set()

        # Find pairs in same cell
        i = 0
        while i < len(sorted_keys) - 1:
            if sorted_keys[i] == sorted_keys[i + 1]:
                # Same cell — check this pair
                a, b = int(sorted_idx[i]), int(sorted_idx[i + 1])
                if a not in merged_set and b not in merged_set:
                    # Complementarity: count modules that differ
                    mods_a = self.module_present[a]
                    mods_b = self.module_present[b]
                    diff_count = int((mods_a != mods_b).sum())
                    if diff_count >= c.endo_complementarity_min:
                        # Merger probability
                        prob = c.endo_probability
                        # VRESIST penalty (immune system resists integration)
                        if self.module_active[a, M_VRESIST] or self.module_active[b, M_VRESIST]:
                            prob *= (1.0 - c.endo_vresist_penalty)
                        # Higher relationship → higher probability
                        rel_avg = (self.relationship_score[a] + self.relationship_score[b]) / 2.0
                        prob *= min(rel_avg / c.endo_relationship_threshold, 2.0)

                        if self.rng.random() < prob:
                            mergers_a.append(a)
                            mergers_b.append(b)
                            self._flag_merger()
                            merged_set.add(a)
                            merged_set.add(b)
                # Skip to next potential pair
                i += 2
            else:
                i += 1

        if not mergers_a:
            return 0

        # Execute mergers: create composite organisms
        nm = len(mergers_a)
        ma = np.array(mergers_a)
        mb = np.array(mergers_b)

        # Composite modules: union of both, capped
        composite_modules = self.module_present[ma] | self.module_present[mb]
        for i in range(nm):
            if composite_modules[i].sum() > c.endo_max_modules:
                # Drop least useful modules (keep energy sources + core)
                present = np.where(composite_modules[i])[0]
                # Priority: energy modules > behavioral > metabolic
                priority = {M_PHOTO: 10, M_CHEMO: 10, M_CONSUME: 10, M_MOVE: 8,
                           M_TOXPROD: 7, M_FORAGE: 6, M_DEFENSE: 6, M_DETOX: 5,
                           M_VRESIST: 4, M_SOCIAL: 9, M_MEDIATE: 3}
                by_pri = sorted(present, key=lambda m: priority.get(m, 0))
                while composite_modules[i].sum() > c.endo_max_modules:
                    composite_modules[i, by_pri.pop(0)] = False

        # Blended weights
        blend = c.endo_weight_blend
        composite_weights = blend * self.weights[ma] + (1.0 - blend) * self.weights[mb]

        # Composite energy
        composite_energy = np.minimum(
            (self.energy[ma] + self.energy[mb]) * c.endo_energy_bonus,
            c.energy_max)

        # Merger count: max of both + 1
        composite_merger = np.maximum(self.merger_count[ma], self.merger_count[mb]) + 1

        # Lysogenic: blend
        composite_lyso_str = np.maximum(self.lysogenic_strength[ma], self.lysogenic_strength[mb])
        composite_lyso_gen = blend * self.lysogenic_genome[ma] + (1.0 - blend) * self.lysogenic_genome[mb]

        # Create composite organisms
        composite_ids = np.arange(self.next_id, self.next_id + nm, dtype=np.int64)
        self.next_id += nm

        self._append_organisms({
            "rows": self.rows[ma].copy(),
            "cols": self.cols[ma].copy(),
            "energy": composite_energy,
            "age": np.zeros(nm, dtype=np.int32),
            "generation": np.maximum(self.generation[ma], self.generation[mb]) + 1,
            "ids": composite_ids,
            "parent_ids": self.ids[ma],  # track one parent
            "weights": composite_weights,
            "module_present": composite_modules,
            "module_active": composite_modules.copy(),
            "transfer_count": np.maximum(self.transfer_count[ma], self.transfer_count[mb]),
            "viral_load": np.zeros(nm, dtype=np.float64),
            "lysogenic_strength": composite_lyso_str,
            "lysogenic_genome": composite_lyso_gen,
            "immune_experience": np.maximum(self.immune_experience[ma], self.immune_experience[mb]),
            "relationship_score": np.zeros(nm, dtype=np.float64),
            "merger_count": composite_merger,
            "module_usage": np.ones((nm, N_MODULES), dtype=np.float64) * composite_modules,
            "genomic_stress": np.maximum(self.genomic_stress[ma], self.genomic_stress[mb]) * 0.5,
            "genomic_cascade_phase": np.zeros(nm, dtype=np.int32),
            "dev_progress": np.ones(nm, dtype=np.float64),  # merged organisms are mature by definition
            "is_mature": np.ones(nm, dtype=bool),
        })

        # Remove both partners
        remove_mask = np.ones(self.pop - nm, dtype=bool)  # pop already increased by append
        # We need to mark the ORIGINAL indices for removal
        # Since append added nm organisms at the end, original indices are still valid
        old_pop = self.pop - nm
        keep = np.ones(old_pop, dtype=bool)
        keep[ma] = False
        keep[mb] = False
        # But we also want to keep the newly appended ones
        full_keep = np.concatenate([keep, np.ones(nm, dtype=bool)])
        self._filter_organisms(full_keep)

        self.total_mergers += nm
        return nm

    # ── Movement ──

    def _update_module_usage(self, readings):
        """Update per-module usage scores based on environmental relevance.
        Modules that are useful in current conditions gain usage.
        All present modules decay. Low usage → dormancy → loss."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        local_light = readings[:, 0]
        local_toxic = readings[:, 1]
        local_density = readings[:, 3]
        local_decomp = readings[:, 8]

        # Relevance signals: is this module actively needed in current conditions?
        # Must be demanding enough that unused modules actually decay
        relevance = np.zeros((n, N_MODULES), dtype=np.float64)
        relevance[:, M_PHOTO] = (local_light > 0.5).astype(np.float64) * (local_toxic < 0.5).astype(np.float64)
        relevance[:, M_CHEMO] = (local_toxic > 0.3).astype(np.float64)
        relevance[:, M_CONSUME] = np.minimum(local_density * 0.3, 1.0)
        relevance[:, M_MOVE] = np.where(local_density > 2, 0.8, 0.2)  # crowded = need to move
        relevance[:, M_FORAGE] = (self.nutrients[self.rows, self.cols] > 1.0).astype(np.float64)
        relevance[:, M_DEFENSE] = np.minimum(local_density * 0.3, 1.0)  # relevant when crowded (predator risk)
        relevance[:, M_DETOX] = (local_toxic > 0.3).astype(np.float64)
        relevance[:, M_TOXPROD] = 1.0  # always relevant (never sheds)
        relevance[:, M_VRESIST] = np.minimum(
            self.viral_particles[self.rows, self.cols] * 10.0, 1.0)
        relevance[:, M_SOCIAL] = np.minimum(local_density * 0.4, 1.0)   # any neighbor helps
        relevance[:, M_MEDIATE] = np.minimum(local_density * 0.35, 1.0)  # pollinators useful near any life

        # Gain: active modules in relevant conditions gain usage
        gain = c.shedding_usage_gain * relevance * self.module_active
        self.module_usage += gain

        # Decay: all present modules decay (dimension-specific rates)
        decay = SHEDDING_DECAY[None, :] * self.module_present
        self.module_usage -= decay

        np.clip(self.module_usage, 0.0, 1.0, out=self.module_usage)

    def _apply_capacity_shedding(self):
        """Check for module deactivation and loss. Stress reactivation."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        # Stress reactivation: high toxic or viral pressure reactivates dormant modules
        local_toxic = self.toxic[self.rows, self.cols]
        local_viral = self.viral_particles[self.rows, self.cols]
        stressed = (local_toxic > c.shedding_stress_reactivation) | (local_viral > 0.1)
        if stressed.any():
            sidx = np.where(stressed)[0]
            dormant = self.module_present[sidx] & ~self.module_active[sidx]
            # Reactivate dormant modules under stress, boost their usage
            reactivate = dormant & (self.module_usage[sidx] > c.shedding_loss_threshold)
            self.module_active[sidx] |= reactivate
            self.module_usage[sidx] += reactivate * 0.3
            np.clip(self.module_usage[sidx], 0.0, 1.0, out=self.module_usage[sidx])

        # Deactivation: low usage → module goes dormant
        low_usage = (self.module_usage < c.shedding_deactivate_threshold) & self.module_active
        # Protect TOXPROD and last energy module
        low_usage[:, M_TOXPROD] = False
        for i in np.where(low_usage.any(axis=1))[0]:
            for m in np.where(low_usage[i])[0]:
                if m in (M_PHOTO, M_CHEMO, M_CONSUME):
                    # Don't deactivate last energy source
                    others = [e for e in (M_PHOTO, M_CHEMO, M_CONSUME)
                             if e != m and self.module_active[i, e]]
                    if not others:
                        low_usage[i, m] = False
        self.module_active[low_usage] = False

        # Genome loss: very low usage → module lost entirely (irreversible)
        very_low = (self.module_usage < c.shedding_loss_threshold) & self.module_present
        very_low[:, M_TOXPROD] = False
        for i in np.where(very_low.any(axis=1))[0]:
            for m in np.where(very_low[i])[0]:
                if m in (M_PHOTO, M_CHEMO, M_CONSUME):
                    others = [e for e in (M_PHOTO, M_CHEMO, M_CONSUME)
                             if e != m and self.module_present[i, e]]
                    if not others:
                        very_low[i, m] = False
        self.module_present[very_low] = False
        self.module_active[very_low] = False
        self.module_usage[very_low] = 0.0

    # ── Genomic Incompatibility ──

    def _apply_genomic_incompatibility(self, energy_gain):
        """Check for and apply genomic incompatibility cascade.
        Modifies energy_gain in-place for phase 1. Returns pruning events count."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return 0

        # Natural stress decay (integration/accommodation over time)
        self.genomic_stress -= c.genomic_stress_decay
        np.clip(self.genomic_stress, 0.0, 10.0, out=self.genomic_stress)

        # Effective stress = genomic_stress + toxic amplification
        local_toxic = self.toxic[self.rows, self.cols]
        effective_stress = self.genomic_stress + local_toxic * c.genomic_toxic_multiplier

        # Phase transitions (higher effective_stress → worse phase)
        # Phase 0 → 1 at cascade_threshold, 1 → 2 at threshold*1.5, 2 → 3 at threshold*2
        thresh = c.genomic_cascade_threshold
        entering_p1 = (effective_stress >= thresh) & (self.genomic_cascade_phase == 0)
        entering_p2 = (effective_stress >= thresh * 1.5) & (self.genomic_cascade_phase == 1)
        entering_p3 = (effective_stress >= thresh * 2.0) & (self.genomic_cascade_phase == 2)

        # Recovery: stress drops below threshold → step down
        recovering = (effective_stress < thresh * 0.7) & (self.genomic_cascade_phase > 0)
        self.genomic_cascade_phase[recovering] = np.maximum(0, self.genomic_cascade_phase[recovering] - 1)

        self.genomic_cascade_phase[entering_p1] = 1
        self.genomic_cascade_phase[entering_p2] = 2
        self.genomic_cascade_phase[entering_p3] = 3

        # Phase 1: Metabolic disruption — energy acquisition reduced
        in_p1 = self.genomic_cascade_phase >= 1
        if in_p1.any():
            energy_gain[in_p1] *= (1.0 - c.genomic_phase1_energy_penalty)

        # Phase 2: Regulatory breakdown — random module deactivation
        in_p2 = np.where(self.genomic_cascade_phase >= 2)[0]
        if len(in_p2) > 0:
            deactivate_mask = self.rng.random(len(in_p2)) < c.genomic_phase2_deactivate_prob
            for idx in in_p2[deactivate_mask]:
                active_mods = np.where(self.module_active[idx] & (np.arange(N_MODULES) != M_TOXPROD))[0]
                # Protect last energy source
                energy_mods = [m for m in active_mods if m in (M_PHOTO, M_CHEMO, M_CONSUME)]
                safe_to_deactivate = [m for m in active_mods
                                     if m not in (M_PHOTO, M_CHEMO, M_CONSUME) or len(energy_mods) > 1]
                if safe_to_deactivate:
                    self.module_active[idx, self.rng.choice(safe_to_deactivate)] = False

        # Phase 3: Identity dissolution — direct energy drain
        in_p3 = self.genomic_cascade_phase == 3
        if in_p3.any():
            self.energy[in_p3] -= c.genomic_phase3_energy_drain

        # Genomic pruning: extreme stress → shed foreign material to survive
        prune_candidates = np.where(self.genomic_stress >= c.genomic_pruning_threshold)[0]
        n_pruned = 0
        for idx in prune_candidates:
            # Lose up to N most recently acquired modules (highest transfer_count correlation)
            expendable = np.where(self.module_present[idx] & (np.arange(N_MODULES) != M_TOXPROD))[0]
            # Protect last energy source
            energy_mods = [m for m in expendable if m in (M_PHOTO, M_CHEMO, M_CONSUME)]
            if len(energy_mods) <= 1:
                expendable = [m for m in expendable if m not in energy_mods]
            else:
                expendable = list(expendable)

            n_to_lose = min(c.genomic_pruning_module_loss, len(expendable))
            if n_to_lose > 0:
                # Lose least-used modules first (synergy with shedding)
                by_usage = sorted(expendable, key=lambda m: self.module_usage[idx, m])
                for m in by_usage[:n_to_lose]:
                    self.module_present[idx, m] = False
                    self.module_active[idx, m] = False
                    self.module_usage[idx, m] = 0.0

                # Relief
                self.genomic_stress[idx] *= (1.0 - c.genomic_pruning_stress_relief)
                self.genomic_cascade_phase[idx] = 0

                # Pruned organisms become highly receptive to new material
                sp = self._standalone_params()
                sp[idx, 0] = min(sp[idx, 0] + c.genomic_pruning_receptivity_boost, 3.0)

                n_pruned += 1

        return n_pruned

    # ── Developmental Dependency ──

