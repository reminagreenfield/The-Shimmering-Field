"""Energy acquisition, costs, toxic damage, detox, nutrient cycling."""

import numpy as np

from .constants import *


class EnergyMixin:
    """Energy acquisition, costs, toxic damage, detox, nutrient cycling."""

    def _compute_module_costs(self):
        maint = self.module_present.astype(np.float64) @ MODULE_MAINTENANCE
        expr = self.module_active.astype(np.float64) @ MODULE_EXPRESSION
        return BASE_MAINTENANCE + maint + expr

    def _effective_energy_max(self):
        """FORAGE storage_cap: foragers can store extra energy."""
        c = self.cfg
        effective_max = np.full(self.pop, c.energy_max)
        has_forage = self.module_active[:, M_FORAGE]
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            storage = c.forage_storage_bonus * (1.0 / (1.0 + np.exp(-fw[:, 2])))
            effective_max += storage * has_forage
        return effective_max

    def _compute_total_costs(self):
        """Module costs + DEFENSE size surcharge + movement complexity scaling."""
        c = self.cfg
        costs = self._compute_module_costs()
        has_defense = self.module_active[:, M_DEFENSE]
        if has_defense.any():
            dw = self._module_weights(M_DEFENSE)
            size_invest = np.maximum(0, np.tanh(dw[:, 2])) * has_defense
            costs += size_invest * (c.defense_size_cost_mult - 1.0) * (
                MODULE_MAINTENANCE[M_DEFENSE] + MODULE_EXPRESSION[M_DEFENSE])
        # Movement complexity penalty: more modules = heavier = MOVE costs more
        # Each module beyond 2 adds 0.05 to MOVE cost.
        has_move = self.module_active[:, M_MOVE]
        if has_move.any():
            module_count = self.module_present.sum(axis=1).astype(np.float64)
            extra_modules = np.maximum(0, module_count - 2.0)
            move_surcharge = extra_modules * 0.05 * has_move
            costs += move_surcharge
            # Sessile-mobile divergence: MOVE costs much more for PURE producers
            # Plants aren't built for locomotion — but omnivores are animals, they move normally
            has_prod_mod = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
            is_pure_producer_mobile = has_move & has_prod_mod & ~self.module_active[:, M_CONSUME]
            move_base = MODULE_MAINTENANCE[M_MOVE] + MODULE_EXPRESSION[M_MOVE]
            # Only PURE producers pay 2.5x the base MOVE cost (not omnivores)
            costs += is_pure_producer_mobile * move_base * (c.move_cost_producer_mult - 1.0)
        return costs

    _ORG_FIELDS = [
        "rows", "cols", "energy", "age", "generation", "ids", "parent_ids",
        "weights", "module_present", "module_active", "transfer_count",
        "viral_load", "lysogenic_strength", "lysogenic_genome",
        "immune_experience", "relationship_score", "merger_count", "module_usage",
        "genomic_stress", "genomic_cascade_phase",
        "dev_progress", "is_mature",
    ]

    def _acquire_energy(self, readings):
        c = self.cfg
        n = self.pop
        total_gain = np.zeros(n)

        local_light = readings[:, 0]
        local_toxic = readings[:, 1]
        local_density = readings[:, 3]
        local_decomp = readings[:, 8]

        # Metabolic interference masks
        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]
        is_omnivore = has_producer & has_consume
        is_specialist = has_consume & ~has_producer

        prod_mult = np.where(is_omnivore, c.consume_producer_penalty, 1.0)
        cons_mult = np.where(is_omnivore, c.producer_consume_penalty, 1.0)
        spec_mult = np.where(is_specialist, 1.0 + c.consumer_specialist_bonus, 1.0)

        # ── FORAGE bonus: extraction efficiency ──
        has_forage = self.module_active[:, M_FORAGE]
        forage_mult = np.ones(n)
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            extract_eff = c.forage_extraction_bonus * (1.0 / (1.0 + np.exp(-fw[:, 0])))
            forage_mult = 1.0 + extract_eff * has_forage

        # PHOTO
        has_photo = self.module_active[:, M_PHOTO]
        if has_photo.any():
            ph = self._module_weights(M_PHOTO)
            eff = c.photosynthesis_base * (1.0 + 0.3 * np.tanh(ph[:, 0]))
            tol = np.maximum(0.1, 1.0 + 0.5 * np.tanh(ph[:, 1]))
            tp = np.maximum(0.0, 1.0 - local_toxic * c.toxic_photo_penalty / tol)
            le = np.maximum(0.3, 1.0 - 0.3 * np.tanh(ph[:, 2]))
            lm = np.power(np.maximum(local_light, 0.01), le)
            st = 0.5 + 0.5 / (1.0 + np.exp(-ph[:, 3]))
            sh = 1.0 / np.maximum(1.0, local_density * 0.5)
            # Sessile-mobile divergence:
            # Sessile producers: 1.7x bonus (deep roots, optimized orientation)
            # Mobile producers: 0.55x penalty (can't maintain chloroplasts while moving)
            is_sessile = has_photo & ~self.module_active[:, M_MOVE]
            is_mobile_prod = has_photo & self.module_active[:, M_MOVE]
            sessile_mobile_mult = np.ones(n)
            sessile_mobile_mult[is_sessile] = c.sessile_photo_bonus
            sessile_mobile_mult[is_mobile_prod] = c.mobile_producer_penalty
            total_gain += (eff * tp * lm * st * sh * prod_mult * forage_mult * sessile_mobile_mult) * has_photo

        # CHEMO
        has_chemo = self.module_active[:, M_CHEMO]
        if has_chemo.any():
            ch = self._module_weights(M_CHEMO)
            eff = c.chemosynthesis_base * (1.0 + 0.3 * np.tanh(ch[:, 0]))
            sat = 1.0 + np.abs(ch[:, 2]) * 0.5
            toxic_factor = local_toxic / (local_toxic + sat)
            sh = 1.0 / np.maximum(1.0, local_density * 0.4)
            total_gain += (eff * toxic_factor * sh * prod_mult * forage_mult) * has_chemo

        # CONSUME (scavenging/detritivory component)
        if has_consume.any():
            uw = self._module_weights(M_CONSUME)
            decomp_pref = 1.0 / (1.0 + np.exp(-uw[:, 2]))
            handling = 1.0 + 0.3 * np.tanh(uw[:, 1])

            # Detritivory: direct decomposition consumption (decomp_pref > 0.3)
            # High decomp_pref = dedicated detritivore, reliable energy from carrion/detritus
            extract_rate = 0.05 + 0.25 * decomp_pref  # up to 0.30 of local decomp
            intake_cap = 0.5 + 2.5 * decomp_pref      # up to 3.0 energy/step
            available = np.minimum(local_decomp * extract_rate, intake_cap)
            detritivore_gain = (c.scavenge_base * (0.3 + 1.2 * decomp_pref)
                               * handling * available * cons_mult * spec_mult * forage_mult)
            # Mobile consumers scavenge better (can reach carcasses faster)
            mobile_scav = np.where(self.module_active[:, M_MOVE] & has_consume,
                                   1.0 + c.consumer_mobility_scavenge_bonus, 1.0)
            total_gain += detritivore_gain * has_consume * mobile_scav

            # Consume from decomp field proportional to what was eaten
            consumed = np.minimum(local_decomp * 0.10 * decomp_pref, detritivore_gain * 0.15) * has_consume
            np.add.at(self.decomposition, (self.rows, self.cols), -consumed)
            np.clip(self.decomposition, 0, 30.0, out=self.decomposition)

        # ── FORAGE cooperative signal: nearby foragers boost each other ──
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            coop_strength = 1.0 / (1.0 + np.exp(-fw[:, 3]))
            forage_density = np.zeros_like(self.density)
            fidx = np.where(has_forage)[0]
            if len(fidx) > 0:
                np.add.at(forage_density, (self.rows[fidx], self.cols[fidx]), 1)
            local_foragers = forage_density[self.rows, self.cols].astype(np.float64)
            coop_bonus = (np.maximum(0, local_foragers - 1) * c.forage_cooperative_bonus
                         * coop_strength * has_forage)
            total_gain += coop_bonus

        return total_gain

    # ── Predation (optimized) ──

    def _apply_toxic_damage(self, readings):
        c = self.cfg
        lt = readings[:, 1]
        dmg = np.zeros(self.pop)

        # CHEMO tolerance (existing)
        has_chemo = self.module_active[:, M_CHEMO]
        chemo_reduction = np.ones(self.pop)
        if has_chemo.any():
            ch = self._module_weights(M_CHEMO)
            tolerance = 0.3 + 0.4 / (1.0 + np.exp(-ch[:, 1]))
            chemo_reduction[has_chemo] = 1.0 - tolerance[has_chemo]

        # DETOX tolerance (new — raises damage thresholds)
        has_detox = self.module_active[:, M_DETOX]
        threshold_boost = np.zeros(self.pop)
        if has_detox.any():
            dtw = self._module_weights(M_DETOX)
            threshold_boost = (c.detox_tolerance_bonus
                              * (1.0 / (1.0 + np.exp(-dtw[:, 1])))
                              * has_detox)

        effective = lt * chemo_reduction
        eff_thresh_med = c.toxic_threshold_medium + threshold_boost
        eff_thresh_high = c.toxic_threshold_high + threshold_boost

        m = effective > eff_thresh_med
        if m.any():
            dmg[m] += (effective[m] - eff_thresh_med[m]) * c.toxic_damage_medium
        h = effective > eff_thresh_high
        if h.any():
            dmg[h] += (effective[h] - eff_thresh_high[h]) * c.toxic_damage_high
        self.energy -= dmg

    def _apply_detox(self):
        """DETOX module: metabolize environmental toxins for energy."""
        c = self.cfg
        has_detox = self.module_active[:, M_DETOX]
        if not has_detox.any():
            return

        dtw = self._module_weights(M_DETOX)
        detox_eff = c.detox_rate_max * (1.0 / (1.0 + np.exp(-dtw[:, 0])))
        # selective_uptake: prefer certain toxin concentrations (Michaelis-Menten style)
        selectivity = 1.0 + np.abs(dtw[:, 3]) * 0.5
        local_toxic = self.toxic[self.rows, self.cols]
        uptake_factor = local_toxic / (local_toxic + selectivity)

        # Amount of toxin metabolized
        metabolized = detox_eff * uptake_factor * has_detox

        # Energy from toxin conversion
        conversion = c.detox_energy_conversion * (1.0 / (1.0 + np.exp(-dtw[:, 2])))
        energy_gained = metabolized * conversion * local_toxic
        self.energy += energy_gained * has_detox

        # Environmental detox: remove toxin from the grid
        toxin_removed = metabolized * c.detox_environment_effect * local_toxic
        np.add.at(self.toxic, (self.rows, self.cols), -toxin_removed * has_detox)
        np.clip(self.toxic, 0, 5.0, out=self.toxic)

    # ── Social System ──

    def _apply_nutrient_cycling(self):
        """Emergent nutrient cycling from module interactions.
        Not a separate module — organisms with DETOX, CONSUME, and FORAGE
        contribute to local nutrient chemistry as a side effect."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        # DETOX byproducts → nutrients (detoxification enriches soil)
        has_detox = self.module_active[:, M_DETOX]
        if has_detox.any():
            dtw = self._module_weights(M_DETOX)
            detox_eff = 1.0 / (1.0 + np.exp(-dtw[:, 0]))
            conversion = 1.0 / (1.0 + np.exp(-dtw[:, 2]))
            local_toxic = self.toxic[self.rows, self.cols]
            deposit = (c.nutrient_detox_deposit * detox_eff * conversion
                      * np.minimum(local_toxic, 2.0) * has_detox)
            np.add.at(self.nutrients, (self.rows, self.cols), deposit)

        # CONSUME processing → nutrients (digestion releases nutrients)
        has_consume = self.module_active[:, M_CONSUME]
        if has_consume.any():
            uw = self._module_weights(M_CONSUME)
            handling = 1.0 / (1.0 + np.exp(-uw[:, 1]))
            deposit = c.nutrient_consume_deposit * handling * has_consume
            np.add.at(self.nutrients, (self.rows, self.cols), deposit)

        # FORAGE cooperative clusters boost local nutrient regeneration
        has_forage = self.module_active[:, M_FORAGE]
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            coop_signal = 1.0 / (1.0 + np.exp(-fw[:, 3]))
            boost = c.nutrient_forage_coop_boost * coop_signal * has_forage
            np.add.at(self.nutrients, (self.rows, self.cols), boost)

        np.clip(self.nutrients, 0, self.cfg.nutrient_max, out=self.nutrients)

    # ── Behavioral Hijacking ──

