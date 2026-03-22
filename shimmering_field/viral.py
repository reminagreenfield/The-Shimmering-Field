"""Viral dynamics and horizontal gene transfer."""

import numpy as np

from .constants import *


class ViralMixin:
    """Viral dynamics and horizontal gene transfer."""

    def _horizontal_transfer(self):
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        sp = self._standalone_params()
        receptivity = 1.0 / (1.0 + np.exp(-sp[:, SP_TRANSFER_RECEPTIVITY]))
        selectivity = np.abs(sp[:, SP_TRANSFER_SELECTIVITY])
        local_toxic = self.toxic[self.rows, self.cols]

        # Fungal network boost: organisms near fungi receive material more easily
        local_fungal = self.fungal_density[self.rows, self.cols]
        fungal_boost = np.minimum(local_fungal * c.fungal_hgt_transport, 0.3)
        receptivity = np.minimum(receptivity + fungal_boost, 1.0)

        can_recent = np.ones(n, dtype=bool)
        can_intermediate = local_toxic >= c.stratum_access_medium
        can_ancient = local_toxic >= c.stratum_access_high

        for sname, access_mask, blend_rate in [
            ("recent",       can_recent,       c.transfer_blend_rate_recent),
            ("intermediate", can_intermediate,  c.transfer_blend_rate_intermediate),
            ("ancient",      can_ancient,       c.transfer_blend_rate_ancient),
        ]:
            sw = self.strata_weight[sname]
            local_sw = sw[self.rows, self.cols]
            eligible = access_mask & (local_sw > 0.1) & (self.rng.random(n) < receptivity)
            eidx = np.where(eligible)[0]
            if len(eidx) == 0:
                continue

            local_frags = self.strata_pool[sname][self.rows[eidx], self.cols[eidx]]
            dists = np.sqrt(np.mean((self.weights[eidx] - local_frags) ** 2, axis=1))
            depth_factor = {"recent": 1.0, "intermediate": 1.5, "ancient": 2.5}[sname]
            thresh = (2.0 * depth_factor) / (1.0 + selectivity[eidx])
            tidx = eidx[dists < thresh]
            if len(tidx) == 0:
                continue

            frags = self.strata_pool[sname][self.rows[tidx], self.cols[tidx]]
            self.weights[tidx] = (1.0 - blend_rate) * self.weights[tidx] + blend_rate * frags
            self.transfer_count[tidx] += 1
            self.total_transfers += len(tidx)
            self.transfers_by_stratum[sname] += len(tidx)

            # Genomic stress from foreign material (deeper strata = more incompatible)
            stratum_mult = {"recent": 1.0, "intermediate": 1.5, "ancient": 2.5}
            self.genomic_stress[tidx] += c.genomic_stress_per_transfer * stratum_mult.get(sname, 1.0)

            # Module acquisition via HGT
            if self.rng.random() < 0.5:
                frag_mods = self.strata_modules[sname][self.rows[tidx], self.cols[tidx]]
                for i, ti in enumerate(tidx):
                    if self.rng.random() < c.module_transfer_rate:
                        available = [gm for gm in GAINABLE_MODULES
                                    if frag_mods[i, gm] > 0.3 and not self.module_present[ti, gm]]
                        if available:
                            self.module_present[ti, self.rng.choice(available)] = True
                            self.module_active[ti] = self.module_present[ti]

    # ── Viral Dynamics (VRESIST-enhanced) ──

    def _viral_dynamics(self):
        """Viral infection, lytic damage, lysogenic activation.
        VRESIST module provides much stronger resistance with specificity/breadth trade-off.
        Organisms without VRESIST use weaker standalone params as fallback."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        sp = self._standalone_params()
        has_vresist = self.module_active[:, M_VRESIST]
        local_viral = self.viral_particles[self.rows, self.cols]
        local_toxic = self.toxic[self.rows, self.cols]

        # Compute viral resistance: VRESIST module vs standalone fallback
        # Standalone: weak baseline resistance (~0.3 effective)
        base_resistance = 0.3 * (1.0 / (1.0 + np.exp(-sp[:, SP_VIRAL_RESISTANCE])))

        if has_vresist.any():
            vw = self._module_weights(M_VRESIST)
            specificity = 1.0 / (1.0 + np.exp(-vw[:, 0]))  # targeted resistance
            breadth = 1.0 / (1.0 + np.exp(-vw[:, 2]))      # broad resistance
            # Specificity/breadth trade-off: can't max both
            # Specificity good in stable environments; breadth good against novel strains
            vresist_resist = (c.vresist_base_resistance
                            + c.vresist_specificity_bonus * specificity
                            + c.vresist_breadth_bonus * breadth)
            # Immune memory from past infections boosts resistance
            memory_boost = (c.vresist_memory_boost
                          * (1.0 / (1.0 + np.exp(-vw[:, 3])))  # immune_memory weight
                          * np.minimum(self.immune_experience, 3.0))
            vresist_resist = np.minimum(vresist_resist + memory_boost, 0.95)
            # Merge: VRESIST organisms use module resistance, others use standalone
            viral_resistance = np.where(has_vresist, vresist_resist, base_resistance)
        else:
            viral_resistance = base_resistance

        # Compute lysogenic suppression: VRESIST enhances
        base_suppression = 0.3 * (1.0 / (1.0 + np.exp(-sp[:, SP_LYSO_SUPPRESSION])))
        if has_vresist.any():
            vw = self._module_weights(M_VRESIST)
            vresist_supp = c.vresist_suppression_max * (1.0 / (1.0 + np.exp(-vw[:, 1])))
            lysogenic_suppression = np.where(has_vresist, vresist_supp, base_suppression)
        else:
            lysogenic_suppression = base_suppression

        # Immune memory decay
        self.immune_experience *= c.vresist_memory_decay

        # New infections
        candidates = (self.viral_load == 0) & (local_viral > 0.05)
        if candidates.any():
            cidx = np.where(candidates)[0]
            inf_prob = (c.viral_infection_rate * np.minimum(local_viral[cidx], 5.0) / 5.0
                       * (1.0 - viral_resistance[cidx]))
            inf_prob = np.maximum(inf_prob, 0.01)  # minimum infection chance
            infected = cidx[self.rng.random(len(cidx)) < inf_prob]

            if len(infected) > 0:
                lyso_rolls = self.rng.random(len(infected))
                lytic = infected[lyso_rolls >= c.lysogenic_probability]
                lysogenic = infected[lyso_rolls < c.lysogenic_probability]

                if len(lytic) > 0:
                    self.viral_load[lytic] = 0.1
                if len(lysogenic) > 0:
                    vgw = self.viral_genome_weight[self.rows[lysogenic], self.cols[lysogenic]]
                    has_mat = vgw > 0.01
                    lwm = lysogenic[has_mat]
                    if len(lwm) > 0:
                        vg = self.viral_genome_pool[self.rows[lwm], self.cols[lwm]]
                        new_str = self.lysogenic_strength[lwm] + 0.3
                        blend = 0.3 / np.maximum(new_str, 0.01)
                        self.lysogenic_genome[lwm] = (
                            self.lysogenic_genome[lwm] * (1.0 - blend[:, None])
                            + vg * blend[:, None])
                        self.lysogenic_strength[lwm] = new_str
                        self.total_lysogenic_integrations += len(lwm)

        # Lytic damage + immune experience from surviving
        lytic_mask = self.viral_load > 0
        if lytic_mask.any():
            lidx = np.where(lytic_mask)[0]
            self.viral_load[lidx] += c.viral_lytic_growth
            self.energy[lidx] -= c.viral_lytic_damage * self.viral_load[lidx]

            # Organisms that survive lytic infection gain immune experience
            # (viral_load will be cleared by death; survivors keep accumulating)
            survivors = lidx[self.energy[lidx] > 0]
            if len(survivors) > 0:
                self.immune_experience[survivors] += 0.05

        # Lysogenic activation under stress (suppression from VRESIST)
        act_cand = ((self.lysogenic_strength > 0.01) &
                    (local_toxic > c.lysogenic_activation_toxic) &
                    (self.viral_load == 0))
        if act_cand.any():
            acidx = np.where(act_cand)[0]
            stress = (local_toxic[acidx] - c.lysogenic_activation_toxic) / c.lysogenic_activation_toxic
            act_prob = np.minimum(stress * 0.3, 0.5) * (1.0 - lysogenic_suppression[acidx])
            activated = acidx[self.rng.random(len(acidx)) < act_prob]
            if len(activated) > 0:
                blend = np.minimum(c.lysogenic_blend_rate * self.lysogenic_strength[activated], 0.4)
                self.weights[activated] = (
                    self.weights[activated] * (1.0 - blend[:, None])
                    + self.lysogenic_genome[activated] * blend[:, None])
                self.viral_load[activated] = 0.2
                self.lysogenic_strength[activated] *= 0.3
                self.total_lysogenic_activations += len(activated)

    # ── Reproduction ──

