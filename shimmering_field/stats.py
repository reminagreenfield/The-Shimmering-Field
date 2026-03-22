"""Statistics, role classification, snapshots."""

import numpy as np
import json
import os

from .constants import *


class StatsMixin:
    """Statistics, role classification, snapshots."""

    def _trophic_roles(self):
        """Per-organism trophic role: 0=producer, 1=herbivore, 2=carnivore, 3=detritivore, 4=omnivore.
        Biological definitions:
          Producer:    PHOTO/CHEMO, no CONSUME — plants/autotrophs
          Herbivore:   CONSUME, low prey_selectivity — eats producers only
          Carnivore:   CONSUME, high prey_selectivity — eats animals only
          Omnivore:    CONSUME, mid prey_selectivity — eats both (less efficient than specialists)
          Detritivore: CONSUME, high decomp_pref — eats dead material
        Organisms with PHOTO+CONSUME are classified by their CONSUME weights (rare hybrids).
        """
        n = self.pop
        roles = np.zeros(n, dtype=np.int32)  # default: producer
        has_cons = self.module_present[:, M_CONSUME]

        if has_cons.any():
            uw = self._module_weights(M_CONSUME)
            dp = 1.0 / (1.0 + np.exp(-uw[:, 2]))  # decomp_pref
            ps = 1.0 / (1.0 + np.exp(-uw[:, 0]))  # prey_selectivity
            cons_idx = np.where(has_cons)[0]
            for i in cons_idx:
                if dp[i] >= self.cfg.detritivore_decomp_threshold:
                    roles[i] = 3  # detritivore
                elif ps[i] < self.cfg.herbivore_selectivity_threshold:
                    roles[i] = 1  # herbivore — eats plants only
                elif ps[i] > self.cfg.carnivore_selectivity_threshold:
                    roles[i] = 2  # carnivore — eats animals only
                else:
                    roles[i] = 4  # omnivore — eats both, less efficiently
        return roles

    def _classify_roles(self):
        n = self.pop
        if n == 0:
            return {"producer": 0, "herbivore": 0, "carnivore": 0, "detritivore": 0, "omnivore": 0}
        roles = self._trophic_roles()
        return {
            "producer": int((roles == 0).sum()),
            "herbivore": int((roles == 1).sum()),
            "carnivore": int((roles == 2).sum()),
            "detritivore": int((roles == 3).sum()),
            "omnivore": int((roles == 4).sum()),
        }

    def _record_stats(self, kills=0):
        p = self.pop
        if p > 0:
            roles = self._classify_roles()
            mod_counts = {MODULE_NAMES[m]: int(self.module_present[:, m].sum())
                         for m in range(N_MODULES)}
            self.stats_history.append({
                "t": self.timestep, "pop": p,
                "avg_energy": round(float(self.energy.mean()), 1),
                "max_gen": int(self.generation.max()),
                "toxic_mean": round(float(self.toxic.mean()), 3),
                "decomp_mean": round(float(self.decomposition.mean()), 2),
                "avg_modules": round(float(self.module_present.sum(axis=1).mean()), 2),
                "module_counts": mod_counts,
                "roles": roles,
                "kills": kills,
                "total_kills": self.total_predation_kills,
                "avg_immune_exp": round(float(self.immune_experience.mean()), 3),
                "avg_relationship": round(float(self.relationship_score.mean()), 3),
                "max_relationship": round(float(self.relationship_score.max()), 3),
                "mediator_field_mean": round(float(self.mediator_field.mean()), 4),
                "nutrient_mean": round(float(self.nutrients.mean()), 3),
                "lyso_fraction": round(float((self.lysogenic_strength > self.cfg.repro_manip_threshold).sum() / max(p, 1)), 3),
                "manipulated_births": self.total_manipulated_births,
                "hijack_fraction": round(float((self.hijack_intensity > 0.1).sum() / max(p, 1)), 3),
                "hijacked_steps": self.total_hijacked_steps,
                "total_mergers": self.total_mergers,
                "max_merger_count": int(self.merger_count.max()) if p > 0 else 0,
                "composite_organisms": int((self.merger_count > 0).sum()),
                "dormant_modules": int((self.module_present & ~self.module_active).sum()),
                "avg_usage": round(float(self.module_usage[self.module_present].mean()), 3) if self.module_present.any() else 0,
                "avg_genomic_stress": round(float(self.genomic_stress.mean()), 3),
                "cascade_organisms": int((self.genomic_cascade_phase > 0).sum()),
                "max_cascade_phase": int(self.genomic_cascade_phase.max()),
                "mature_fraction": round(float(self.is_mature.sum() / max(p, 1)), 3),
                "compromised_count": int(((self.age > self.cfg.dev_window_length) & ~self.is_mature).sum()),
                "collapsed_zones": int(self.zone_collapsed.sum()),
                "avg_integrity": round(float(self.ecosystem_integrity.mean()), 3),
                "fungal_mean": round(float(self.fungal_density.mean()), 4),
                "fungal_max": round(float(self.fungal_density.max()), 3),
                "sessile_producers": int((self.module_present[:, M_PHOTO] & ~self.module_active[:, M_MOVE]).sum()),
                "mobile_producers": int((self.module_present[:, M_PHOTO] & self.module_active[:, M_MOVE]).sum()),
                "mobile_consumers": int((self.module_present[:, M_CONSUME] & self.module_active[:, M_MOVE]).sum()),
                "sessile_consumers": int((self.module_present[:, M_CONSUME] & ~self.module_active[:, M_MOVE]).sum()),
            })
        else:
            self.stats_history.append({
                "t": self.timestep, "pop": 0, "avg_energy": 0, "max_gen": 0,
                "toxic_mean": round(float(self.toxic.mean()), 3),
                "decomp_mean": round(float(self.decomposition.mean()), 2),
                "avg_modules": 0, "module_counts": {MODULE_NAMES[m]: 0 for m in range(N_MODULES)},
                "roles": {"producer": 0, "herbivore": 0, "carnivore": 0, "detritivore": 0, "omnivore": 0},
                "kills": 0, "total_kills": self.total_predation_kills,
                "avg_immune_exp": 0, "avg_relationship": 0, "max_relationship": 0,
                "mediator_field_mean": 0, "nutrient_mean": round(float(self.nutrients.mean()), 3),
                "lyso_fraction": 0, "manipulated_births": self.total_manipulated_births,
                "hijack_fraction": 0, "hijacked_steps": self.total_hijacked_steps,
                "total_mergers": self.total_mergers, "max_merger_count": 0, "composite_organisms": 0,
                "dormant_modules": 0, "avg_usage": 0,
                "avg_genomic_stress": 0, "cascade_organisms": 0, "max_cascade_phase": 0,
                "mature_fraction": 0, "compromised_count": 0,
                "collapsed_zones": int(self.zone_collapsed.sum()),
                "avg_integrity": round(float(self.ecosystem_integrity.mean()), 3),
                "fungal_mean": round(float(self.fungal_density.mean()), 4),
                "fungal_max": round(float(self.fungal_density.max()), 3),
                "sessile_producers": 0, "mobile_producers": 0,
                "mobile_consumers": 0, "sessile_consumers": 0,
            })

    def save_snapshot(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        p = self.pop
        idx = self.rng.choice(p, min(p, 500), replace=False) if p > 500 else np.arange(p)
        orgs = [{
            "id": int(self.ids[i]), "row": int(self.rows[i]), "col": int(self.cols[i]),
            "energy": round(float(self.energy[i]), 2), "age": int(self.age[i]),
            "generation": int(self.generation[i]),
            "modules": [MODULE_NAMES[m] for m in range(N_MODULES) if self.module_present[i, m]],
        } for i in idx]
        s = self.stats_history[-1] if self.stats_history else {}
        with open(os.path.join(output_dir, f"snapshot_{self.timestep:06d}.json"), 'w') as f:
            json.dump({"timestep": self.timestep, "population": p, "organisms": orgs, "stats": s}, f)


# ─────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────

