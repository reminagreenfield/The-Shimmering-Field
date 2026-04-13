"""Pivotal moment detection and full-state snapshot export."""

import json
import os

import numpy as np

from .constants import (
    MODULE_NAMES, MODULE_WEIGHT_OFFSETS, MODULE_WEIGHT_SIZES,
    N_MODULES, N_STANDALONE_PARAMS, STANDALONE_OFFSET,
    SP_TRANSFER_RECEPTIVITY, SP_TRANSFER_SELECTIVITY,
    SP_VIRAL_RESISTANCE, SP_LYSO_SUPPRESSION,
)

# ── Weight parameter name map ──────────────────────────
# Built from the comments in constants.py so every exported weight
# is keyed by a human-readable name rather than a bare index.

_MODULE_WEIGHT_NAMES = {
    0:  ["efficiency", "toxic_tolerance", "light_sensitivity", "storage_rate"],
    1:  ["efficiency", "specificity", "saturation_threshold", "gradient_follow"],
    2:  ["prey_selectivity", "handling_efficiency", "decomp_preference", "aggression"],
    3:  ["light_seek", "density_avoid", "nutrient_stay", "chemo_toxic_seek",
         "random_wt", "stay_tend", "light_str", "toxic_response"],
    4:  ["extraction_eff", "resource_discrim", "storage_cap", "cooperative_signal"],
    5:  ["shell", "camouflage", "size_invest", "counter_attack"],
    6:  ["detox_eff", "toxin_tolerance", "conversion_rate", "selective_uptake"],
    7:  [],  # TOXPROD has no evolvable weights
    8:  ["recognition_specificity", "suppression_strength", "resistance_breadth",
         "immune_memory"],
    9:  ["identity_signal", "compatibility_assessment", "approach_avoidance",
         "relationship_strength"],
    10: ["pollination_drive", "route_memory", "network_coordination",
         "reward_sensitivity"],
}

_STANDALONE_WEIGHT_NAMES = [
    "transfer_receptivity", "transfer_selectivity",
    "viral_resistance", "lyso_suppression",
]


def _weight_vector_to_dict(weights_1d):
    """Convert a flat (TOTAL_WEIGHT_PARAMS,) weight array to a nested dict
    keyed by module name -> parameter name -> value."""
    out = {}
    for m in range(N_MODULES):
        off = int(MODULE_WEIGHT_OFFSETS[m])
        sz = int(MODULE_WEIGHT_SIZES[m])
        if sz == 0:
            continue
        names = _MODULE_WEIGHT_NAMES[m]
        mod_dict = {}
        for j, name in enumerate(names):
            mod_dict[name] = round(float(weights_1d[off + j]), 6)
        out[MODULE_NAMES[m]] = mod_dict
    # Standalone params
    sa = {}
    for j, name in enumerate(_STANDALONE_WEIGHT_NAMES):
        sa[name] = round(float(weights_1d[STANDALONE_OFFSET + j]), 6)
    out["standalone"] = sa
    return out


class PivotalMomentMixin:
    """Detects ecologically significant events and exports full-state snapshots."""

    # ── Initialisation (called from World.__init__ via _init_pivotal) ──

    def _init_pivotal(self):
        self._pivotal_events = []          # collected during the run
        self._pivotal_pop_history = []     # (timestep, pop) ring buffer for crash detect
        self._pivotal_peak_diversity = 0   # running max unique-species count
        self._pivotal_peak_module_count = 0  # running max module count on any organism
        self._pivotal_peak_cx_score = 0.0    # running max complexity score
        self._pivotal_merger_flagged = False  # set by _flag_merger, consumed each tick
        self._pivotal_prev_fungal_mean = 0.0
        self._pivotal_prev_pop = 0
        self._pivotal_export_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "exports")
        print(f"  Pivotal moments export dir: {self._pivotal_export_dir}")

    # ── Hook: called from _attempt_endosymbiosis when a merger succeeds ──

    def _flag_merger(self):
        """Flag that an endosymbiotic merger occurred this tick."""
        self._pivotal_merger_flagged = True

    # ── Per-tick detection (called at end of update()) ──

    def _detect_pivotal_moments(self):
        """Run all detectors; any that fire produce an event entry."""
        if self.pop == 0:
            return

        t = self.timestep

        # -- 1. Endosymbiotic merger --
        if self._pivotal_merger_flagged:
            self._pivotal_events.append(self._make_event(t, "endosymbiotic_merger"))
            self._pivotal_merger_flagged = False

        # -- 2. Population crash (>30 % drop within 50 ticks) --
        self._pivotal_pop_history.append((t, self.pop))
        # trim to last 50 ticks
        self._pivotal_pop_history = [
            (tt, p) for tt, p in self._pivotal_pop_history if t - tt <= 50
        ]
        if len(self._pivotal_pop_history) >= 2:
            oldest_pop = self._pivotal_pop_history[0][1]
            if oldest_pop > 0 and self.pop < oldest_pop * 0.7:
                self._pivotal_events.append(self._make_event(
                    t, "population_crash",
                    {"from": oldest_pop, "to": self.pop,
                     "window_ticks": t - self._pivotal_pop_history[0][0]}))
                # reset so we don't fire every tick of a continuing crash
                self._pivotal_pop_history = [(t, self.pop)]

        # -- 3. New high-module species (4+ modules) --
        mod_counts = self.module_present.sum(axis=1)  # per organism
        max_now = int(mod_counts.max()) if self.pop > 0 else 0
        if max_now >= 4 and max_now > self._pivotal_peak_module_count:
            idx = int(np.argmax(mod_counts))
            self._pivotal_events.append(self._make_event(
                t, "new_high_module_species",
                {"module_count": max_now,
                 "modules": [MODULE_NAMES[m] for m in range(N_MODULES)
                             if self.module_present[idx, m]]}))
            self._pivotal_peak_module_count = max_now

        # -- 4. Peak species diversity --
        # "species" = unique module-present signature
        sigs = set()
        for i in range(self.pop):
            sigs.add(tuple(self.module_present[i].tolist()))
        diversity = len(sigs)
        if diversity > self._pivotal_peak_diversity:
            self._pivotal_peak_diversity = diversity
            self._pivotal_events.append(self._make_event(
                t, "peak_species_diversity",
                {"unique_species": diversity}))

        # -- 5. Viral epidemic (>60 % infected) --
        infected_frac = float((self.viral_load > 0).sum()) / max(self.pop, 1)
        if infected_frac > 0.6:
            self._pivotal_events.append(self._make_event(
                t, "viral_epidemic",
                {"infected_fraction": round(infected_frac, 3)}))

        # -- 6. Max complexity organism --
        # complexity = modules_present * (1 + merger_count) + transfer_count
        complexity = (mod_counts.astype(np.float64)
                      * (1 + self.merger_count)
                      + self.transfer_count)
        max_cx = float(complexity.max())
        if max_cx > self._pivotal_peak_cx_score:
            self._pivotal_peak_cx_score = max_cx
            idx = int(np.argmax(complexity))
            self._pivotal_events.append(self._make_event(
                t, "max_complexity_organism",
                {"complexity_score": round(max_cx, 2),
                 "organism_id": int(self.ids[idx])}))

        # -- 7. Fungal network surge after mass death --
        fungal_mean = float(self.fungal_density.mean())
        pop_dropped = (self._pivotal_prev_pop > 0
                       and self.pop < self._pivotal_prev_pop * 0.8)
        fungal_surged = (self._pivotal_prev_fungal_mean > 0
                         and fungal_mean > self._pivotal_prev_fungal_mean * 1.5)
        if pop_dropped and fungal_surged:
            self._pivotal_events.append(self._make_event(
                t, "fungal_surge_after_death",
                {"prev_pop": self._pivotal_prev_pop, "cur_pop": self.pop,
                 "prev_fungal": round(self._pivotal_prev_fungal_mean, 6),
                 "cur_fungal": round(fungal_mean, 6)}))

        self._pivotal_prev_fungal_mean = fungal_mean
        self._pivotal_prev_pop = self.pop

    # ── Snapshot builder ──

    def _make_event(self, timestep, trigger, extra=None):
        """Build a full-state snapshot dict for one pivotal event."""
        snap = {
            "timestep": timestep,
            "trigger": trigger,
            "extra": extra or {},
            "population": self.pop,
            "organisms": self._snapshot_all_organisms(),
            "environment": self._snapshot_environment(),
        }
        return snap

    def _snapshot_all_organisms(self):
        """Export every organism with full detail."""
        orgs = []
        for i in range(self.pop):
            orgs.append({
                "id": int(self.ids[i]),
                "row": int(self.rows[i]),
                "col": int(self.cols[i]),
                "energy": round(float(self.energy[i]), 4),
                "age": int(self.age[i]),
                "generation": int(self.generation[i]),
                "parent_ids": int(self.parent_ids[i]),
                "weights": _weight_vector_to_dict(self.weights[i]),
                "module_present": {MODULE_NAMES[m]: bool(self.module_present[i, m])
                                   for m in range(N_MODULES)},
                "module_active": {MODULE_NAMES[m]: bool(self.module_active[i, m])
                                  for m in range(N_MODULES)},
                "module_usage": {MODULE_NAMES[m]: round(float(self.module_usage[i, m]), 4)
                                 for m in range(N_MODULES)},
                "hijack_intensity": round(float(self.hijack_intensity[i]), 4)
                                    if hasattr(self, "hijack_intensity")
                                       and len(self.hijack_intensity) > i
                                    else 0.0,
                "viral_load": round(float(self.viral_load[i]), 4),
                "lysogenic_strength": round(float(self.lysogenic_strength[i]), 4),
                "merger_count": int(self.merger_count[i]),
                "transfer_count": int(self.transfer_count[i]),
                "immune_experience": round(float(self.immune_experience[i]), 4),
                "relationship_score": round(float(self.relationship_score[i]), 4),
                "genomic_stress": round(float(self.genomic_stress[i]), 4),
                "genomic_cascade_phase": int(self.genomic_cascade_phase[i]),
                "dev_progress": round(float(self.dev_progress[i]), 4),
                "is_mature": bool(self.is_mature[i]),
            })
        return orgs

    def _snapshot_environment(self):
        """Export all environment grids as nested lists."""
        return {
            "toxic": self.toxic.tolist(),
            "nutrients": self.nutrients.tolist(),
            "decomposition": self.decomposition.tolist(),
            "fungal_density": self.fungal_density.tolist(),
            "ecosystem_integrity": self.ecosystem_integrity.tolist(),
            "viral_particles": self.viral_particles.tolist(),
        }

    # ── End-of-run export ──

    def _export_run_index(self):
        """Write all collected pivotal events to a JSON index file."""
        if not self._pivotal_events:
            print(f"  Pivotal moments: 0 events detected, nothing to export.")
            return
        out_dir = self._pivotal_export_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "pivotal_moments.json")
        with open(path, "w") as f:
            json.dump({
                "total_events": len(self._pivotal_events),
                "triggers": [e["trigger"] for e in self._pivotal_events],
                "events": self._pivotal_events,
            }, f, indent=2)
        print(f"  Pivotal moments: {len(self._pivotal_events)} events -> {path}")
