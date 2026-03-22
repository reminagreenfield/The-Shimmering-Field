"""Ecosystem integrity, collapse dynamics, fungal networks."""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

from .constants import *


class CollapseMixin:
    """Ecosystem integrity, collapse dynamics, fungal networks."""

    def _update_ecosystem_integrity(self):
        """Compute per-cell ecosystem integrity from ACTIVE ecosystem signals.
        Background nutrients/fungi don't prop up integrity — only living activity."""
        from scipy.ndimage import uniform_filter
        c = self.cfg
        N = c.grid_size

        # Active signals — things requiring living organisms to maintain
        pop_signal = np.minimum(self.density.astype(np.float64) / 3.0, 1.0)
        med_signal = np.minimum(self.mediator_field / 0.3, 1.0)
        # Diversity: mix of producers and consumers in area
        producer_dens = np.zeros((N, N))
        consumer_dens = np.zeros((N, N))
        if self.pop > 0:
            has_prod = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
            has_cons = self.module_active[:, M_CONSUME]
            pidx = np.where(has_prod)[0]
            cidx = np.where(has_cons)[0]
            if len(pidx) > 0:
                np.add.at(producer_dens, (self.rows[pidx], self.cols[pidx]), 1)
            if len(cidx) > 0:
                np.add.at(consumer_dens, (self.rows[cidx], self.cols[cidx]), 1)
        diversity_signal = np.minimum(producer_dens, 1.0) * np.minimum(consumer_dens + 0.1, 1.0)
        fungal_signal = np.minimum(self.fungal_density / 2.0, 1.0)

        raw_integrity = (pop_signal * 0.45 + med_signal * 0.2 +
                        diversity_signal * 0.2 + fungal_signal * 0.15)

        # Negative signals REDUCE integrity (toxic stress, viral load)
        toxic_penalty = np.minimum(self.toxic / 0.5, 1.0) * 0.3  # toxic > 0.5 → significant penalty
        viral_penalty = np.minimum(self.viral_particles / 0.3, 1.0) * 0.15
        raw_integrity -= toxic_penalty
        raw_integrity -= viral_penalty
        np.clip(raw_integrity, 0, 1.0, out=raw_integrity)

        # Empty cells = pristine, not collapsed
        habitated = self.density > 0
        raw_integrity[~habitated] = np.maximum(raw_integrity[~habitated], c.recovery_threshold + 0.1)

        self.ecosystem_integrity = uniform_filter(raw_integrity, size=7, mode='wrap')

    def _update_collapse_state(self):
        """Apply hysteresis: collapse when below threshold, recover only above recovery threshold.
        Only applies to cells that have been populated (habitated zones)."""
        c = self.cfg
        # Only collapse zones that have had organisms (density > 0 or recently had)
        habitated = self.density > 0
        # Expand habitated zone slightly (organisms affect neighbors)
        from scipy.ndimage import maximum_filter
        habitated = maximum_filter(habitated.astype(np.float64), size=5) > 0

        # New collapses (only in habitated zones)
        new_collapse = (self.ecosystem_integrity < c.collapse_threshold) & habitated & ~self.zone_collapsed
        self.zone_collapsed |= new_collapse
        # Recovery (must exceed HIGHER threshold — hysteresis)
        recovering = (self.ecosystem_integrity > c.recovery_threshold) & self.zone_collapsed
        self.zone_collapsed &= ~recovering

    def _collapse_reproduction_modifier(self):
        """Sigmoid reproduction penalty for organisms in collapsed zones."""
        c = self.cfg
        if self.pop == 0:
            return np.ones(0)
        in_collapsed = self.zone_collapsed[self.rows, self.cols]
        modifier = np.ones(self.pop, dtype=np.float64)
        if in_collapsed.any():
            # Steep sigmoid: drops from 1.0 to (1-penalty) as integrity decreases
            local_int = self.ecosystem_integrity[self.rows, self.cols]
            # Below threshold: penalty scales with how far below
            deficit = np.maximum(0, c.collapse_threshold - local_int[in_collapsed]) / c.collapse_threshold
            sigmoid_penalty = 1.0 / (1.0 + np.exp(-c.collapse_sigmoid_steepness * (deficit - 0.3)))
            modifier[in_collapsed] = 1.0 - c.collapse_repro_penalty * sigmoid_penalty
        return modifier

    def _compute_compounding_stress(self):
        """Multiplicative stress from overlapping failure modes.
        Returns per-organism multiplier (>= 1.0) applied to energy costs."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return np.ones(0)

        stress_mult = np.ones(n, dtype=np.float64)
        # Count active failure modes per organism
        local_toxic = self.toxic[self.rows, self.cols]
        f1 = local_toxic > 0.3                              # toxic stress
        f2 = self.genomic_cascade_phase > 0                  # genomic cascade
        f3 = self.hijack_intensity > 0.1                     # behavioral hijack
        f4 = (self.age > c.dev_window_length) & ~self.is_mature  # developmental compromise
        f5 = self.zone_collapsed[self.rows, self.cols]       # ecosystem collapse

        failure_count = (f1.astype(np.float64) + f2.astype(np.float64) +
                        f3.astype(np.float64) + f4.astype(np.float64) +
                        f5.astype(np.float64))

        # Each overlapping failure multiplies stress
        stress_mult = np.power(c.compounding_base, failure_count)
        return stress_mult

    # ── Fungal Networks ──

    def _update_fungal_network(self):
        """Grow, spread, and decay fungal network. Transport nutrients and genome fragments."""
        c = self.cfg

        # Growth: fungi feed on decomposition
        growth_mask = self.decomposition > c.fungal_decomp_threshold
        growth = np.zeros_like(self.fungal_density)
        growth[growth_mask] = (c.fungal_growth_rate *
                               np.minimum(self.decomposition[growth_mask] / 5.0, 1.0))

        # Surge growth during mass death events (high decomposition)
        surge_mask = self.decomposition > c.fungal_surge_threshold
        growth[surge_mask] *= c.fungal_surge_mult

        # Growth also consumes decomposition (fungi eat dead material)
        consumed = growth * 0.5
        self.decomposition -= consumed
        np.clip(self.decomposition, 0, 30.0, out=self.decomposition)

        self.fungal_density += growth

        # Spread: mycelial diffusion to neighbors
        diffused = gaussian_filter(self.fungal_density, sigma=1.5, mode='wrap')
        self.fungal_density = (1.0 - c.fungal_diffusion_rate) * self.fungal_density + \
                              c.fungal_diffusion_rate * diffused

        # Decay: maintenance cost
        self.fungal_density -= c.fungal_decay_rate
        np.clip(self.fungal_density, 0, 5.0, out=self.fungal_density)

    def _fungal_nutrient_transport(self):
        """Redistribute nutrients along fungal network — from high-decomp to low-decomp areas."""
        c = self.cfg
        if self.fungal_density.max() < 0.01:
            return

        # Nutrient transport: fungi move nutrients toward low-nutrient zones
        avg_nutrients = gaussian_filter(self.nutrients, sigma=3.0, mode='wrap')
        gradient = avg_nutrients - self.nutrients  # positive = local deficit
        transport = c.fungal_nutrient_transport * self.fungal_density * gradient
        self.nutrients += transport
        np.clip(self.nutrients, 0, 3.0, out=self.nutrients)

    def _fungal_toxic_conduit(self):
        """Fungal networks accelerate toxic diffusion along their paths (conduit effect)."""
        c = self.cfg
        if self.fungal_density.max() < 0.01:
            return

        # Extra toxic spread along fungal conduits
        fungal_boost = c.fungal_toxic_conduit * self.fungal_density
        toxic_gradient = gaussian_filter(self.toxic, sigma=2.0, mode='wrap') - self.toxic
        self.toxic += fungal_boost * toxic_gradient
        np.clip(self.toxic, 0, None, out=self.toxic)

    # ── Main Loop ──

