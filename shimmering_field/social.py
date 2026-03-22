"""Social interactions, mediators, and development."""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

from .constants import *


class SocialMixin:
    """Social interactions, mediators, and development."""

    def _update_social_field(self):
        """Deposit social identity signals on the grid.
        Channel 0: producer-type signal (PHOTO/CHEMO organisms)
        Channel 1: consumer-type signal (CONSUME organisms)
        Signal strength modulated by SOCIAL module if present."""
        c = self.cfg
        n = self.pop
        self.social_field *= 0.5  # decay old signals

        if n == 0:
            return

        has_social = self.module_active[:, M_SOCIAL]
        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]

        # Base signal: all organisms emit weak identity signal
        base_strength = np.ones(n) * 0.3

        # SOCIAL module amplifies signal
        if has_social.any():
            sw = self._module_weights(M_SOCIAL)
            signal_strength = 1.0 / (1.0 + np.exp(-sw[:, 0]))  # identity_signal weight
            base_strength = np.where(has_social, 0.3 + 0.7 * signal_strength, base_strength)

        # Deposit producer signal
        prod_signal = base_strength * has_producer
        if prod_signal.any():
            np.add.at(self.social_field[:, :, 0], (self.rows, self.cols), prod_signal)

        # Deposit consumer signal
        cons_signal = base_strength * has_consume
        if cons_signal.any():
            np.add.at(self.social_field[:, :, 1], (self.rows, self.cols), cons_signal)

        np.clip(self.social_field, 0, 10.0, out=self.social_field)

    def _apply_social_interactions(self):
        """Compatible neighbors build relationship; SOCIAL module accelerates it.
        ALL organisms accumulate relationship from co-location with complementary types.
        SOCIAL holders get 5x faster growth + energy bonus."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]

        # Local social signals at each organism's cell
        local_prod = self.social_field[self.rows, self.cols, 0]
        local_cons = self.social_field[self.rows, self.cols, 1]

        # Compatibility: producers benefit from nearby consumers, consumers from producers
        compatible_signal = np.zeros(n)
        compatible_signal += has_producer * local_cons  # producers near consumers
        compatible_signal += has_consume * local_prod   # consumers near producers
        compatible_signal += has_producer * local_prod * 0.3  # same-type mild benefit
        compatible_signal += has_consume * local_cons * 0.2

        # Also: raw co-location signal (even without social field, density = proximity)
        local_dens = self.density[self.rows, self.cols].astype(np.float64)
        colocation_signal = np.minimum(local_dens - 1, 3.0)  # neighbors
        colocation_signal = np.maximum(colocation_signal, 0)

        # SOCIAL module holders: energy bonus + accelerated relationship
        has_social = self.module_active[:, M_SOCIAL]
        if has_social.any():
            sw = self._module_weights(M_SOCIAL)
            compat_skill = 1.0 / (1.0 + np.exp(-sw[:, 1]))
            rel_strength = 1.0 / (1.0 + np.exp(-sw[:, 3]))
            bonus = (c.social_compatibility_bonus * compat_skill
                    * np.minimum(compatible_signal, 3.0) * has_social)
            self.energy += bonus

        # Relationship: ALL organisms grow from compatible co-location
        base_growth = c.social_relationship_growth * 0.6  # meaningful baseline for everyone
        base_growth_arr = base_growth * np.minimum(compatible_signal + colocation_signal * 0.3, 5.0)

        # Sessile organisms build relationships faster (permanent co-location)
        has_move_arr = self.module_active[:, M_MOVE]
        sessile_mult = np.where(has_move_arr, 1.0, 3.0)
        base_growth_arr *= sessile_mult

        # SOCIAL holders get 5x growth rate
        if has_social.any():
            sw = self._module_weights(M_SOCIAL)
            rel_strength = 1.0 / (1.0 + np.exp(-sw[:, 3]))
            social_mult = 5.0 * rel_strength * has_social
            base_growth_arr *= (1.0 + social_mult)

        self.relationship_score *= 0.999  # slower decay (was 0.998)
        self.relationship_score += base_growth_arr
        np.clip(self.relationship_score, 0, 10.0, out=self.relationship_score)

    # ── Mediator System (Pollination/Dispersal) ──

    def _update_mediator_field(self):
        """Mediators deposit pollination service signal on the grid.
        Higher pollination_drive = stronger signal = more reproduction facilitation."""
        c = self.cfg
        n = self.pop
        self.mediator_field *= c.mediate_network_decay

        if n == 0:
            return

        has_mediate = self.module_active[:, M_MEDIATE]
        if not has_mediate.any():
            return

        mw = self._module_weights(M_MEDIATE)
        poll_drive = 1.0 / (1.0 + np.exp(-mw[:, 0]))  # pollination_drive
        network_coord = 1.0 / (1.0 + np.exp(-mw[:, 2]))  # network_coordination

        # Signal strength: base + coordination bonus when multiple mediators nearby
        local_med = self.mediator_field[self.rows, self.cols]
        signal = poll_drive * (1.0 + network_coord * np.minimum(local_med, 3.0) * 0.3)
        signal *= has_mediate

        np.add.at(self.mediator_field, (self.rows, self.cols), signal)
        np.clip(self.mediator_field, 0, 10.0, out=self.mediator_field)

        # Passive reward: mediators near immature organisms earn energy (developmental support)
        midx = np.where(has_mediate)[0]
        if len(midx) > 0 and (~self.is_mature).any():
            imm_dens = np.zeros((c.grid_size, c.grid_size), dtype=np.float64)
            imm_idx = np.where(~self.is_mature)[0]
            np.add.at(imm_dens, (self.rows[imm_idx], self.cols[imm_idx]), 1.0)
            # Smooth to cover mediate_radius
            imm_dens = gaussian_filter(imm_dens, sigma=c.mediate_radius, mode='wrap')
            local_imm = imm_dens[self.rows[midx], self.cols[midx]]
            self.energy[midx] += c.mediate_passive_reward * np.minimum(local_imm, 3.0)

    # ── Nutrient Cycling ──

    def _update_development(self):
        """Update developmental progress. Immature organisms near compatible
        neighbors gain progress. Window expiration → permanent compromise."""
        c = self.cfg
        n = self.pop
        if n == 0:
            return

        immature = ~self.is_mature
        if not immature.any():
            return

        iidx = np.where(immature)[0]

        # Check for compatible neighbors in same cell
        # Compatibility: producers benefit from nearby consumers, consumers from producers
        has_prod = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]
        has_cons = self.module_active[:, M_CONSUME]

        # Build per-cell producer and consumer density
        prod_density = np.zeros((c.grid_size, c.grid_size), dtype=np.float64)
        cons_density = np.zeros((c.grid_size, c.grid_size), dtype=np.float64)
        pidx_prod = np.where(has_prod)[0]
        pidx_cons = np.where(has_cons)[0]
        if len(pidx_prod) > 0:
            np.add.at(prod_density, (self.rows[pidx_prod], self.cols[pidx_prod]), 1.0)
        if len(pidx_cons) > 0:
            np.add.at(cons_density, (self.rows[pidx_cons], self.cols[pidx_cons]), 1.0)

        # Immature producers need consumers nearby, immature consumers need producers
        is_prod_imm = has_prod[iidx]
        is_cons_imm = has_cons[iidx]

        local_cons = cons_density[self.rows[iidx], self.cols[iidx]]
        local_prod = prod_density[self.rows[iidx], self.cols[iidx]]

        # Progress: compatible neighbors present → progress increases
        # Cross-type: producers near consumers (fast)
        # Same-type: conspecific support (0.4x rate — species develop from own kind too)
        local_med = self.mediator_field[self.rows[iidx], self.cols[iidx]]
        compat_present = np.zeros(len(iidx), dtype=np.float64)
        # Cross-type (full rate)
        compat_present += is_prod_imm * np.minimum(local_cons, 3.0) / 3.0
        compat_present += is_cons_imm * np.minimum(local_prod, 3.0) / 3.0
        # Same-type conspecific (0.6x — organisms develop from own kind)
        compat_present += is_prod_imm * np.minimum(local_prod, 3.0) / 3.0 * 0.6
        compat_present += is_cons_imm * np.minimum(local_cons, 3.0) / 3.0 * 0.6
        # Mediators help everyone develop
        compat_present += np.minimum(local_med, 1.0) * 0.5

        # Raw co-location: any neighbor helps
        local_dens = self.density[self.rows[iidx], self.cols[iidx]].astype(np.float64)
        compat_present += np.minimum(local_dens - 1, 3.0) * 0.25

        # Solo development extremely slow — isolation is dangerous
        compat_present = np.maximum(compat_present, 0.05)

        self.dev_progress[iidx] += c.dev_progress_rate * compat_present

        # Maturation check
        new_mature = iidx[self.dev_progress[iidx] >= 1.0]
        self.is_mature[new_mature] = True
        self.dev_progress[new_mature] = 1.0

    def _compute_dev_compromise(self):
        """Return per-organism energy multiplier from developmental compromise.
        Mature organisms: 1.0. Compromised (past window, not mature): < 1.0."""
        c = self.cfg
        n = self.pop
        mult = np.ones(n, dtype=np.float64)
        if n == 0:
            return mult

        # Past developmental window and not mature → compromised
        past_window = (self.age > c.dev_window_length) & ~self.is_mature
        if past_window.any():
            deficit = 1.0 - np.minimum(self.dev_progress[past_window], 1.0)
            mult[past_window] = 1.0 - deficit * c.dev_compromise_energy

        return mult

    # ── Nonlinear Collapse Dynamics ──

