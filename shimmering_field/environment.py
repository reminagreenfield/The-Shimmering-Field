"""Environment: toxicity, nutrients, decomposition, density, sensing."""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

from .constants import *


class EnvironmentMixin:
    """Environment: toxicity, nutrients, decomposition, density, sensing."""

    def _update_environment(self):
        c = self.cfg

        # Toxic diffusion + decay
        p = np.pad(self.toxic, 1, mode='edge')
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        self.toxic += c.toxic_diffusion_rate * (nb - self.toxic)
        self.toxic *= (1.0 - c.toxic_decay_rate / np.maximum(0.3, self.zone_map))

        # Toxic production by organisms
        if self.pop > 0:
            has_toxprod = self.module_active[:, M_TOXPROD]
            rates = np.where(has_toxprod, c.toxic_production_rate, 0.0)
            np.add.at(self.toxic, (self.rows, self.cols),
                      rates * self.zone_map[self.rows, self.cols])

        # Nutrients: slow regeneration + very slow decomp→nutrient conversion
        self.nutrients += c.nutrient_base_rate
        xfer = self.decomposition * 0.001
        self.nutrients += xfer
        self.decomposition -= xfer

        # Decomp decay
        self.decomposition *= c.decomp_decay_rate

        # Passive litter: producers shed organic material constantly (leaf litter, dead roots)
        # This provides baseline food for detritivores independent of death events
        if self.pop > 0:
            has_photo = self.module_present[:, M_PHOTO] & ~self.module_present[:, M_CONSUME]
            prod_idx = np.where(has_photo)[0]
            if len(prod_idx) > 0:
                np.add.at(self.decomposition, (self.rows[prod_idx], self.cols[prod_idx]), 0.06)

        # Decomp local diffusion: spreads food 2-3 cells from death sites
        if self.timestep % c.decomp_diffusion_interval == 0:
            dp = np.pad(self.decomposition, 1, mode='edge')
            dnb = (dp[:-2, 1:-1] + dp[2:, 1:-1] + dp[1:-1, :-2] + dp[1:-1, 2:]) / 4.0
            self.decomposition += c.decomp_diffusion_rate * (dnb - self.decomposition)

        # Scent layer: gaussian blur for detritivore navigation (cached)
        if self.timestep % c.decomp_scent_interval == 0:
            self.decomp_scent = gaussian_filter(
                self.decomposition, sigma=c.decomp_scent_sigma, mode='constant')

        # Stratified sedimentation
        self._update_strata()

        # Viral diffusion + decay
        self._update_viral_environment()

        # Clamp
        np.clip(self.toxic, 0, 5.0, out=self.toxic)
        np.clip(self.nutrients, 0, c.nutrient_max, out=self.nutrients)
        np.clip(self.decomposition, 0, 30.0, out=self.decomposition)
        np.clip(self.viral_particles, 0, 20.0, out=self.viral_particles)

    def _update_strata(self):
        c = self.cfg
        # Recent → Intermediate
        xfer_w = self.strata_weight["recent"] * c.sedimentation_rate_recent
        self.strata_weight["recent"] -= xfer_w
        new_iw = self.strata_weight["intermediate"] + xfer_w
        blend = xfer_w / np.maximum(new_iw, 1e-8)
        for key in ("strata_pool", "strata_modules"):
            pool = getattr(self, key)
            pool["intermediate"] = (pool["intermediate"] * (1.0 - blend[:, :, None])
                                   + pool["recent"] * blend[:, :, None])
        self.strata_weight["intermediate"] = new_iw

        # Intermediate → Ancient
        xfer_w2 = self.strata_weight["intermediate"] * c.sedimentation_rate_intermediate
        self.strata_weight["intermediate"] -= xfer_w2
        new_aw = self.strata_weight["ancient"] + xfer_w2
        blend2 = xfer_w2 / np.maximum(new_aw, 1e-8)
        for key in ("strata_pool", "strata_modules"):
            pool = getattr(self, key)
            pool["ancient"] = (pool["ancient"] * (1.0 - blend2[:, :, None])
                              + pool["intermediate"] * blend2[:, :, None])
        self.strata_weight["ancient"] = new_aw

        # Decay
        self.strata_weight["recent"] *= (1.0 - c.decomp_fragment_decay)
        self.strata_weight["intermediate"] *= (1.0 - c.decomp_fragment_decay * 0.5)
        self.strata_weight["ancient"] *= (1.0 - c.ancient_decay_rate)

        # Fragment diffusion (periodic)
        if self.timestep % 10 == 0:
            k = c.decomp_fragment_diffusion * 10
            for sname in ("recent", "intermediate", "ancient"):
                sw = self.strata_weight[sname]
                if sw.max() < 0.001:
                    continue
                pw = np.pad(sw, 1, mode='edge')
                wn = (pw[:-2, 1:-1] + pw[2:, 1:-1] + pw[1:-1, :-2] + pw[1:-1, 2:]) / 4.0
                self.strata_weight[sname] += k * (wn - sw)
                self.strata_weight[sname] = np.maximum(self.strata_weight[sname], 0.0)
                sp = self.strata_pool[sname]
                pp = np.pad(sp, ((1, 1), (1, 1), (0, 0)), mode='edge')
                gn = (pp[:-2, 1:-1, :] + pp[2:, 1:-1, :] + pp[1:-1, :-2, :] + pp[1:-1, 2:, :]) / 4.0
                self.strata_pool[sname] += k * (gn - sp)

    def _update_viral_environment(self):
        c = self.cfg
        self.viral_particles *= (1.0 - c.viral_decay_rate)
        pv = np.pad(self.viral_particles, 1, mode='edge')
        vn = (pv[:-2, 1:-1] + pv[2:, 1:-1] + pv[1:-1, :-2] + pv[1:-1, 2:]) / 4.0
        self.viral_particles += c.viral_diffusion_rate * (vn - self.viral_particles)
        self.viral_particles = np.maximum(self.viral_particles, 0.0)

        if self.timestep % 10 == 0 and self.viral_genome_weight.max() > 0.001:
            self.viral_genome_weight *= (1.0 - c.viral_decay_rate * 10)
            self.viral_genome_weight = np.maximum(self.viral_genome_weight, 0.0)
            pp = np.pad(self.viral_genome_pool, ((1, 1), (1, 1), (0, 0)), mode='edge')
            gn = (pp[:-2, 1:-1, :] + pp[2:, 1:-1, :] + pp[1:-1, :-2, :] + pp[1:-1, 2:, :]) / 4.0
            self.viral_genome_pool += c.viral_diffusion_rate * (gn - self.viral_genome_pool)

    def _update_density(self):
        self.density[:] = 0
        if self.pop > 0:
            np.add.at(self.density, (self.rows, self.cols), 1)

    def _update_prey_scent(self):
        """Build diffused scent fields: producers emit producer_scent, animals emit animal_scent.
        Predators follow scent gradients to navigate toward prey."""
        c = self.cfg
        n = self.pop
        if n == 0:
            self.producer_scent *= c.prey_scent_decay
            self.animal_scent *= c.prey_scent_decay
            return

        has_prod = self.module_present[:, M_PHOTO] | self.module_present[:, M_CHEMO]
        has_cons = self.module_present[:, M_CONSUME]

        # Deposit scent at organism locations
        self.producer_scent *= c.prey_scent_decay
        self.animal_scent *= c.prey_scent_decay

        prod_idx = np.where(has_prod & ~has_cons)[0]
        if len(prod_idx) > 0:
            np.add.at(self.producer_scent, (self.rows[prod_idx], self.cols[prod_idx]),
                      c.prey_scent_deposit)

        anim_idx = np.where(has_cons)[0]
        if len(anim_idx) > 0:
            np.add.at(self.animal_scent, (self.rows[anim_idx], self.cols[anim_idx]),
                      c.prey_scent_deposit)

        # Diffuse via gaussian blur (every 3 steps for performance)
        if self.timestep % 3 == 0:
            self.producer_scent = gaussian_filter(
                self.producer_scent, sigma=c.prey_scent_sigma, mode='wrap')
            self.animal_scent = gaussian_filter(
                self.animal_scent, sigma=c.prey_scent_sigma, mode='wrap')

        # Cap to prevent runaway accumulation
        np.clip(self.producer_scent, 0, 20.0, out=self.producer_scent)
        np.clip(self.animal_scent, 0, 20.0, out=self.animal_scent)

    # ── Sensing ──

    def _sense_local(self):
        rows, cols = self.rows, self.cols
        N = self.cfg.grid_size
        ru = np.clip(rows - 1, 0, N - 1)
        rd = np.clip(rows + 1, 0, N - 1)
        cl = np.clip(cols - 1, 0, N - 1)
        cr = np.clip(cols + 1, 0, N - 1)
        return np.column_stack([
            self.light[rows, cols],                                          # 0: local_light
            self.toxic[rows, cols],                                          # 1: local_toxic
            self.nutrients[rows, cols],                                      # 2: local_nutrients
            self.density[rows, cols].astype(np.float64),                     # 3: local_density
            self.light[ru, cols] - self.light[rd, cols],                     # 4: light_grad_y
            self.light[rows, cr] - self.light[rows, cl],                     # 5: light_grad_x
            self.toxic[ru, cols] - self.toxic[rd, cols],                     # 6: toxic_grad_y
            self.toxic[rows, cr] - self.toxic[rows, cl],                     # 7: toxic_grad_x
            self.decomposition[rows, cols],                                  # 8: local_decomp (raw)
            self.density[ru, cols].astype(np.float64)
                - self.density[rd, cols].astype(np.float64),                 # 9: density_grad_y
            self.density[rows, cr].astype(np.float64)
                - self.density[rows, cl].astype(np.float64),                 # 10: density_grad_x
            self.decomp_scent[ru, cols] - self.decomp_scent[rd, cols],       # 11: scent_grad_y
            self.decomp_scent[rows, cr] - self.decomp_scent[rows, cl],       # 12: scent_grad_x
            self.nutrients[ru, cols] - self.nutrients[rd, cols],              # 13: nutrient_grad_y
            self.nutrients[rows, cr] - self.nutrients[rows, cl],              # 14: nutrient_grad_x
            self.social_field[rows, cols, 0],                                 # 15: social_prod_signal
            self.social_field[rows, cols, 1],                                 # 16: social_cons_signal
            self.social_field[ru, cols, 0] - self.social_field[rd, cols, 0],  # 17: social_prod_grad_y
            self.social_field[rows, cr, 0] - self.social_field[rows, cl, 0],  # 18: social_prod_grad_x
            self.social_field[ru, cols, 1] - self.social_field[rd, cols, 1],  # 19: social_cons_grad_y
            self.social_field[rows, cr, 1] - self.social_field[rows, cl, 1],  # 20: social_cons_grad_x
            self.mediator_field[rows, cols],                                   # 21: local_mediator
            # NEW: prey scent gradients for predator navigation
            self.producer_scent[ru, cols] - self.producer_scent[rd, cols],    # 22: producer_scent_grad_y
            self.producer_scent[rows, cr] - self.producer_scent[rows, cl],   # 23: producer_scent_grad_x
            self.animal_scent[ru, cols] - self.animal_scent[rd, cols],       # 24: animal_scent_grad_y
            self.animal_scent[rows, cr] - self.animal_scent[rows, cl],       # 25: animal_scent_grad_x
        ])

    # ── Energy Acquisition ──

