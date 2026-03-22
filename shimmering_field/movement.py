"""Movement decisions and execution."""

import numpy as np

from .constants import *


class MovementMixin:
    """Movement decisions and execution."""

    def _decide_movement(self, readings):
        n = self.pop
        if n == 0:
            return np.array([], dtype=np.int32)

        has_move = self.module_active[:, M_MOVE]
        has_photo = self.module_active[:, M_PHOTO]
        has_chemo = self.module_active[:, M_CHEMO]
        has_consume = self.module_active[:, M_CONSUME]
        mp = self._module_weights(M_MOVE)

        sc = np.zeros((n, 5))

        # Stay score: nutrient attraction + decomp attraction for consumers
        sc[:, 0] = (mp[:, 5] + mp[:, 2] * readings[:, 2]
                    - mp[:, 1] * np.minimum(readings[:, 3], 10) * 0.1)
        if has_consume.any():
            sc[:, 0] += has_consume * readings[:, 8] * 0.3

        # Gradient weights
        light_w = mp[:, 6] * has_photo.astype(np.float64)
        net_toxic = mp[:, 7] * has_photo.astype(np.float64) - mp[:, 3] * has_chemo.astype(np.float64)

        # Consumer-specific: follow PREY SCENT (not generic density)
        # Herbivores follow producer_scent, Carnivores follow animal_scent
        prey_scent_gy = np.zeros(n)
        prey_scent_gx = np.zeros(n)
        if has_consume.any():
            uw = self._module_weights(M_CONSUME)
            dp = 1.0 / (1.0 + np.exp(-uw[:, 2]))   # decomp_pref
            ps = 1.0 / (1.0 + np.exp(-uw[:, 0]))   # prey_selectivity: low=herb, high=carn

            # Herbivores (low ps) follow producer scent
            # Carnivores (high ps) follow animal scent
            # Blend based on prey_selectivity
            herb_gy = readings[:, 22] * (1.0 - ps)  # producer_scent_grad
            herb_gx = readings[:, 23] * (1.0 - ps)
            carn_gy = readings[:, 24] * ps           # animal_scent_grad
            carn_gx = readings[:, 25] * ps

            # Strong prey-seeking drive — this is the key to consumer viability
            prey_drive = has_consume * (1.0 - dp) * 12.0  # detritivores don't hunt, low dp = hunter
            prey_scent_gy = prey_drive * (herb_gy + carn_gy)
            prey_scent_gx = prey_drive * (herb_gx + carn_gx)

        # Detritivore scent following (existing)
        consume_scent_w = np.zeros(n)
        if has_consume.any():
            uw = self._module_weights(M_CONSUME)
            dp = 1.0 / (1.0 + np.exp(-uw[:, 2]))
            consume_scent_w = has_consume * dp * 8.0

        # FORAGE: seek nutrient-rich areas
        has_forage = self.module_active[:, M_FORAGE]
        forage_nutr_w = np.zeros(n)
        if has_forage.any():
            fw = self._module_weights(M_FORAGE)
            resource_discrim = 1.0 / (1.0 + np.exp(-fw[:, 1]))
            forage_nutr_w = has_forage * resource_discrim * 2.0

        # DETOX: seek toxic areas (food source)
        has_detox = self.module_active[:, M_DETOX]
        detox_toxic_w = np.zeros(n)
        if has_detox.any():
            dtw = self._module_weights(M_DETOX)
            detox_seek = 1.0 / (1.0 + np.exp(-dtw[:, 0]))
            detox_toxic_w = has_detox * detox_seek * 1.5

        # SOCIAL: seek compatible organisms via social signal gradients
        has_social = self.module_active[:, M_SOCIAL]
        social_compat_gy = np.zeros(n)
        social_compat_gx = np.zeros(n)
        if has_social.any():
            sw = self._module_weights(M_SOCIAL)
            approach = np.tanh(sw[:, 2])  # approach_avoidance: positive = approach
            has_producer = self.module_active[:, M_PHOTO] | self.module_active[:, M_CHEMO]

            # Producers follow consumer signal gradients (seek metabolic complement)
            # Consumers follow producer signal gradients
            # readings 17-20: social gradients
            spg_y, spg_x = readings[:, 17], readings[:, 18]  # producer signal gradients
            scg_y, scg_x = readings[:, 19], readings[:, 20]  # consumer signal gradients

            # Each organism seeks the complementary type
            compat_gy = np.where(has_producer, scg_y, spg_y) * has_social
            compat_gx = np.where(has_producer, scg_x, spg_x) * has_social
            # Also mild attraction to same-type (cooperative)
            same_gy = np.where(has_producer, spg_y, scg_y) * has_social * 0.3
            same_gx = np.where(has_producer, spg_x, scg_x) * has_social * 0.3

            social_compat_gy = approach * (compat_gy + same_gy) * 1.5
            social_compat_gx = approach * (compat_gx + same_gx) * 1.5

        # MEDIATE: seek organism-dense areas (go where pollination is needed)
        has_mediate = self.module_active[:, M_MEDIATE]
        mediate_dens_gy = np.zeros(n)
        mediate_dens_gx = np.zeros(n)
        if has_mediate.any():
            mew = self._module_weights(M_MEDIATE)
            route_mem = 1.0 / (1.0 + np.exp(-mew[:, 1]))  # route_memory
            # Mediators seek social signal gradients (where organisms are)
            # Combined producer+consumer social signal as target
            spg_y, spg_x = readings[:, 17], readings[:, 18]
            scg_y, scg_x = readings[:, 19], readings[:, 20]
            mediate_dens_gy = has_mediate * (spg_y + scg_y) * (0.5 + route_mem) * 2.0
            mediate_dens_gx = has_mediate * (spg_x + scg_x) * (0.5 + route_mem) * 2.0

        # Directional scores
        lg_y, lg_x = readings[:, 4], readings[:, 5]
        tg_y, tg_x = readings[:, 6], readings[:, 7]
        dg_y, dg_x = readings[:, 9], readings[:, 10]
        sg_y, sg_x = readings[:, 11], readings[:, 12]
        ng_y, ng_x = readings[:, 13], readings[:, 14]

        gy = (light_w * lg_y - net_toxic * tg_y + prey_scent_gy
              + consume_scent_w * sg_y + forage_nutr_w * ng_y + detox_toxic_w * tg_y
              + social_compat_gy + mediate_dens_gy)
        gx = (light_w * lg_x - net_toxic * tg_x + prey_scent_gx
              + consume_scent_w * sg_x + forage_nutr_w * ng_x + detox_toxic_w * tg_x
              + social_compat_gx + mediate_dens_gx)
        sc[:, 1] = gy;  sc[:, 2] = -gy
        sc[:, 3] = gx;  sc[:, 4] = -gx

        # Behavioral hijacking: override movement for infected organisms
        hijacked = self.hijack_intensity > 0.1
        if hijacked.any():
            hidx = np.where(hijacked)[0]
            hi = self.hijack_intensity[hidx]
            # Hijacked organisms seek high-density, low-viral areas (spread virus to new hosts)
            hijack_gy = (readings[hidx, 9] * self.cfg.hijack_density_seek   # toward density
                        - readings[hidx, 6] * 1.0)                    # away from existing viral zones
            hijack_gx = (readings[hidx, 10] * self.cfg.hijack_density_seek
                        - readings[hidx, 7] * 1.0)
            # Blend: hijack overrides host movement proportional to intensity
            sc[hidx, 0] *= (1.0 - hi * 0.8)  # suppress stay tendency
            sc[hidx, 1] = sc[hidx, 1] * (1.0 - hi) + hijack_gy * hi
            sc[hidx, 2] = sc[hidx, 2] * (1.0 - hi) - hijack_gy * hi
            sc[hidx, 3] = sc[hidx, 3] * (1.0 - hi) + hijack_gx * hi
            sc[hidx, 4] = sc[hidx, 4] * (1.0 - hi) - hijack_gx * hi
            # Heavy hijack: erratic movement noise
            heavy = hi > 0.7
            if heavy.any():
                hh = hidx[heavy]
                sc[hh, 1:] += self.rng.normal(0, 2.0, (len(hh), 4))

        # Exploration noise
        sc += mp[:, 4:5] * self.rng.normal(0, 0.3, (n, 5))

        # Sessile organisms stay
        sc[~has_move, 1:] = -999.0

        return np.argmax(sc, axis=1)

    def _execute_movement(self, actions):
        N = self.cfg.grid_size
        m = actions == 1; self.rows[m] = np.maximum(0, self.rows[m] - 1)
        m = actions == 2; self.rows[m] = np.minimum(N - 1, self.rows[m] + 1)
        m = actions == 3; self.cols[m] = np.minimum(N - 1, self.cols[m] + 1)
        m = actions == 4; self.cols[m] = np.maximum(0, self.cols[m] - 1)
        self.energy[actions > 0] -= self.cfg.energy_movement_cost

    # ── Horizontal Transfer ──

