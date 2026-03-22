"""Module definitions, weights, costs, and constants."""

import numpy as np

# Module Definitions
# ─────────────────────────────────────────────────────

M_PHOTO   = 0
M_CHEMO   = 1
M_CONSUME = 2
M_MOVE    = 3
M_FORAGE  = 4
M_DEFENSE = 5
M_DETOX   = 6
M_TOXPROD = 7
M_VRESIST = 8
M_SOCIAL  = 9
M_MEDIATE = 10

N_MODULES = 11

MODULE_NAMES = [
    "PHOTO", "CHEMO", "CONSUME", "MOVE", "FORAGE",
    "DEFENSE", "DETOX", "TOXPROD", "VRESIST", "SOCIAL", "MEDIATE"
]

MODULE_WEIGHT_SIZES = np.array([
    4,  # PHOTO:   [efficiency, toxic_tolerance, light_sensitivity, storage_rate]
    4,  # CHEMO:   [efficiency, specificity, saturation_threshold, gradient_follow]
    4,  # CONSUME: [prey_selectivity, handling_efficiency, decomp_preference, aggression]
    8,  # MOVE:    [light_seek, density_avoid, nutrient_stay, chemo_toxic_seek,
        #           random_wt, stay_tend, light_str, toxic_response]
    4,  # FORAGE:  [extraction_eff, resource_discrim, storage_cap, cooperative_signal]
    4,  # DEFENSE: [shell, camouflage, size_invest, counter_attack]
    4,  # DETOX:   [detox_eff, toxin_tolerance, conversion_rate, selective_uptake]
    0,  # TOXPROD: (no evolvable weights)
    4,  # VRESIST: [recognition_specificity, suppression_strength, resistance_breadth, immune_memory]
    4,  # SOCIAL:  [identity_signal, compatibility_assessment, approach_avoidance, relationship_strength]
    4,  # MEDIATE: [pollination_drive, route_memory, network_coordination, reward_sensitivity]
], dtype=np.int32)

MODULE_WEIGHT_OFFSETS = np.zeros(N_MODULES, dtype=np.int32)
_off = 0
for _m in range(N_MODULES):
    MODULE_WEIGHT_OFFSETS[_m] = _off
    _off += MODULE_WEIGHT_SIZES[_m]
TOTAL_MODULE_WEIGHTS = _off  # 32

N_STANDALONE_PARAMS = 4
STANDALONE_OFFSET = TOTAL_MODULE_WEIGHTS
TOTAL_WEIGHT_PARAMS = TOTAL_MODULE_WEIGHTS + N_STANDALONE_PARAMS  # 36

SP_TRANSFER_RECEPTIVITY = 0
SP_TRANSFER_SELECTIVITY = 1
SP_VIRAL_RESISTANCE = 2
SP_LYSO_SUPPRESSION = 3

# Module costs: [maintenance, expression]
#                PH    CH    CO    MV    FO    DE    DT    TP    VR    SO    ME
MODULE_MAINTENANCE = np.array([0.20, 0.25, 0.12, 0.08, 0.12, 0.18, 0.20, 0.03, 0.15, 0.10, 0.07])
MODULE_EXPRESSION  = np.array([0.10, 0.12, 0.08, 0.06, 0.06, 0.10, 0.10, 0.02, 0.08, 0.05, 0.04])
BASE_MAINTENANCE = 0.05
# Cost examples:
#   PHOTO+MOVE+TOXPROD                    = 0.54
#   + FORAGE (efficient producer)         = 0.72
#   + DEFENSE (armored producer)          = 1.00
#   + CONSUME (all-in generalist)         = 1.20
#   + DETOX (toxic zone specialist)       = 1.50
#   + VRESIST (immune system)             = 1.73
#   + SOCIAL (relational)                 = 1.88
#   + MEDIATE (pollinator)                = 2.08

GAINABLE_MODULES = [M_CHEMO, M_CONSUME, M_MOVE, M_FORAGE, M_DEFENSE, M_DETOX, M_VRESIST, M_SOCIAL, M_MEDIATE]

# Shedding decay rates per module per check (behavioral > immune > metabolic)
# TOXPROD = 0 (never sheds — structural)
SHEDDING_DECAY = np.array([
    0.008,   # PHOTO    — metabolic
    0.008,   # CHEMO    — metabolic
    0.008,   # CONSUME  — metabolic
    0.025,   # MOVE     — behavioral
    0.008,   # FORAGE   — metabolic
    0.012,   # DEFENSE  — immune
    0.008,   # DETOX    — metabolic
    0.0,     # TOXPROD  — structural (never sheds)
    0.012,   # VRESIST  — immune
    0.025,   # SOCIAL   — behavioral
    0.025,   # MEDIATE  — behavioral
])


# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────

