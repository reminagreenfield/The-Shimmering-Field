"""Simulation configuration."""


class Config:
    grid_size = 80
    light_max = 1.0
    light_min = 0.05
    zone_count = 8

    # Toxicity
    toxic_decay_rate = 0.01
    toxic_diffusion_rate = 0.06
    toxic_production_rate = 0.015
    toxic_threshold_low = 0.3
    toxic_threshold_medium = 0.8
    toxic_threshold_high = 1.5
    toxic_damage_medium = 1.5
    toxic_damage_high = 5.0
    toxic_photo_penalty = 1.0

    # Nutrients
    nutrient_base_rate = 0.001       # was 0.002 — slower baseline regeneration
    nutrient_from_decomp = 0.25      # was 0.4 — less nutrient from death
    nutrient_max = 2.0               # was 3.0 — scarcity is the driver of complexity

    # Population
    initial_population = 400
    initial_consumer_fraction = 0.25     # was 0.25
    initial_detritivore_fraction = 0.15
    initial_omnivore_fraction = 0.08     # generalist animals that eat both plants and animals
    energy_initial = 40.0
    energy_max = 150.0
    energy_reproduction_threshold = 42.0
    energy_reproduction_cost = 40.0
    energy_movement_cost = 0.2
    max_age = 200
    min_reproduction_age = 8
    offspring_distance = 5
    # Density-dependent reproduction: threshold scales up with local crowding
    repro_density_penalty = 8.0   # extra energy needed per neighbor
    carrying_capacity = 1000       # soft population cap
    carrying_capacity_penalty = 1.0 # repro threshold multiplier at cap

    # Energy production
    photosynthesis_base = 3.5
    chemosynthesis_base = 2.2
    scavenge_base = 4.5              # was 2.5 — detritivores more viable

    # Sessile-mobile divergence
    initial_sessile_fraction = 0.90  # was 0.40 — most producers start rooted
    sessile_photo_bonus = 1.70       # was 1.50 — rooted producers photosynthesize 70% better
    mobile_producer_penalty = 0.55   # producers WITH MOVE photosynthesize at 55% efficiency
    move_cost_producer_mult = 2.5    # MOVE module costs 2.5x for producers (heavy, not built for it)
    consumer_mobility_hunt_bonus = 0.12   # mobile consumers get +12% hunt success
    consumer_mobility_scavenge_bonus = 0.20  # mobile consumers scavenge 20% better
    move_gain_producer_mult = 0.15   # producers 85% less likely to evolve MOVE
    move_lose_consumer_mult = 0.15   # consumers 85% less likely to lose MOVE
    sessile_prey_vulnerability = 0.30 # big bonus — sessile prey cannot flee

    # Prey scent: diffused signals for predator navigation
    prey_scent_sigma = 8.0       # diffusion radius (cells) — how far scent travels
    prey_scent_decay = 0.90      # per-step retention — scent lingers
    prey_scent_deposit = 1.0     # amount each organism deposits

    # Trophic rules (hard constraints)
    # Herbivore: CONSUME + low prey_selectivity → can ONLY eat producers
    # Carnivore: CONSUME + high prey_selectivity → can ONLY eat animals (herb/carn/omni)
    # Omnivore:  PHOTO + CONSUME → can eat everything
    # Detritivore: CONSUME + high decomp_pref → eats decomposition, doesn't hunt
    detritivore_decomp_threshold = 0.45  # decomp_pref above this = detritivore (doesn't hunt)
    herbivore_selectivity_threshold = 0.45  # below = herbivore (plants only)
    carnivore_selectivity_threshold = 0.55  # above = carnivore (animals only)
    # Between 0.35 and 0.65 = omnivore (eats both, less efficient than specialists)
    detritivore_predation_resist = 0.85  # 70% chance predators skip detritivores (unpalatable)
    detritivore_predation_penalty = 0.80  # predation success reduced 60% against detritivores (taste bad)
    detritivore_repro_bonus = 0.60        # detritivores reproduce at 70% of normal threshold (r-selected)

    # Metabolic interference (asymmetric)
    producer_consume_penalty = 0.65      # hunting reduction for rare PHOTO+CONSUME hybrids
    consume_producer_penalty = 0.85      # photosynthesis reduction for rare PHOTO+CONSUME hybrids
    consumer_specialist_bonus = 1.2    # was 0.5 — obligate consumers are much better hunters

    # Predation
    predation_check_interval = 1
    predation_base_success = 0.20       # higher base success — slightly better hunters
    predation_energy_fraction = 0.55
    predation_hunt_radius_base = 1      # sessile predators still reach nearby
    predation_hunt_radius_mobile = 4    # mobile animals range widely
    predation_hunt_radius_carnivore = 6  # apex predators range even further
    predation_max_kills_per_cell = 1
    predator_satiation = 0.90  # probability a fed predator skips next hunt
    # Herbivore/carnivore gradient (via prey_selectivity weight)
    herbivore_producer_bonus = 0.10
    carnivore_consumer_bonus = 0.20     # was 0.15 — carnivores better at catching prey
    herbivore_energy_mult = 1.5        # herbivores are efficient grazers
    carnivore_energy_mult = 1.20        # carnivores extract well from kills
    graze_energy_fraction = 0.35        # fraction of available energy taken per graze
    graze_min_prey_energy = 15.0        # plants below this energy can't be grazed (root reserve)        # herbivores take 35% of producer energy per graze
    omnivore_hunt_penalty = 0.50        # omnivores hunt at 75% of specialist effectiveness
    omnivore_graze_mult = 0.50          # omnivore grazing less efficient than herbivore
    omnivore_kill_mult = 0.50           # omnivore kills less efficient than carnivore

    # Decomposition
    decomp_death_deposit = 5.0       # was 2.0 — more material for detritivores
    decomp_decay_rate = 0.999
    decomp_diffusion_rate = 0.008  # light local diffusion for feeding zone
    decomp_diffusion_interval = 3
    decomp_scent_sigma = 5  # gaussian blur for navigation scent
    decomp_scent_interval = 3  # recompute scent every N steps

    # FORAGE module
    forage_extraction_bonus = 0.25     # max +25% resource extraction
    forage_storage_bonus = 30.0        # max extra energy capacity
    forage_cooperative_radius = 1      # cells to check for cooperators
    forage_cooperative_bonus = 0.08    # energy bonus per cooperating neighbor

    # DEFENSE module
    defense_shell_max = 0.55           # max predation probability reduction
    defense_camouflage_max = 0.35      # max chance predator skips target
    defense_counter_damage = 5.0       # energy damage to predator on failed hunt
    defense_size_cost_mult = 1.3       # size investment increases module cost

    # DETOX module
    detox_rate_max = 0.08              # max fraction of local toxin removed per step
    detox_tolerance_bonus = 0.6        # max addition to toxic damage threshold
    detox_energy_conversion = 0.4      # fraction of metabolized toxin → energy
    detox_environment_effect = 0.5     # fraction of detox that cleans the environment

    # VRESIST module
    vresist_base_resistance = 0.6      # base infection resistance with VRESIST (vs ~0.3 standalone)
    vresist_specificity_bonus = 0.25   # extra resistance from specificity (familiar strains)
    vresist_breadth_bonus = 0.15       # extra resistance from breadth (all strains)
    vresist_suppression_max = 0.85     # max lysogenic suppression probability
    vresist_memory_boost = 0.3         # resistance boost after surviving infection
    vresist_memory_decay = 0.995       # how fast immune memory decays per step

    # SOCIAL module
    social_signal_radius = 1           # cells for social signal deposit (0 = own cell only)
    social_compatibility_bonus = 0.15  # energy bonus per compatible neighbor
    social_incompatibility_penalty = 0.05  # energy penalty per incompatible neighbor
    social_relationship_growth = 0.02  # relationship score growth per step near compatible
    social_relationship_decay = 0.998  # relationship score decay per step
    social_update_interval = 2         # how often social field updates

    # MEDIATE module (pollination/dispersal)
    mediate_repro_bonus = 0.25         # reproduction threshold reduction per mediator nearby
    mediate_radius = 3                 # wider radius — mediators serve larger area
    mediate_energy_reward = 2.0        # energy reward to mediator per facilitated reproduction
    mediate_network_decay = 0.97       # mediator field decay per step
    mediate_update_interval = 2        # how often mediator field updates
    mediate_passive_reward = 0.15      # energy per step when near immature organisms (dev support)

    # Nutrient cycling (emergent from module interactions)
    nutrient_detox_deposit = 0.3       # fraction of detox byproduct that becomes nutrients
    nutrient_consume_deposit = 0.15    # nutrient release from consume processing
    nutrient_death_per_module = 0.3    # nutrients deposited per module on death
    nutrient_forage_coop_boost = 0.1   # local nutrient regen boost per FORAGE cooperator

    # Reproductive manipulation (Wolbachia-style, via lysogenic genome)
    repro_manip_threshold = 0.3        # min lysogenic_strength for manipulation to activate
    repro_manip_trait_bias = 0.15      # max weight blend toward lysogenic genome in offspring
    repro_manip_receptivity_boost = 0.4  # boost to offspring's transfer_receptivity param
    repro_manip_viability_cost = 3.0   # energy penalty to divergent offspring
    repro_manip_divergence_thresh = 0.5  # weight distance above which viability penalty applies
    repro_manip_saturation = 0.7       # population lysogenic fraction where self-limiting kicks in

    # Behavioral hijacking (Toxoplasma/cordyceps-style, via lytic viral load)
    hijack_load_min = 0.05             # minimum viral_load for hijack effects
    hijack_load_heavy = 0.5            # viral_load above which heavy hijack kicks in
    hijack_defense_suppress = 0.6      # max defense suppression at full hijack
    hijack_vresist_suppress = 0.4      # max VRESIST suppression at full hijack
    hijack_energy_suppress = 0.7       # fraction of energy acquisition blocked at heavy hijack
    hijack_density_seek = 3.0          # movement weight toward high-density areas (spread virus)
    hijack_stress_amplifier = 1.5      # toxic stress multiplier on hijack intensity

    # Endosymbiosis
    endo_check_interval = 10           # check for mergers every N steps
    endo_relationship_threshold = 0.6  # lower — co-location builds slowly
    endo_energy_threshold = 50.0       # min energy for both partners
    endo_toxic_min = 0.05              # min local toxic for stress window
    endo_toxic_max = 1.0               # max local toxic (too harsh = can't merge)
    endo_complementarity_min = 1       # min module difference count (sessile/mobile = complementary)
    endo_probability = 0.15            # base probability of merger when conditions met
    endo_vresist_penalty = 0.5         # merger probability reduction for VRESIST holders
    endo_weight_blend = 0.5            # weight blending ratio (0.5 = equal blend)
    endo_energy_bonus = 1.2            # energy multiplier for merged organism
    endo_max_modules = 8               # max modules a merged organism can hold

    # Capacity shedding (use-it-or-lose-it)
    shedding_check_interval = 5        # check shedding every N steps
    shedding_usage_gain = 0.015        # usage gained per check when module contributes
    shedding_deactivate_threshold = 0.15  # usage below this → module goes dormant
    shedding_loss_threshold = 0.03     # usage below this → module lost from genome
    shedding_stress_reactivation = 0.3 # toxic level that reactivates dormant modules
    # Decay rates per check (per shedding_check_interval steps):
    shedding_decay_behavioral = 0.025  # MOVE, SOCIAL, MEDIATE — ~200 steps to deactivate
    shedding_decay_immune = 0.012      # VRESIST, DEFENSE — ~400 steps
    shedding_decay_metabolic = 0.008   # PHOTO, CHEMO, CONSUME, FORAGE, DETOX — ~600 steps

    # Genomic incompatibility
    genomic_check_interval = 10        # check cascade every N steps
    genomic_stress_per_transfer = 0.15  # stress added per HGT event
    genomic_stress_decay = 0.005       # faster natural integration per step
    genomic_cascade_threshold = 3.0    # higher threshold — cascades are rare events
    genomic_toxic_multiplier = 1.5     # toxic amplifies stress for cascade check
    genomic_phase1_energy_penalty = 0.4  # energy acquisition reduction
    genomic_phase2_deactivate_prob = 0.3  # prob of random module deactivation
    genomic_phase3_energy_drain = 2.0  # direct energy drain per step
    genomic_pruning_threshold = 4.0    # stress level triggering pruning
    genomic_pruning_module_loss = 2    # max modules lost during pruning
    genomic_pruning_stress_relief = 0.8  # fraction of stress removed
    genomic_pruning_receptivity_boost = 0.5  # transfer receptivity boost post-prune

    # Developmental dependency
    dev_window_length = 75             # developmental window (steps)
    dev_progress_rate = 0.04           # progress per step near compatible organism
    dev_compromise_energy = 0.5        # max energy penalty for fully compromised
    dev_compromise_repro = 1.5         # reproduction threshold multiplier when compromised
    dev_compromise_aging = 1.5         # aging speed multiplier when compromised

    # Nonlinear collapse dynamics
    collapse_threshold = 0.20          # ecosystem integrity below this → collapsed
    recovery_threshold = 0.42          # must exceed this to recover (hysteresis)
    collapse_repro_penalty = 0.8       # max reproduction success reduction in collapsed zone
    collapse_energy_penalty = 0.15      # energy acquisition penalty in collapsed zone
    collapse_shedding_mult = 2.0       # shedding accelerates in collapsed zones
    collapse_sigmoid_steepness = 15.0  # steepness of sigmoid collapse function
    compounding_base = 1.2             # each active failure mode multiplies stress by this

    # Fungal networks
    fungal_growth_rate = 0.02          # growth from decomposition per step
    fungal_decomp_threshold = 0.5     # min decomp to feed fungal growth
    fungal_diffusion_rate = 0.05       # spread rate to neighbors
    fungal_decay_rate = 0.005          # maintenance decay per step
    fungal_nutrient_transport = 0.03   # fraction of nutrients redistributed via fungi
    fungal_toxic_conduit = 0.02        # toxic diffusion bonus along fungal paths
    fungal_hgt_transport = 0.2         # fraction of genome fragments carried by fungi
    fungal_surge_mult = 3.0            # growth rate multiplier during decomp surges
    fungal_surge_threshold = 5.0       # decomp level that triggers surge growth
    fungal_update_interval = 3         # update fungal grid every N steps

    # Evolution
    mutation_rate = 0.08
    module_gain_rate = 0.005       # slightly higher — more modules available now
    module_lose_rate = 0.003

    # Horizontal transfer
    transfer_check_interval = 5
    transfer_blend_rate_recent = 0.10
    transfer_blend_rate_intermediate = 0.18
    transfer_blend_rate_ancient = 0.30
    decomp_fragment_decay = 0.005
    decomp_fragment_diffusion = 0.008
    module_transfer_rate = 0.015
    sedimentation_rate_recent = 0.005
    sedimentation_rate_intermediate = 0.002
    ancient_decay_rate = 0.001
    stratum_access_medium = 0.3
    stratum_access_high = 0.8

    # Viral system
    viral_decay_rate = 0.01
    viral_diffusion_rate = 0.08
    viral_infection_rate = 0.3
    viral_lytic_damage = 2.0
    viral_lytic_growth = 0.1
    viral_burst_threshold = 1.0
    viral_burst_amount = 8.0
    viral_burst_radius = 3
    lysogenic_probability = 0.4
    lysogenic_activation_toxic = 0.6
    lysogenic_blend_rate = 0.1
    lysogenic_inheritance = 0.8
    viral_check_interval = 3

    # Simulation
    total_timesteps = 10000
    snapshot_interval = 100
    output_dir = "output_p4s6"
    random_seed = None  # None = different each run


# ─────────────────────────────────────────────────────
# World
# ─────────────────────────────────────────────────────

