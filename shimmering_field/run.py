"""Entry point for running the simulation."""

import time
import os
import json

from .constants import *
from .config import Config
from .world import World


def run_simulation(cfg=None):
    cfg = cfg or Config()
    world = World(cfg)

    print(f"The Shimmering Field — Phase 4 Step 6: Sessile-Mobile Divergence")
    print(f"Grid: {cfg.grid_size}×{cfg.grid_size}  |  Pop: {cfg.initial_population}  |  Weights: {TOTAL_WEIGHT_PARAMS}")
    print(f"Nutrient scarcity: max={cfg.nutrient_max}  base_rate={cfg.nutrient_base_rate}  from_decomp={cfg.nutrient_from_decomp}")
    print(f"Sessile-mobile: sessile_bonus={cfg.sessile_photo_bonus}x  mobile_penalty={cfg.mobile_producer_penalty}x  move_cost_prod={cfg.move_cost_producer_mult}x")
    print(f"Module costs — PH:{MODULE_MAINTENANCE[M_PHOTO]+MODULE_EXPRESSION[M_PHOTO]:.2f}  "
          f"CH:{MODULE_MAINTENANCE[M_CHEMO]+MODULE_EXPRESSION[M_CHEMO]:.2f}  "
          f"CO:{MODULE_MAINTENANCE[M_CONSUME]+MODULE_EXPRESSION[M_CONSUME]:.2f}  "
          f"MV:{MODULE_MAINTENANCE[M_MOVE]+MODULE_EXPRESSION[M_MOVE]:.2f}  "
          f"FO:{MODULE_MAINTENANCE[M_FORAGE]+MODULE_EXPRESSION[M_FORAGE]:.2f}  "
          f"DE:{MODULE_MAINTENANCE[M_DEFENSE]+MODULE_EXPRESSION[M_DEFENSE]:.2f}  "
          f"DT:{MODULE_MAINTENANCE[M_DETOX]+MODULE_EXPRESSION[M_DETOX]:.2f}  "
          f"VR:{MODULE_MAINTENANCE[M_VRESIST]+MODULE_EXPRESSION[M_VRESIST]:.2f}  "
          f"SO:{MODULE_MAINTENANCE[M_SOCIAL]+MODULE_EXPRESSION[M_SOCIAL]:.2f}  "
          f"ME:{MODULE_MAINTENANCE[M_MEDIATE]+MODULE_EXPRESSION[M_MEDIATE]:.2f}")
    print(f"Repro manipulation: threshold={cfg.repro_manip_threshold}  "
          f"trait_bias={cfg.repro_manip_trait_bias}  saturation={cfg.repro_manip_saturation}")
    print(f"{'─' * 160}")

    start = time.time()
    for t in range(cfg.total_timesteps):
        world.update()

        if world.timestep % cfg.snapshot_interval == 0:
            world.save_snapshot(cfg.output_dir)
            s = world.stats_history[-1]
            el = time.time() - start
            r = s["roles"]
            mc = s["module_counts"]
            print(
                f"  t={s['t']:5d}  |  pop={s['pop']:5d}  |  e={s['avg_energy']:5.1f}  |  "
                f"gen={s['max_gen']:4d}  |  tox={s['toxic_mean']:.3f} ntr={s['nutrient_mean']:.3f}  |  "
                f"prod={r['producer']:4d} herb={r['herbivore']:3d} carn={r['carnivore']:3d} detr={r['detritivore']:3d} omni={r['omnivore']:3d}  |  "
                f"sessP={s['sessile_producers']:4d} mobP={s['mobile_producers']:3d} mobC={s['mobile_consumers']:3d}  |  "
                f"FO={mc['FORAGE']:4d} DE={mc['DEFENSE']:4d} VR={mc['VRESIST']:4d} SO={mc['SOCIAL']:4d} ME={mc['MEDIATE']:3d}  |  "
                f"lyso={s['lyso_fraction']:.2f} hjk={s['hijack_fraction']:.2f}  |  "
                f"mrg={s['total_mergers']:3d} comp={s['composite_organisms']:3d}  |  "
                f"gstr={s['avg_genomic_stress']:.2f} casc={s['cascade_organisms']:3d}  |  "
                f"clps={s['collapsed_zones']:5d} fng={s['fungal_mean']:.3f}  |  "
                f"kill={s['kills']:3d}  |  mod={s['avg_modules']:.2f}  |  {el:.1f}s"
            )

        if world.pop == 0:
            print(f"\n  *** EXTINCTION at t={world.timestep} ***")
            break

    el = time.time() - start
    print(f"{'─' * 160}")
    print(f"Done in {el:.1f}s  |  Pop: {world.pop}  |  Total predation kills: {world.total_predation_kills}")

    if world.pop > 0:
        r = world.stats_history[-1]["roles"]
        mc = world.stats_history[-1]["module_counts"]
        s = world.stats_history[-1]
        print(f"Roles — Producers: {r['producer']}  Herbivores: {r['herbivore']}  "
              f"Carnivores: {r['carnivore']}  Detritivores: {r['detritivore']}  "
              f"Omnivores: {r['omnivore']}")
        print(f"Modules — PH:{mc['PHOTO']} CH:{mc['CHEMO']} CO:{mc['CONSUME']} "
              f"MV:{mc['MOVE']} FO:{mc['FORAGE']} DE:{mc['DEFENSE']} DT:{mc['DETOX']} "
              f"VR:{mc['VRESIST']} SO:{mc['SOCIAL']} ME:{mc['MEDIATE']} TP:{mc['TOXPROD']}")
        print(f"Social — avg_relationship: {s['avg_relationship']:.3f}  "
              f"max: {s['max_relationship']:.3f}  "
              f"avg_immune: {s['avg_immune_exp']:.3f}  "
              f"mediator: {s['mediator_field_mean']:.4f}  "
              f"nutrients: {s['nutrient_mean']:.3f}")
        print(f"Manipulation — lyso_fraction: {s['lyso_fraction']:.3f}  "
              f"total_manipulated_births: {s['manipulated_births']}  "
              f"hijack_fraction: {s['hijack_fraction']:.3f}  "
              f"total_hijacked_steps: {s['hijacked_steps']}")
        print(f"Endosymbiosis — total_mergers: {s['total_mergers']}  "
              f"composite_organisms: {s['composite_organisms']}  "
              f"max_merger_count: {s['max_merger_count']}")
        print(f"Shedding — dormant_modules: {s['dormant_modules']}  "
              f"avg_usage: {s['avg_usage']:.3f}")
        print(f"Genomic — avg_stress: {s['avg_genomic_stress']:.3f}  "
              f"cascade_organisms: {s['cascade_organisms']}  "
              f"max_phase: {s['max_cascade_phase']}")
        print(f"Development — mature: {s['mature_fraction']:.3f}  "
              f"compromised: {s['compromised_count']}")
        print(f"Collapse — collapsed_zones: {s['collapsed_zones']}  "
              f"avg_integrity: {s['avg_integrity']:.3f}")
        print(f"Fungal — mean_density: {s['fungal_mean']:.4f}  "
              f"max_density: {s['fungal_max']:.3f}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_summary.json"), 'w') as f:
        json.dump({"config": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                   "stats_history": world.stats_history}, f, indent=2)

    # Export pivotal moment index
    world._export_run_index()

    return world


if __name__ == "__main__":
    run_simulation()


if __name__ == "__main__":
    run_simulation()
