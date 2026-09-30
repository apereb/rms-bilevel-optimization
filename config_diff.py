"""
Qualitative comparison of two configurations by global config_idx: per-
stage composition (machine type, config, module set, unit count) and the
reconfiguration effort between them in both directions.

Nothing here recomputes cost or Theta/Phi -- this is purely descriptive,
meant for eyeballing *why* two configs differ (e.g. the Phase-1 winner at
one alpha vs. another, or two candidate winners across an epsilon/tau
sweep), not for re-deriving any of the pipeline's own selection logic.

Usage
-----
    # configs come from a previous run's persisted (stage, machine_id,
    # config_id, n_machines) detail -- resolved against the instance file
    # exactly like orchestrate.load_scheduling_results does:
    python config_diff.py instance.json --run baseline.meta.json --ids 1234,5678

    # or, with no --run, global ids are interpreted as positions in
    # config_space.feasible_configs(instance) (the canonical universe
    # ordering every module in this pipeline uses):
    python config_diff.py instance.json --ids 1234,5678
"""

import argparse
from typing import List

import config_space as cs
import orchestrate as orch
import structural as st
from config_space import SystemConfig


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------

def describe_config(y: SystemConfig) -> List[dict]:
    """Per-stage composition of a SystemConfig, one dict per stage."""
    return [
        {
            "stage": s,
            "machine_id": mc.machine_id,
            "config_id": mc.config_id,
            "n_machines": n,
            "setup_cost": mc.setup_cost,
            "modules": sorted(mc.modules),
            "capable_ops": sorted(mc.capable_ops),
        }
        for s, (mc, n) in enumerate(y)
    ]


def print_config_description(label: str, cid: int, y: SystemConfig):
    print(f"{label}: config_idx={cid}  {cs.config_label(y)}")
    print("| stage | machine_id | config_id | n | setup_cost | modules | capable_ops |")
    print("|---|---|---|---|---|---|---|")
    for row in describe_config(y):
        print(f"| {row['stage']} | {row['machine_id']} | {row['config_id']} | "
              f"{row['n_machines']} | {row['setup_cost']:.2f} | "
              f"{row['modules']} | {row['capable_ops']} |")


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def diff_configs(y1: SystemConfig, y2: SystemConfig, instance: cs.RMSInstance) -> dict:
    """
    Per-stage diff between y1 and y2, plus the directional reconfiguration
    effort both ways (structural.reconfiguration_time -- +inf if any
    stage's machine type differs, per structural.same_machine_types).
    """
    if len(y1) != len(y2):
        raise ValueError("diff_configs: y1 and y2 have a different number of stages.")

    stage_diffs = []
    for s, ((mc1, n1), (mc2, n2)) in enumerate(zip(y1, y2)):
        same_type = mc1.machine_id == mc2.machine_id
        same_config = same_type and mc1.config_id == mc2.config_id
        modules_1, modules_2 = set(mc1.modules), set(mc2.modules)
        stage_diffs.append({
            "stage": s,
            "same_machine_type": same_type,
            "same_config": same_config,
            "machine_id": (mc1.machine_id, mc2.machine_id),
            "config_id": (mc1.config_id, mc2.config_id),
            "n_machines": (n1, n2),
            "delta_n": n2 - n1,
            "modules_added": sorted(modules_2 - modules_1),
            "modules_removed": sorted(modules_1 - modules_2),
            "modules_shared": sorted(modules_1 & modules_2),
        })

    return {
        "same_machine_types_everywhere": st.same_machine_types(y1, y2),
        "stage_diffs": stage_diffs,
        "T_rec_1_to_2": st.reconfiguration_time(y1, y2, instance),
        "T_rec_2_to_1": st.reconfiguration_time(y2, y1, instance),
        "C_config_1": cs.config_setup_cost(y1),
        "C_config_2": cs.config_setup_cost(y2),
    }


def print_diff(diff: dict, id1: int, id2: int):
    print(f"\nDiff #{id1} -> #{id2}")
    print(f"Reachable by reconfiguration: {diff['same_machine_types_everywhere']}")
    print(f"  T_rec(#{id1} -> #{id2}) = {diff['T_rec_1_to_2']}")
    print(f"  T_rec(#{id2} -> #{id1}) = {diff['T_rec_2_to_1']}")
    print(f"  C_config: {diff['C_config_1']:.2f} -> {diff['C_config_2']:.2f} "
          f"(delta {diff['C_config_2'] - diff['C_config_1']:+.2f})")
    print()
    print("| stage | same_type | same_config | machine_id | config_id | n | "
          "modules_added | modules_removed | modules_shared |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in diff["stage_diffs"]:
        print(f"| {d['stage']} | {d['same_machine_type']} | {d['same_config']} | "
              f"{d['machine_id'][0]}->{d['machine_id'][1]} | "
              f"{d['config_id'][0]}->{d['config_id'][1]} | "
              f"{d['n_machines'][0]}->{d['n_machines'][1]} ({d['delta_n']:+d}) | "
              f"{d['modules_added']} | {d['modules_removed']} | {d['modules_shared']} |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_configs(instance: cs.RMSInstance, ids: List[int], run_json: str = None):
    """
    Map global config_idx -> SystemConfig, either from a persisted run
    (config_idx as recorded when that run scheduled it) or, absent a
    run, from position in config_space.feasible_configs(instance) -- the
    canonical universe ordering every module in this pipeline assumes
    when no run-specific mapping is given.
    """
    if run_json:
        results = orch.load_scheduling_results(run_json, instance)
        by_idx = {r["config_idx"]: r["y"] for r in results}
        missing = [i for i in ids if i not in by_idx]
        if missing:
            raise KeyError(
                f"config_idx {missing} not found in {run_json} -- that run "
                f"never scheduled them (try without --run to resolve "
                f"against the full feasible universe instead)."
            )
        return [by_idx[i] for i in ids]
    else:
        universe = cs.feasible_configs(instance)
        missing = [i for i in ids if not (0 <= i < len(universe))]
        if missing:
            raise IndexError(
                f"config_idx {missing} out of range for the "
                f"{len(universe)}-config feasible universe."
            )
        return [universe[i] for i in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instance_file")
    parser.add_argument("--ids", required=True,
                        help="comma-separated global config_idx values (exactly 2)")
    parser.add_argument("--run", dest="run_json", default=None,
                        help="resolve config_idx against this run's persisted mapping "
                             "instead of the full feasible universe")

    args = parser.parse_args()
    ids = [int(i) for i in args.ids.split(",")]
    if len(ids) != 2:
        parser.error("--ids must contain exactly 2 comma-separated config_idx values")

    instance = cs.load_instance_json(args.instance_file)
    y1, y2 = resolve_configs(instance, ids, args.run_json)

    print_config_description("Config A", ids[0], y1)
    print()
    print_config_description("Config B", ids[1], y2)

    diff = diff_configs(y1, y2, instance)
    print_diff(diff, ids[0], ids[1])


if __name__ == "__main__":
    main()