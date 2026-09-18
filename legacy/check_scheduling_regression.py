"""
Regression check: re-solve a sample of configs from an old run.json with
the new scheduling.py and compare Q(y) to the historically recorded value.

Usage:
    python check_scheduling_regression.py instance_period1.json run1.json \
        [config_idx1 config_idx2 ...]

If no config_idx list is given, defaults to the Phase-2 winner plus 4
other configs spread across the corridor.

Note: the original run's configs mostly hit TIME_LIMIT (not proven
optimal), so exact Q equality isn't expected -- what matters is that the
new Q values are close (same ballpark, ideally <= old Q since the MIP
formulation is unchanged) and that solver_status/gap look sane, not a
byte-for-byte match.
"""

import json
import sys

import config_space as cs
import scheduling as sch


def build_mc_lookup(instance):
    lookup = {}
    for s, configs in enumerate(instance.stage_configs):
        for mc in configs:
            lookup[(s, mc.machine_id, mc.config_id)] = mc
    return lookup


def reconstruct(c, mc_lookup):
    mc_list, n_list = [], []
    for stage_entry in c["y"]:
        key = (stage_entry["stage"], stage_entry["machine_id"], stage_entry["config_id"])
        mc_list.append(mc_lookup[key])
        n_list.append(stage_entry["n_machines"])
    return tuple(zip(mc_list, n_list))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    instance_path = sys.argv[1]
    run_path = sys.argv[2]
    explicit_ids = [int(a) for a in sys.argv[3:]] if len(sys.argv) > 3 else None

    instance = cs.load_instance_json(instance_path)
    with open(run_path) as f:
        run = json.load(f)

    by_idx = {c["config_idx"]: c for c in run["configs"]}

    if explicit_ids is None:
        winner_idx = run["summary"]["phase2_best_config_idx"]
        corridor_ids = [e["config_idx"] for e in run["summary"]["corridor_detail"]]
        others = corridor_ids[:4] if len(corridor_ids) >= 4 else corridor_ids
        sample_ids = sorted(set([winner_idx] + others))
    else:
        sample_ids = explicit_ids

    mc_lookup = build_mc_lookup(instance)
    sample_configs = [reconstruct(by_idx[i], mc_lookup) for i in sample_ids]

    time_limit = run["meta"]["time_limit_per_config"]
    print(f"Re-solving {len(sample_ids)} config(s) with time_limit={time_limit}s "
          f"(matching the original run's meta)...")
    print("-" * 72)

    results, run_info = sch.schedule_configs(
        instance, sample_configs, config_ids=sample_ids,
        time_limit_per_config=time_limit, verbose=False,
    )

    for r in results:
        old = by_idx[r["config_idx"]]
        delta = r["Q"] - old["Q"]
        print(f"config_idx={r['config_idx']}")
        print(f"  old: Q={old['Q']:.4f}  status={old['solver_status']:12s} "
              f"gap%={old['mip_gap_pct']}")
        print(f"  new: Q={r['Q']:.4f}  status={r['solve_info']['status']:12s} "
              f"gap%={r['solve_info']['gap']}")
        print(f"  delta Q = {delta:+.4f}   "
              f"C_config match: {abs(r['C_config'] - old['C_config']) < 1e-9}")
        print()


if __name__ == "__main__":
    main()