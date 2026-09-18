"""
Time-limit calibration probe for rms_bilevel_gurobi.py.

Purpose
-------
Answer empirically: "how much Gurobi time_limit per config do I actually
need?" by re-solving the *same* sample of feasible configs at several time
budgets (e.g. 5s, 15s, 30s, 60s) and recording how Cmax, the MIP gap, and
the resulting config ranking move between budgets.

This is a diagnostic/calibration tool, kept separate from rms_bilevel_gurobi.py
on purpose: it is not part of the main enumerate-everything pipeline, it is
meant to be run once (or occasionally) on a small representative sample to
decide what time_limit_per_config to use for the real overnight run.

Sampling strategy
------------------
Rather than the first N feasible configs (which, as observed on
instance_period1.json, mostly share the same stage-0..3 choice and only vary
on the last stage -- not very representative), this script enumerates ALL
feasible configs first (cheap: enumerate_configs + validate_config does not
call the solver), then picks `--n-samples` of them at an even stride across
the full feasible list. This spreads the sample across genuinely different
system configurations rather than near-duplicates.

Output
------
A single JSON file: <output>.json, with one entry per (config, time_limit)
pair solved. Structure:
{
  "meta": {...},
  "probes": [
    {"config_idx": ..., "y": [...], "time_limit": 5,  "Q": ..., "gap": ..., "status": ..., "runtime": ...},
    {"config_idx": ..., "y": [...], "time_limit": 15, "Q": ..., ...},
    ...
  ]
}
Written incrementally (same atomic-rewrite strategy as JsonRunWriter in
rms_bilevel_gurobi.py) so a long calibration run is also safe to interrupt.

Usage
-----
    python calibrate_time_limit.py instance.json [--n-samples 50] \\
        [--time-limits 5,15,30,60] [--mode unit|representative] \\
        [--output results/calibration]
"""

import sys
import os
import json
import time
import argparse

from legacy.rms_bilevel_gurobi import (
    load_instance_json, enumerate_configs, validate_config,
    build_pt_matrix, solve_flowshop_gurobi, config_setup_cost,
    _config_to_dict, _config_label,
)
import math


def collect_feasible_configs(instance):
    """
    Enumerate every feasible SystemConfig (no solver call -- this mirrors
    the pre-filtering in solve_bilevel, just without scheduling anything).
    Returns a list of (config_idx, y) pairs, config_idx matching the index
    solve_bilevel would report for the same config (i.e. position in the
    raw enumerate_configs() stream, not in the feasible-only list) so that
    calibration results can be cross-referenced with a real run's JSON.
    """
    feasible = []
    for idx, y in enumerate(enumerate_configs(instance)):
        if validate_config(y, instance):
            feasible.append((idx, y))
    return feasible


def even_stride_sample(items, n_samples):
    """
    Pick n_samples items at an even stride across the full list, always
    including the first and last item. Falls back to the full list if it's
    already smaller than n_samples.
    """
    n = len(items)
    if n <= n_samples:
        return list(items)
    step = (n - 1) / (n_samples - 1)
    indices = sorted(set(round(i * step) for i in range(n_samples)))
    return [items[i] for i in indices]


def build_scheduled_jobs(instance, mode):
    """Same job-list construction as solve_bilevel, extracted standalone."""
    active_jobs = [j for j in instance.jobs if j.demand > 0.0]
    if not active_jobs:
        active_jobs = instance.jobs

    if mode == "unit":
        return active_jobs
    elif mode == "representative":
        if instance.K is None:
            raise ValueError("mode='representative' requires instance.K to be set.")
        total_demand = sum(j.demand for j in active_jobs)
        scheduled = []
        for j in active_jobs:
            n_copies = math.ceil(instance.K * j.demand / total_demand)
            scheduled.extend([j] * n_copies)
        return scheduled
    else:
        raise ValueError(f"Unknown mode '{mode}'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_file")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--time-limits", type=str, default="5,15,30,60",
                        help="Comma-separated list of Gurobi time limits (s) to test, e.g. '5,15,30,60'")
    parser.add_argument("--mode", choices=["unit", "representative"], default="representative")
    parser.add_argument("--output", type=str, default="results/calibration",
                        help="Output base path; writes <output>.json")
    args = parser.parse_args()

    time_limits = [int(t) for t in args.time_limits.split(",")]

    instance = load_instance_json(args.instance_file)
    scheduled_jobs = build_scheduled_jobs(instance, args.mode)
    batch_size = len(scheduled_jobs)

    print(f"[Calibration] Enumerating feasible configs for {args.instance_file}...")
    feasible = collect_feasible_configs(instance)
    print(f"[Calibration] {len(feasible)} feasible configs found.")

    sample = even_stride_sample(feasible, args.n_samples)
    print(f"[Calibration] Sampling {len(sample)} configs at even stride "
          f"across the feasible list.")
    print(f"[Calibration] Time limits to test: {time_limits}")
    print(f"[Calibration] Mode={args.mode}  batch_size={batch_size}")

    n_total_solves = len(sample) * len(time_limits)
    print(f"[Calibration] Total Gurobi calls planned: {n_total_solves} "
          f"(worst-case wall time ~= {sum(time_limits) * len(sample) / 60:.1f} min)")
    print("-" * 64)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    json_path = f"{args.output}.json"

    data = {
        "meta": {
            "instance_path": args.instance_file,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": args.mode,
            "n_feasible_total": len(feasible),
            "n_samples": len(sample),
            "time_limits": time_limits,
            "batch_size": batch_size,
        },
        "probes": [],
        "finished_at": None,
    }

    def flush():
        tmp = json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, json_path)

    flush()  # write meta immediately so the file exists from the start

    run_t0 = time.time()
    solve_count = 0

    for sample_pos, (config_idx, y) in enumerate(sample):
        pt_matrix = build_pt_matrix(y, instance, scheduled_jobs)
        n_machines = [
            max(1, min(n, sum(1 for j in range(batch_size) if pt_matrix[s][j] > 0)))
            for s, (_, n) in enumerate(y)
        ]
        cconfig = config_setup_cost(y)
        label = _config_label(y)

        for tl in time_limits:
            t0 = time.time()
            makespan, activity, gantt, solve_info = solve_flowshop_gurobi(
                pt_matrix, batch_size, instance.nb_stages,
                n_machines=n_machines, time_limit=tl,
            )
            wall = time.time() - t0
            solve_count += 1

            if args.mode == "unit":
                perf = makespan
                total_cost = cconfig + instance.alpha * makespan
            else:
                tp = batch_size / makespan if makespan > 0 else float("inf")
                perf = tp
                total_cost = cconfig + instance.alpha / tp

            data["probes"].append({
                "sample_pos":  sample_pos,     # 0..len(sample)-1, stable x-axis key
                "config_idx":  config_idx,     # matches solve_bilevel's numbering
                "y_label":     label,
                "time_limit":  tl,
                "Q":           makespan,
                "total_cost":  total_cost,
                "solver_status": solve_info["status"],
                "mip_gap_pct":   solve_info["gap"],
                "solve_runtime_s": solve_info["runtime"],
                "node_count":    solve_info["node_count"],
            })
            flush()

            print(f"  [{solve_count:3d}/{n_total_solves}] sample#{sample_pos:2d} "
                  f"(config_idx={config_idx})  tl={tl:3d}s  "
                  f"Cmax={makespan:.3f}  gap={solve_info['gap']:.2f}%  "
                  f"status={solve_info['status']}  ({wall:.1f}s wall)")

    data["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["meta"]["total_wall_time_s"] = time.time() - run_t0
    flush()

    print("-" * 64)
    print(f"[Calibration] Done. {solve_count} solves written to {json_path}")
    print(f"[Calibration] Total wall time: {(time.time() - run_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
