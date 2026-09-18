"""
Orchestration for the RMS bi-level optimization pipeline.

Composes config_space.py, structural.py, scheduling.py and evaluation.py
into four run modes -- the only thing this script does is decide which
configs get scheduled and in what order; all the actual logic lives in
the four modules it imports.

Modes
-----
full
    Baseline: schedule every feasible configuration once. This is the
    ~hours-long one-shot run meant to be compared against later,
    lighter-weight experiments.

two-pass
    Like full, but configs within a wide provisional corridor
    (refine_corridor_factor * epsilon around the pass-1 best) are
    re-scheduled with a larger time limit before the final evaluation --
    catches configs that pass-1 under-resolved before they get discarded
    or wrongly declared the winner.

partial-inversion
    Structural filtering *before* scheduling: compute Phi over the full
    feasible universe (cheap, no Gurobi), keep only the top
    `phi_top_fraction` of configs by Phi, and schedule only those. Trades
    completeness (the true C* might be outside the kept subset) for
    speed -- meant for experimentation, not as a replacement for the
    full baseline.

recompute
    No scheduling at all: reload a previous run's JSON (produced by any
    of the modes above, via JsonRunWriter), recompute Theta/Phi/corridor
    under the *current* instance parameters (alpha/epsilon/tau/lambda/
    mu/delta can be overridden on the CLI), and re-select the winner.
    This is the direct replacement for the old "explore_phase2.py"
    script: same purpose, now sharing structural.py/evaluation.py with
    every other mode instead of maintaining its own copy of the
    measures.

Usage
-----
    python orchestrate.py full instance.json --time-limit 15 --output runs/run1
    python orchestrate.py two-pass instance.json --time-limit 15 --refine-time-limit 60
    python orchestrate.py partial-inversion instance.json --phi-top-fraction 0.3
    python orchestrate.py recompute instance.json --run-json runs/run1.json --epsilon 0.2
"""

import argparse
import json
import random
import os
import time
from typing import Dict, List

import config_space as cs
import evaluation as ev
import scheduling as sch
import structural as st


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_meta(instance: cs.RMSInstance, mode: str, **extra) -> dict:
    meta = {
        "mode":            mode,
        "started_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "nb_stages":       instance.nb_stages,
        "nb_jobs":         instance.nb_jobs,
        "N_max":           instance.N_max,
        "K":               instance.K,
        "alpha":           instance.alpha,
        "epsilon":         instance.epsilon,
        "tau":             instance.tau,
        "lambda_coef":     instance.lambda_coef,
        "mu_coef":         instance.mu_coef,
        "delta_coef":      instance.delta_coef,
    }
    meta.update(extra)
    return meta


def finalize(
    instance: cs.RMSInstance,
    results: List[dict],
    theta: Dict[int, float],
    phi: Dict[int, float],
    log: sch.RunLogger,
    writer: sch.JsonRunWriter = None,
) -> dict:
    """
    Run evaluation.evaluate(), log a human-readable summary, and persist
    it to the JSON writer if given. Shared tail end of every mode.
    """
    out = ev.evaluate(instance, results, theta, phi)

    log.write("-" * 64)
    log.write(f"[Result] C* = {out['C_star']:.4f}")
    log.write(f"[Result] threshold = (1+eps)*C* = {out['threshold']:.4f}")
    log.write(f"[Result] corridor size = {len(out['corridor'])} / {len(results)} scheduled")

    if out["winner"] is not None:
        w = out["winner"]
        log.write(f"[Result] Phase-2 winner: config_idx={w['config_idx']}  "
                  f"{cs.config_label(w['y'])}")
        log.write(f"    total_cost={w['total_cost']:.4f}  Q={w['Q']:.4f}  "
                  f"Theta={w['Theta']:.4f}  Phi={w['Phi']:.4f}")
    else:
        log.write("[Result] Corridor is empty -- no winner.")

    if writer is not None:
        winner = out["winner"]
        writer.set_summary({
            "finished_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "C_star":       out["C_star"],
            "threshold":    out["threshold"],
            "corridor_size": len(out["corridor"]),
            "phase2_best_config_idx": winner["config_idx"] if winner else None,
            "phase2_best_y": cs.config_to_dict(winner["y"]) if winner else None,
        })

    return out


# ---------------------------------------------------------------------------
# Mode: full
# ---------------------------------------------------------------------------

def run_full(instance: cs.RMSInstance, output_base: str,
            time_limit_per_config: int, verbose: bool = True,
            limit: int = None, seed: int = 42) -> dict:
    all_configs = cs.feasible_configs(instance)
    all_ids = list(range(len(all_configs)))

    if limit is not None and limit < len(all_configs):
        rng = random.Random(seed)
        sampled_ids = sorted(rng.sample(all_ids, limit))
        configs = [all_configs[i] for i in sampled_ids]
        ids = sampled_ids
    else:
        configs, ids = all_configs, all_ids

    log = sch.RunLogger(f"{output_base}.log" if output_base else None)
    writer = sch.JsonRunWriter(
        output_base,
        meta=build_meta(instance, "full", time_limit_per_config=time_limit_per_config,
                        limit=limit, sample_seed=seed if limit else None),
    ) if output_base else None

    if limit is not None and limit < len(all_configs):
        log.write(f"[Full] TIMING TEST: {len(configs)}/{len(all_configs)} feasible "
                  f"config(s), random sample (seed={seed})")
        log.write("[Full] NOTE: C*/corridor/winner below are only meaningful for "
                  "this sample, not the full instance -- use this run to time a "
                  "full run, not to draw conclusions about it.")
    else:
        log.write(f"[Full] {len(configs)} feasible config(s)")

    theta, phi, _ = st.compute_measures(instance, all_configs, all_ids)
    theta = {i: theta[i] for i in ids}
    phi = {i: phi[i] for i in ids}

    t0 = time.time()
    results, run_info = sch.schedule_configs(
        instance, configs, config_ids=ids,
        time_limit_per_config=time_limit_per_config,
        logger=log, json_writer=writer, verbose=verbose,
    )
    elapsed = time.time() - t0

    log.write(f"[Full] {run_info['n_time_limit_hits']}/{len(results)} config(s) "
              f"did not reach a proven optimum within time_limit_per_config="
              f"{time_limit_per_config}s")

    if limit is not None and limit < len(all_configs):
        avg_per_config = elapsed / len(configs)
        projected_s = avg_per_config * len(all_configs)
        log.write(f"[Full] sample took {elapsed:.1f}s for {len(configs)} config(s) "
                  f"-> {avg_per_config:.2f}s/config average")
        log.write(f"[Full] projected full run ({len(all_configs)} configs): "
                  f"~{projected_s/3600:.1f}h ({projected_s:.0f}s)")

    out = finalize(instance, results, theta, phi, log, writer)
    log.close()
    return out


# ---------------------------------------------------------------------------
# Mode: two-pass
# ---------------------------------------------------------------------------

def run_two_pass(instance: cs.RMSInstance, output_base: str,
                 time_limit_per_config: int, refine_time_limit: int,
                 refine_corridor_factor: float, verbose: bool = True) -> dict:
    configs = cs.feasible_configs(instance)
    ids = list(range(len(configs)))

    log = sch.RunLogger(f"{output_base}.log" if output_base else None)
    writer = sch.JsonRunWriter(
        output_base,
        meta=build_meta(instance, "two-pass",
                        time_limit_per_config=time_limit_per_config,
                        refine_time_limit=refine_time_limit,
                        refine_corridor_factor=refine_corridor_factor),
    ) if output_base else None

    log.write(f"[Two-pass] {len(configs)} feasible config(s), pass 1 "
              f"(time_limit={time_limit_per_config}s)")

    theta, phi, _ = st.compute_measures(instance, configs, ids)

    pass1_results, run_info = sch.schedule_configs(
        instance, configs, config_ids=ids,
        time_limit_per_config=time_limit_per_config,
        logger=log, json_writer=writer, verbose=verbose,
    )

    total_cost = ev.compute_total_cost(pass1_results, instance.alpha)
    subset = ev.refinement_subset(
        pass1_results, total_cost, refine_corridor_factor, instance.epsilon
    )
    subset_configs = [r["y"] for r in subset]
    subset_ids = [r["config_idx"] for r in subset]

    log.write(f"[Two-pass] pass 1 provisional best total_cost = "
              f"{min(total_cost.values()):.4f}")
    log.write(f"[Two-pass] refining {len(subset_ids)} config(s) with "
              f"time_limit={refine_time_limit}s")

    refined_results, _ = sch.schedule_configs(
        instance, subset_configs, config_ids=subset_ids,
        time_limit_per_config=refine_time_limit,
        logger=log, json_writer=writer, verbose=verbose,
    )

    refined_by_id = {r["config_idx"]: r for r in refined_results}
    merged = [refined_by_id.get(r["config_idx"], r) for r in pass1_results]

    out = finalize(instance, merged, theta, phi, log, writer)
    log.close()
    return out


def run_simulate_partial_inversion(instance: cs.RMSInstance, baseline_run_path: str,
                                   phi_top_fraction: float, output_base: str,
                                   verbose: bool = True) -> dict:
    """
    Approximates what a partial-inversion run would find, WITHOUT calling
    Gurobi: reuses the baseline's own already-solved Q/C_config/perf for
    whichever configs fall in the top phi_top_fraction by Phi, instead of
    re-scheduling them.

    CAVEAT: this is a lookup-based approximation, not a substitute for an
    actual partial-inversion run. The baseline's Q values were solved
    once; a real re-solve of the same configs (even at the same
    time_limit) can differ slightly due to MIP solver stochasticity (see
    the ~0.999, not exactly 1.0, Spearman rho observed between a real
    heuristic run and the baseline on common configs). Use this for fast
    exploration of several phi_top_fraction values before committing
    Gurobi time to the one you actually want to run for real.
    """
    log = sch.RunLogger(f"{output_base}.log" if output_base else None)
    writer = sch.JsonRunWriter(
        output_base,
        meta=build_meta(instance, "simulate-partial-inversion",
                        baseline_run=baseline_run_path,
                        phi_top_fraction=phi_top_fraction,
                        note="NO Gurobi calls -- reuses baseline's own Q values"),
    ) if output_base else None

    log.write(f"[Simulate] NO Gurobi calls -- reusing Q/C_config/perf from {baseline_run_path}")
    log.write("[Simulate] CAVEAT: approximates a real partial-inversion run; treat as a fast "
              "preview for choosing phi_top_fraction, not as final numbers.")

    baseline_results = load_scheduling_results(baseline_run_path, instance)
    universe_configs = [r["y"] for r in baseline_results]
    universe_ids = [r["config_idx"] for r in baseline_results]
    theta, phi, _ = st.compute_measures(instance, universe_configs, universe_ids)

    sorted_ids = sorted(universe_ids, key=lambda i: phi[i], reverse=True)
    keep_n = max(1, round(len(sorted_ids) * phi_top_fraction))
    kept_ids = set(sorted_ids[:keep_n])
    kept_results = [r for r in baseline_results if r["config_idx"] in kept_ids]

    log.write(f"[Simulate] keeping top {phi_top_fraction:.0%} by Phi: "
              f"{len(kept_results)}/{len(baseline_results)} config(s)")

    if writer:
        for r in kept_results:
            writer.add_config_result({
                "config_idx":      r["config_idx"],
                "y":               cs.config_to_dict(r["y"]),
                "C_config":        r["C_config"],
                "Q":               r["Q"],
                "perf":            r["perf"],
                "solver_status":   r.get("solve_info", {}).get("status"),
                "mip_gap_pct":     r.get("solve_info", {}).get("gap"),
                "activity":        r.get("activity"),
                "gantt":           r.get("gantt"),
            })

    out = finalize(instance, kept_results, theta, phi, log, writer)
    log.close()
    return out


# ---------------------------------------------------------------------------
# Mode: partial-inversion
# ---------------------------------------------------------------------------

def run_partial_inversion(instance: cs.RMSInstance, output_base: str,
                          phi_top_fraction: float, time_limit_per_config: int,
                          verbose: bool = True) -> dict:
    if not 0.0 < phi_top_fraction <= 1.0:
        raise ValueError("phi_top_fraction must be in (0, 1].")

    configs = cs.feasible_configs(instance)
    ids = list(range(len(configs)))

    log = sch.RunLogger(f"{output_base}.log" if output_base else None)
    writer = sch.JsonRunWriter(
        output_base,
        meta=build_meta(instance, "partial-inversion",
                        time_limit_per_config=time_limit_per_config,
                        phi_top_fraction=phi_top_fraction),
    ) if output_base else None

    log.write(f"[Partial-inversion] {len(configs)} feasible config(s) -- "
              f"computing Phi over the full universe first")

    theta, phi, _ = st.compute_measures(instance, configs, ids)

    sorted_ids = sorted(ids, key=lambda i: phi[i], reverse=True)
    keep_n = max(1, round(len(sorted_ids) * phi_top_fraction))
    kept_ids = sorted(sorted_ids[:keep_n])
    kept_configs = [configs[i] for i in kept_ids]

    log.write(f"[Partial-inversion] keeping top {phi_top_fraction:.0%} by Phi: "
              f"{len(kept_ids)}/{len(ids)} config(s) will be scheduled")
    log.write("[Partial-inversion] NOTE: this is a heuristic -- the true C* "
              "may lie in a discarded (low-Phi) config; not a substitute "
              "for the full baseline run.")

    results, run_info = sch.schedule_configs(
        instance, kept_configs, config_ids=kept_ids,
        time_limit_per_config=time_limit_per_config,
        logger=log, json_writer=writer, verbose=verbose,
    )

    out = finalize(instance, results, theta, phi, log, writer)
    log.close()
    return out


# ---------------------------------------------------------------------------
# Mode: recompute (no scheduling -- replacement for explore_phase2.py)
# ---------------------------------------------------------------------------

def load_scheduling_results(run_json_path: str, instance: cs.RMSInstance) -> List[dict]:
    """
    Reload a previous run's scheduling results and reconstruct each
    entry's SystemConfig against the (possibly updated) instance file --
    exactly like the old explore_phase2.py did against
    rms_bilevel_gurobi.py output.

    Accepts two formats, auto-detected:
      - legacy single-file format: a JSON file with a top-level "configs"
        list (e.g. an old rms_bilevel_gurobi.py / explore_phase2.py run,
        or anything produced by the earlier single-file JsonRunWriter).
      - current JSONL-sidecar format: a "<path_base>.meta.json" file
        with a "configs_file" key pointing to a sibling "*.configs.jsonl"
        (see scheduling.JsonRunWriter).

    Returns a list of dicts shaped like scheduling.schedule_configs'
    results (config_idx, y, C_config, Q, perf; activity/gantt kept if
    present in the source).
    """
    with open(run_json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "configs" in data and isinstance(data["configs"], list):
        raw_configs = data["configs"]
    elif "configs_file" in data:
        jsonl_path = os.path.join(os.path.dirname(run_json_path), data["configs_file"])
        raw_configs = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_configs.append(json.loads(line))
    else:
        raise ValueError(
            f"{run_json_path}: unrecognized run file format (no 'configs' "
            f"list and no 'configs_file' pointer)."
        )

    # Dedupe by config_idx, keeping the *last* entry written for each --
    # a config can legitimately be scheduled more than once within a
    # single run (e.g. two-pass refinement re-schedules a subset on top
    # of pass 1, appending to the same writer), and the later write is
    # always the more authoritative one. Without this, a reloaded
    # two-pass run would double-count refined configs downstream (wrong
    # n_scheduled, and structural.compute_measures would see the same
    # config twice, corrupting neighbour counts and Phi).
    raw_by_idx: Dict[int, dict] = {}
    for c in raw_configs:
        raw_by_idx[c["config_idx"]] = c
    raw_configs = list(raw_by_idx.values())

    mc_lookup: Dict[tuple, cs.MachineConfig] = {}
    for s, stage_configs in enumerate(instance.stage_configs):
        for mc in stage_configs:
            mc_lookup[(s, mc.machine_id, mc.config_id)] = mc

    results = []
    for c in raw_configs:
        mc_list, n_list = [], []
        for stage_entry in c["y"]:
            key = (stage_entry["stage"], stage_entry["machine_id"], stage_entry["config_id"])
            if key not in mc_lookup:
                raise KeyError(
                    f"Config (stage={key[0]}, machine={key[1]}, config={key[2]}) "
                    f"from {run_json_path} not found in the instance file -- "
                    f"are you sure this run.json matches this instance?"
                )
            mc_list.append(mc_lookup[key])
            n_list.append(stage_entry["n_machines"])
        y = tuple(zip(mc_list, n_list))

        results.append({
            "config_idx": c["config_idx"],
            "y":          y,
            "C_config":   c["C_config"],
            "Q":          c["Q"],
            "perf":       c["perf"],
            "activity":   c.get("activity"),
            "gantt":      c.get("gantt"),
            "solve_info": {"status": c.get("solver_status"), "gap": c.get("mip_gap_pct")},
        })

    return results


def run_recompute(instance: cs.RMSInstance, run_json_path: str,
                  output_base: str = None, verbose: bool = True) -> dict:
    log = sch.RunLogger(f"{output_base}.log" if output_base else None)

    log.write(f"[Recompute] loading scheduling results from {run_json_path} "
              f"-- Gurobi is never called in this mode")
    log.write(f"[Recompute] using alpha={instance.alpha}  epsilon={instance.epsilon}  "
              f"tau={instance.tau}  lambda={instance.lambda_coef}  "
              f"mu={instance.mu_coef}  delta={instance.delta_coef}")

    results = load_scheduling_results(run_json_path, instance)

    # Theta/Phi over the FULL feasible universe, never over `results`'
    # own (possibly truncated) config list -- Phi is neighbourhood-
    # dependent, so recomputing it from a partial-inversion run's file
    # (which only persists its kept subset) would silently disagree with
    # the full-universe value orchestrate.py used internally when that
    # run first picked its winner. Harmless no-op for full/two-pass
    # files, which already contain the whole universe.
    universe_configs = cs.feasible_configs(instance)
    universe_ids = list(range(len(universe_configs)))
    theta, phi, _ = st.compute_measures(instance, universe_configs, universe_ids)

    writer = sch.JsonRunWriter(
        output_base,
        meta=build_meta(instance, "recompute", source_run_json=run_json_path),
    ) if output_base else None
    # Recompute mode doesn't re-run Gurobi, so per-config detail is just a
    # copy of what was reloaded -- write it through for a self-contained
    # output file, without re-persisting activity/gantt for every config
    # (already on disk in the source file; keep this file lighter).
    if writer is not None:
        for r in results:
            writer.add_config_result({
                "config_idx": r["config_idx"],
                "y":          cs.config_to_dict(r["y"]),
                "C_config":   r["C_config"],
                "Q":          r["Q"],
                "perf":       r["perf"],
            })

    out = finalize(instance, results, theta, phi, log, writer)
    log.close()
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def apply_overrides(instance: cs.RMSInstance, args) -> cs.RMSInstance:
    """Apply CLI overrides for alpha/epsilon/tau/lambda/mu/delta, if given."""
    for field_name, cli_val in [
        ("alpha", args.alpha), ("epsilon", args.epsilon), ("tau", args.tau),
        ("lambda_coef", args.lambda_coef), ("mu_coef", args.mu_coef),
        ("delta_coef", args.delta_coef),
    ]:
        if cli_val is not None:
            setattr(instance, field_name, cli_val)
    return instance


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["full", "two-pass", "partial-inversion",
                                         "simulate-partial-inversion", "recompute"])
    parser.add_argument("instance_file")
    parser.add_argument("--output", dest="output_base", default=None,
                        help="Base path for <output>.log / <output>.json")
    parser.add_argument("--baseline-run", default=None,
                        help="[simulate-partial-inversion] path to an existing baseline run to reuse")
    parser.add_argument("--time-limit", dest="time_limit", type=int, default=5,
                        help="Gurobi time limit per config, in seconds (full / "
                             "two-pass pass-1 / partial-inversion)")
    parser.add_argument("--limit", type=int, default=None,
                        help="[full] schedule only a random sample of this many "
                             "feasible configs, and project the full run's "
                             "duration from it -- for timing tests before "
                             "committing to a full (potentially many-hour) run")
    parser.add_argument("--seed", type=int, default=42,
                        help="[full, with --limit] random seed for the sample")
    parser.add_argument("--refine-time-limit", type=int, default=20,
                        help="[two-pass] Gurobi time limit for the refinement pass")
    parser.add_argument("--refine-corridor-factor", type=float, default=2.0,
                        help="[two-pass] width multiplier on epsilon for the "
                             "refinement corridor")
    parser.add_argument("--phi-top-fraction", type=float, default=0.3,
                        help="[partial-inversion] fraction of the feasible "
                             "universe to keep, ranked by Phi, before scheduling")
    parser.add_argument("--run-json", default=None,
                        help="[recompute] path to a previous run's JSON output")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--lambda-coef", dest="lambda_coef", type=float, default=None)
    parser.add_argument("--mu-coef", dest="mu_coef", type=float, default=None)
    parser.add_argument("--delta-coef", dest="delta_coef", type=float, default=None)

    args = parser.parse_args()

    instance = cs.load_instance_json(args.instance_file)
    instance = apply_overrides(instance, args)

    if args.mode == "full":
        run_full(instance, args.output_base, args.time_limit,
                limit=args.limit, seed=args.seed)
    elif args.mode == "two-pass":
        run_two_pass(instance, args.output_base, args.time_limit,
                    args.refine_time_limit, args.refine_corridor_factor)
    elif args.mode == "partial-inversion":
        run_partial_inversion(instance, args.output_base,
                              args.phi_top_fraction, args.time_limit)
    elif args.mode == "simulate-partial-inversion":
        if not args.baseline_run:
            parser.error("simulate-partial-inversion requires --baseline-run")
        run_simulate_partial_inversion(instance, args.baseline_run,
                                       args.phi_top_fraction, args.output_base)
    elif args.mode == "recompute":
        if not args.run_json:
            parser.error("recompute mode requires --run-json")
        run_recompute(instance, args.run_json, args.output_base)


if __name__ == "__main__":
    main()