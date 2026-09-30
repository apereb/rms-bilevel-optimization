"""
Parameter sweeps for epsilon, tau, and phi_top_fraction -- the mechanical
counterpart of heuristic_analysis.py's alpha_sweep, for the three other
bi-level parameters that can be swept from a single existing `full` (or
`two-pass`) baseline run, with NO new Gurobi calls.

(The fourth natural sweep, over refine_corridor_factor, is already
covered by `heuristic_analysis.py twopass-preview --refine-corridor-factors
f1,f2,...` -- it already loops over a comma-separated list, so it is not
duplicated here.)

epsilon-sweep
    Recomputes the (1+epsilon) corridor and Phase-2 winner for each
    epsilon, from the baseline's own total_cost (alpha fixed, tau fixed
    -- Theta/Phi computed once). Cheapest of the three: corridor
    membership is the only thing that changes.

tau-sweep
    Recomputes Phi and the reconfiguration neighbourhoods for each tau
    (alpha/epsilon fixed, so the corridor itself is FIXED across the
    sweep -- only which corridor member maximizes Phi can change).
    Also reports the neighbourhood-size distribution at each tau, over
    the full feasible universe, to help spot the saturation point where
    growing tau stops meaningfully growing neighbourhoods (e.g. because
    every same-machine-type config is already mutually reachable).
    This is the expensive one: Phi/neighbourhoods are recomputed from
    scratch (O(n^2) within machine-type buckets) at every tau value.

phi-fraction-sweep
    Loops orchestrate.run_simulate_partial_inversion's own logic (reuse
    the baseline's already-solved Q values for whichever configs fall in
    the top phi_top_fraction by Phi -- no Gurobi) over a grid of
    fractions, and reports how the resulting C*/corridor/winner compare
    to the true baseline at each fraction. Complements
    heuristic_analysis.py's phi-recall (which only checks whether the
    true corridor *survives* the cut) by showing what the pipeline would
    actually have concluded at each fraction, including a corridor that
    may be biased by exclusion of the true C*.

Usage
-----
    python param_sweep.py epsilon-sweep instance.json baseline.meta.json \
        --epsilons 0.05,0.1,0.15,0.2,0.3

    python param_sweep.py tau-sweep instance.json baseline.meta.json \
        --taus 1,2,3,5,8,13,21

    python param_sweep.py phi-fraction-sweep instance.json baseline.meta.json \
        --fractions 0.1,0.2,0.3,0.5,0.7,1.0
"""

import argparse
import dataclasses
from typing import Dict, List, Tuple

import config_space as cs
import evaluation as ev
import orchestrate as orch
import structural as st

from heuristic_analysis import jaccard  # reuse, no need to redefine


# ---------------------------------------------------------------------------
# epsilon-sweep
# ---------------------------------------------------------------------------

def epsilon_sweep(instance: cs.RMSInstance, baseline_results: List[dict],
                  epsilons: List[float]) -> List[dict]:
    """
    Recompute the corridor and Phase-2 winner for each epsilon, holding
    alpha and tau fixed. Theta/Phi are computed once over the full
    feasible universe (same convention as orchestrate.run_recompute and
    heuristic_analysis.compare_baseline_vs_heuristic), so this is correct
    even if baseline_results only covers a subset of the universe (e.g.
    a partial-inversion baseline) -- total_cost, on the other hand, only
    exists for configs baseline_results actually scheduled, so the
    corridor and winner at each epsilon are necessarily drawn from that
    same scheduled subset.
    """
    universe_configs = cs.feasible_configs(instance)
    universe_ids = list(range(len(universe_configs)))
    theta, phi, _ = st.compute_measures(instance, universe_configs, universe_ids)

    total_cost = ev.compute_total_cost(baseline_results, instance.alpha)
    by_idx = {r["config_idx"]: r for r in baseline_results}

    rows = []
    for eps in epsilons:
        C_star, threshold, corridor_entries = ev.compute_corridor(
            baseline_results, total_cost, eps
        )
        winner = ev.select_phase2_winner(corridor_entries, phi)
        w = by_idx[winner["config_idx"]] if winner else None
        rows.append({
            "epsilon": eps,
            "C_star": C_star,
            "threshold": threshold,
            "corridor_size": len(corridor_entries),
            "winner_idx": winner["config_idx"] if winner else None,
            "winner_C_config": w["C_config"] if w else None,
            "winner_Q": w["Q"] if w else None,
            "winner_Phi": phi[winner["config_idx"]] if winner else None,
        })
    return rows


def print_epsilon_sweep_table(rows: List[dict]):
    print("| epsilon | C* | threshold | corridor_size | winner_idx | "
          "winner_C_config | winner_Q | winner_Phi |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['epsilon']:g} | {r['C_star']:.4f} | {r['threshold']:.4f} | "
              f"{r['corridor_size']} | {r['winner_idx']} | "
              f"{r['winner_C_config']:.2f} | {r['winner_Q']:.2f} | "
              f"{r['winner_Phi']:.4f} |")


# ---------------------------------------------------------------------------
# tau-sweep
# ---------------------------------------------------------------------------

def _neighbourhood_stats(sizes: List[int]) -> dict:
    """Cheap distribution summary, no numpy dependency."""
    n = len(sizes)
    s = sorted(sizes)

    def pct(p):
        if n == 0:
            return float("nan")
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return s[idx]

    return {
        "min": s[0] if n else float("nan"),
        "p25": pct(0.25),
        "median": pct(0.50),
        "mean": sum(sizes) / n if n else float("nan"),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "max": s[-1] if n else float("nan"),
        "frac_isolated": sum(1 for x in sizes if x == 0) / n if n else float("nan"),
    }


def tau_sweep(instance: cs.RMSInstance, baseline_results: List[dict],
             taus: List[float]) -> Tuple[List[dict], List[dict]]:
    """
    Recompute Phi and neighbourhoods for each tau, holding alpha/epsilon
    (hence the corridor itself) fixed -- only which corridor member
    maximizes Phi can change across the sweep. Theta doesn't depend on
    tau at all and is computed once; only Phi/neighbour_sets are redone
    per tau (the O(n^2)-within-buckets part of compute_measures).

    Returns (winner_rows, neighbourhood_rows):
      - winner_rows: one row per tau with corridor winner + its Phi.
      - neighbourhood_rows: one row per tau with the neighbourhood-size
        distribution over the FULL feasible universe (not just the
        corridor) -- this is what to look at for the saturation point.
    """
    universe_configs = cs.feasible_configs(instance)
    universe_ids = list(range(len(universe_configs)))

    total_cost = ev.compute_total_cost(baseline_results, instance.alpha)
    C_star, threshold, corridor_entries = ev.compute_corridor(
        baseline_results, total_cost, instance.epsilon
    )
    by_idx = {r["config_idx"]: r for r in baseline_results}

    winner_rows = []
    neighbourhood_rows = []
    for tau in taus:
        inst_tau = dataclasses.replace(instance, tau=tau)
        theta, phi, neighbour_sets = st.compute_measures(
            inst_tau, universe_configs, universe_ids
        )

        winner = ev.select_phase2_winner(corridor_entries, phi)
        w = by_idx[winner["config_idx"]] if winner else None
        winner_rows.append({
            "tau": tau,
            "corridor_size": len(corridor_entries),
            "winner_idx": winner["config_idx"] if winner else None,
            "winner_C_config": w["C_config"] if w else None,
            "winner_Q": w["Q"] if w else None,
            "winner_Theta": theta[winner["config_idx"]] if winner else None,
            "winner_Phi": phi[winner["config_idx"]] if winner else None,
        })

        sizes = [len(neighbour_sets[cid]) for cid in universe_ids]
        stats = _neighbourhood_stats(sizes)
        neighbourhood_rows.append({"tau": tau, **stats})

    return winner_rows, neighbourhood_rows


def print_tau_sweep_tables(winner_rows: List[dict], neighbourhood_rows: List[dict]):
    print("Corridor winner vs. tau (corridor itself is fixed -- alpha/epsilon unchanged):")
    print("| tau | corridor_size | winner_idx | winner_C_config | winner_Q | "
          "winner_Theta | winner_Phi |")
    print("|---|---|---|---|---|---|---|")
    for r in winner_rows:
        print(f"| {r['tau']:g} | {r['corridor_size']} | {r['winner_idx']} | "
              f"{r['winner_C_config']:.2f} | {r['winner_Q']:.2f} | "
              f"{r['winner_Theta']:.4f} | {r['winner_Phi']:.4f} |")
    print()
    print("Neighbourhood-size distribution over the FULL feasible universe "
          "(watch mean/median for the saturation point):")
    print("| tau | min | p25 | median | mean | p75 | p90 | max | frac_isolated |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in neighbourhood_rows:
        print(f"| {r['tau']:g} | {r['min']:g} | {r['p25']:g} | {r['median']:g} | "
              f"{r['mean']:.2f} | {r['p75']:g} | {r['p90']:g} | {r['max']:g} | "
              f"{r['frac_isolated']:.1%} |")


# ---------------------------------------------------------------------------
# phi-fraction-sweep
# ---------------------------------------------------------------------------

def phi_fraction_sweep(instance: cs.RMSInstance, baseline_results: List[dict],
                       fractions: List[float]) -> List[dict]:
    """
    Loops orchestrate.run_simulate_partial_inversion's own kept-subset
    logic (reuse the baseline's own solved Q/C_config/perf for whichever
    configs fall in the top `fraction` by Phi -- no Gurobi) over a grid
    of fractions, and evaluates each kept subset exactly the way the
    real pipeline would (ev.evaluate), rather than just checking
    corridor recall as heuristic_analysis.phi_recall_curve does.

    Theta/Phi are computed once over baseline_results' own config set
    (matching orchestrate.run_simulate_partial_inversion's own
    convention -- if baseline_results is a `full` run this IS the whole
    universe; if baseline_results is itself already partial, this sweep
    is exploring fractions of THAT subset, which is what
    simulate-partial-inversion also does when pointed at a non-full
    baseline).
    """
    ids = [r["config_idx"] for r in baseline_results]
    configs = [r["y"] for r in baseline_results]
    theta, phi, _ = st.compute_measures(instance, configs, ids)

    b_out = ev.evaluate(instance, baseline_results, theta, phi)
    baseline_winner_idx = b_out["winner"]["config_idx"] if b_out["winner"] else None

    sorted_ids = sorted(ids, key=lambda i: phi[i], reverse=True)
    by_idx = {r["config_idx"]: r for r in baseline_results}

    rows = []
    for frac in fractions:
        keep_n = max(1, round(len(sorted_ids) * frac))
        kept_ids = set(sorted_ids[:keep_n])
        kept_results = [r for r in baseline_results if r["config_idx"] in kept_ids]

        out = ev.evaluate(instance, kept_results, theta, phi)
        winner_idx = out["winner"]["config_idx"] if out["winner"] else None

        cost_gap_pct = None
        if out["winner"] is not None and b_out["C_star"] > 0:
            cost_gap_pct = (100.0 * (out["winner"]["total_cost"] - b_out["C_star"])
                            / b_out["C_star"])

        rows.append({
            "fraction": frac,
            "kept_n": len(kept_results),
            "C_star": out["C_star"],
            "corridor_size": len(out["corridor"]),
            "winner_idx": winner_idx,
            "winner_matches_baseline": (
                (winner_idx == baseline_winner_idx) if winner_idx is not None else None
            ),
            "cost_gap_pct": cost_gap_pct,
        })

    return rows


def print_phi_fraction_sweep_table(rows: List[dict], baseline_winner_idx):
    print(f"Baseline (fraction=1.0 equivalent) winner: config_idx={baseline_winner_idx}")
    print()
    print("| fraction | kept_n | C* | corridor_size | winner_idx | "
          "winner==baseline | cost_gap_vs_true_C* |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        gap_str = f"+{r['cost_gap_pct']:.2f}%" if r["cost_gap_pct"] is not None else "n/a"
        print(f"| {r['fraction']:.0%} | {r['kept_n']} | {r['C_star']:.4f} | "
              f"{r['corridor_size']} | {r['winner_idx']} | "
              f"{r['winner_matches_baseline']} | {gap_str} |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", choices=["epsilon-sweep", "tau-sweep", "phi-fraction-sweep"])
    parser.add_argument("instance_file")
    parser.add_argument("baseline_run")
    parser.add_argument("--epsilons", default="0.05,0.1,0.15,0.2,0.3",
                        help="[epsilon-sweep] comma-separated epsilon values")
    parser.add_argument("--taus", default=None,
                        help="[tau-sweep] comma-separated tau values")
    parser.add_argument("--fractions", default="0.1,0.2,0.3,0.5,0.7,1.0",
                        help="[phi-fraction-sweep] comma-separated phi_top_fraction values")

    args = parser.parse_args()
    instance = cs.load_instance_json(args.instance_file)
    baseline_results = orch.load_scheduling_results(args.baseline_run, instance)

    if args.analysis == "epsilon-sweep":
        epsilons = [float(e) for e in args.epsilons.split(",")]
        rows = epsilon_sweep(instance, baseline_results, epsilons)
        print_epsilon_sweep_table(rows)

    elif args.analysis == "tau-sweep":
        if not args.taus:
            parser.error("'tau-sweep' requires --taus")
        taus = [float(t) for t in args.taus.split(",")]
        winner_rows, neighbourhood_rows = tau_sweep(instance, baseline_results, taus)
        print_tau_sweep_tables(winner_rows, neighbourhood_rows)

    elif args.analysis == "phi-fraction-sweep":
        fractions = [float(f) for f in args.fractions.split(",")]
        rows = phi_fraction_sweep(instance, baseline_results, fractions)
        ids = [r["config_idx"] for r in baseline_results]
        configs = [r["y"] for r in baseline_results]
        theta, phi, _ = st.compute_measures(instance, configs, ids)
        b_out = ev.evaluate(instance, baseline_results, theta, phi)
        baseline_winner_idx = b_out["winner"]["config_idx"] if b_out["winner"] else None
        print_phi_fraction_sweep_table(rows, baseline_winner_idx)


if __name__ == "__main__":
    main()