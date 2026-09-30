"""
Offline test of a combinatorial lower bound on the Phase-1 optimum, used to
anchor the degradation corridor so that the partial-inversion cut cannot
shift it.

    LB_cost(y) = C_config(y) + alpha * LB_Q(y) / batch_size      (<= total_cost(y))
    anchor     = min_y LB_cost(y)   over the FULL universe (no scheduling)
    corridor'  = { y kept by the Phi cut : total_cost(y) <= (1 + eps') * anchor }

Needs only a `full` baseline run. No Gurobi solve is performed (scheduling.py
is imported only for build_scheduled_jobs).

Usage
-----
    python lb_analysis.py instance.json baseline.meta.json \
        --eps-primes 0.15,0.2,0.3,0.4,0.5 --phi-fractions 1.0,0.5,0.3
"""

import argparse
import statistics
from typing import List, Sequence


# The bound itself lives in lower_bounds.py (shared with orchestrate.py, mode lb-exact).
from lower_bounds import effective_machines, makespan_lb  # noqa: E402


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _stats(values: List[float]) -> str:
    if not values:
        return "n/a"
    return (f"min {min(values):.3f} / median {statistics.median(values):.3f} "
            f"/ max {max(values):.3f}")


def lb_analysis(instance, baseline_results: List[dict],
                eps_primes: List[float], phi_fractions: List[float],
                ms: List[int], tol: float = 1e-3) -> None:
    import config_space as cs
    import evaluation as ev
    import scheduling as sched
    import structural as st

    scheduled_jobs, batch_size = sched.build_scheduled_jobs(instance)
    universe = cs.feasible_configs(instance)
    ids = list(range(len(universe)))

    # --- LB for every config of the universe (no scheduling) --------------
    lb_q, lb_cost = {}, {}
    for i, y in zip(ids, universe):
        pt = cs.build_pt_matrix(y, instance, scheduled_jobs)
        lb_q[i] = makespan_lb(pt, effective_machines(y, pt))
        lb_cost[i] = cs.config_setup_cost(y) + instance.alpha * lb_q[i] / batch_size

    # --- baseline quantities ----------------------------------------------
    total_cost = ev.compute_total_cost(baseline_results, instance.alpha)
    C_star, threshold, corridor = ev.compute_corridor(
        baseline_results, total_cost, instance.epsilon)
    corridor_ids = {r["config_idx"] for r in corridor}
    _, phi, _ = st.compute_measures(instance, universe, ids)
    winner = ev.select_phase2_winner(corridor, phi)
    winner_id = winner["config_idx"] if winner else None
    by_idx = {r["config_idx"]: r for r in baseline_results}

    # --- 1. validity: LB_Q <= Q for every scheduled config -----------------
    gaps = [(r["config_idx"], lb_q[r["config_idx"]] - r["Q"]) for r in baseline_results]
    worst = max(g for _, g in gaps)
    n_pos = sum(1 for _, g in gaps if g > 1e-9)
    viol = [(i, g) for i, g in gaps if g > tol]
    print(f"[LB validity] {len(baseline_results)} configs checked; "
          f"worst LB_Q - Q = {worst:.2e}; {n_pos} above 0 (rounding-level if tiny); "
          f"{len(viol)} above tol={tol:g}")
    if viol:
        print("  -> genuine violation: the bound is NOT valid for this model; "
              f"examine these config_idx first: {[i for i, _ in viol[:10]]}")

    # --- 2. tightness -------------------------------------------------------
    ratio_all = [lb_cost[i] / total_cost[i] for i in total_cost
                 if total_cost[i] not in (0.0, float("inf"))]
    ratio_cor = [lb_cost[i] / total_cost[i] for i in corridor_ids]
    lb_min_id = min(ids, key=lambda i: (lb_cost[i], i))
    lb_min = lb_cost[lb_min_id]
    print(f"\n[Tightness] LB_cost / total_cost")
    print(f"  all configs : {_stats(ratio_all)}")
    print(f"  corridor    : {_stats(ratio_cor)}")
    if winner_id is not None:
        print(f"  baseline P2 winner (idx {winner_id}): "
              f"{lb_cost[winner_id] / total_cost[winner_id]:.3f}")
    print(f"\n[Anchor] LB_min = {lb_min:.4f} (config_idx {lb_min_id})   "
          f"C* = {C_star:.4f}   LB_min / C* = {lb_min / C_star:.3f}")
    eps_exact = (1 + instance.epsilon) * C_star / lb_min - 1
    print(f"  eps' reproducing the baseline threshold exactly : {eps_exact:.3f}")
    if winner_id is not None:
        eps_keep = total_cost[winner_id] / lb_min - 1
        print(f"  smallest eps' keeping the baseline P2 winner     : {eps_keep:.3f}")
    n_unprunable = sum(1 for i in ids if lb_cost[i] <= threshold)
    print(f"  configs with LB_cost <= (1+eps)C* (cannot be pruned by their own LB): "
          f"{n_unprunable} / {len(ids)} ({n_unprunable / len(ids):.1%})")

    # --- 2b. Phi rank of the baseline winner --------------------------------
    order = sorted(ids, key=lambda i: (-phi[i], i))
    if winner_id is not None:
        rank = order.index(winner_id) + 1
        print(f"\n[Phi] baseline P2 winner has Phi rank {rank} / {len(order)} "
              f"(kept by any phi_top_fraction >= {rank / len(order):.1%})")

    # --- 2c. exact anchor by best-first solving in LB order -----------------
    # Solve configs in increasing LB_cost; stop when the next LB >= best solved
    # cost. The best solved cost is then C* exactly (same 15 s incumbents).
    # Simulated here with the baseline's own costs, no Gurobi call.
    by_lb = sorted(ids, key=lambda i: (lb_cost[i], i))
    best, n_solved, solved_set = float("inf"), 0, set()
    for i in by_lb:
        if lb_cost[i] >= best:
            break
        best = min(best, total_cost[i])
        n_solved += 1
        solved_set.add(i)
    print(f"\n[Certified anchor] best-first in LB order: {n_solved} solves "
          f"({n_solved / len(ids):.1%} of the universe) give C* = {best:.4f} "
          f"(baseline C* = {C_star:.4f}, match: {abs(best - C_star) < 1e-9})")

    # --- 2c'. exact selection: walk LB-eligible configs in decreasing Phi ---
    # Only configs with LB_cost <= (1+eps)*C* can be in the corridor. Solve them
    # in decreasing Phi and stop at the first whose real cost is inside the
    # threshold: it is argmax Phi over the corridor, i.e. the baseline winner
    # (up to Phi ties). No phi_top_fraction needed.
    thr_exact = (1 + instance.epsilon) * best
    eligible = [i for i in order if lb_cost[i] <= thr_exact]
    walked, hit = set(), None
    for i in eligible:
        walked.add(i)
        if total_cost[i] <= thr_exact:
            hit = i
            break
    n_total_solves = len(solved_set | walked)
    print(f"\n[Exact selection] {len(eligible)} LB-eligible configs "
          f"({len(eligible) / len(ids):.1%}); walking them by decreasing Phi hits the "
          f"first in-corridor config after {len(walked)} solves -> idx {hit} "
          f"(baseline winner {winner_id}, match: {hit == winner_id})")
    print(f"  total solves incl. anchor: {n_total_solves} "
          f"({n_total_solves / len(ids):.1%} of the universe)")

    # --- 2d. UB anchor from the m configs with smallest LB_cost -------------
    print("\n[UB anchor] anchor = min total_cost over the m smallest-LB configs "
          "(>= C*, so corridor is a superset of the baseline one)")
    print("| m | UB_m | UB_m / C* | corridor size | extra vs baseline corridor |")
    print("|---|---|---|---|---|")
    ub_of = {}
    for m in ms:
        top = by_lb[:m]
        ub = min(total_cost[i] for i in top)
        ub_of[m] = ub
        cor_m = {r["config_idx"] for r in baseline_results
                 if total_cost[r["config_idx"]] <= (1 + instance.epsilon) * ub}
        print(f"| {m} | {ub:.4f} | {ub / C_star:.4f} | {len(cor_m)} | "
              f"{len(cor_m - corridor_ids)} |")

    # --- 3. eps' x Phi-fraction sweep ---------------------------------------
    print("\n[Sweep] partial-inversion with anchored corridor "
          f"(baseline P2 winner = {winner_id}, baseline corridor = {len(corridor_ids)})")
    print("| phi_fraction | kept | anchor | eps' | corridor size | extra vs baseline "
          "corridor | P2 winner | = baseline winner |")
    print("|---|---|---|---|---|---|---|---|")
    for frac in phi_fractions:
        keep_n = max(1, round(len(order) * frac))
        kept = set(order[:keep_n])
        kept_entries = [r for r in baseline_results if r["config_idx"] in kept]
        if not kept_entries:
            continue

        # current heuristic: anchor = best cost among the KEPT configs (shifted)
        c_kept = min(total_cost[r["config_idx"]] for r in kept_entries)
        cor_s = [r for r in kept_entries
                 if total_cost[r["config_idx"]] <= (1 + instance.epsilon) * c_kept]
        w_s = ev.select_phase2_winner(cor_s, phi) if cor_s else None
        w_s_id = w_s["config_idx"] if w_s else None
        extra_s = len({r["config_idx"] for r in cor_s} - corridor_ids)
        print(f"| {frac:.0%} | {keep_n} | shifted (current) | {instance.epsilon:g} | "
              f"{len(cor_s)} | {extra_s} | {w_s_id} | {w_s_id == winner_id} |")

        for ep in eps_primes:
            thr = (1 + ep) * lb_min
            cor = [r for r in kept_entries if total_cost[r["config_idx"]] <= thr]
            w = ev.select_phase2_winner(cor, phi) if cor else None
            w_id = w["config_idx"] if w else None
            extra = len({r["config_idx"] for r in cor} - corridor_ids)
            print(f"| {frac:.0%} | {keep_n} | LB_min | {ep:g} | {len(cor)} | "
                  f"{extra} | {w_id} | {w_id == winner_id} |")

        for m in ms:
            thr = (1 + instance.epsilon) * ub_of[m]
            cor = [r for r in kept_entries if total_cost[r["config_idx"]] <= thr]
            w = ev.select_phase2_winner(cor, phi) if cor else None
            w_id = w["config_idx"] if w else None
            extra = len({r["config_idx"] for r in cor} - corridor_ids)
            print(f"| {frac:.0%} | {keep_n} | UB(top-{m} by LB) | {instance.epsilon:g} | "
                  f"{len(cor)} | {extra} | {w_id} | {w_id == winner_id} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instance_file")
    parser.add_argument("baseline_run")
    parser.add_argument("--eps-primes", default="0.15,0.2,0.25,0.3,0.4,0.5",
                        help="comma-separated eps' values for the LB-anchored corridor")
    parser.add_argument("--phi-fractions", default="1.0,0.5,0.3",
                        help="comma-separated phi_top_fraction values to combine with it")
    parser.add_argument("--ms", default="5,10,25,50",
                        help="comma-separated m values for the UB anchor (top-m by LB)")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="absolute tolerance on LB_Q <= Q (MIP time-scale rounding)")
    args = parser.parse_args()

    import config_space as cs
    import orchestrate as orch

    instance = cs.load_instance_json(args.instance_file)
    baseline_results = orch.load_scheduling_results(args.baseline_run, instance)
    lb_analysis(instance, baseline_results,
                [float(x) for x in args.eps_primes.split(",")],
                [float(x) for x in args.phi_fractions.split(",")],
                [int(x) for x in args.ms.split(",")], args.tol)


if __name__ == "__main__":
    main()