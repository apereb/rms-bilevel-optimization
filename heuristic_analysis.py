"""
Analysis tools for judging a heuristic (two-pass, partial-inversion, or any
future one) against the full baseline run.

Two independent analyses:

phi-recall
    Uses the baseline run ALONE (no heuristic run needed). Sweeps a set of
    candidate phi_top_fraction values and asks, for each: "if
    partial-inversion had kept only the top X% of configs by Phi, would
    the true corridor (and the true winner) have survived the cut?" This
    tells you upfront whether a given fraction is even structurally
    capable of finding the right answer, before spending any scheduling
    time on it.

compare
    Needs both a baseline run and a heuristic run over the same instance.
    Reports:
      - corridor overlap (Jaccard) between the two runs' (1+epsilon)
        corridors
      - Spearman's rho and Kendall's tau between the two runs'
        total_cost rankings, restricted to configs both runs actually
        scheduled -- this is the "does the heuristic preserve relative
        standing, not just the winner" check: a heuristic could get the
        same winner by luck while scrambling everyone else's order,
        which this would catch and a plain winner-comparison wouldn't.
      - whether the baseline's true winner survived into the heuristic's
        corridor at all
      - the cost gap between the heuristic's chosen winner and the
        baseline's true C*

Usage
-----
    python heuristic_analysis.py phi-recall instance.json baseline.meta.json \
        --fractions 0.05,0.1,0.2,0.3,0.5,0.7,1.0

    python heuristic_analysis.py compare instance.json baseline.meta.json \
        partial_inversion.meta.json
"""

import argparse
from typing import Dict, List, Optional, Tuple

import config_space as cs
import evaluation as ev
import orchestrate as orch
import structural as st


# ---------------------------------------------------------------------------
# Rank correlation (pure Python -- no scipy dependency)
# ---------------------------------------------------------------------------

def _rank_data(values: List[float]) -> List[float]:
    """1-based ranks, ties broken by average rank (standard Spearman convention)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    varx = sum((xi - mx) ** 2 for xi in x)
    vary = sum((yi - my) ** 2 for yi in y)
    if varx == 0 or vary == 0:
        return float("nan")
    return cov / (varx ** 0.5 * vary ** 0.5)


def spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation. NaN if either series is constant."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return _pearson(_rank_data(x), _rank_data(y))


def kendall_tau(x: List[float], y: List[float]) -> float:
    """
    Kendall's tau-a (concordant - discordant) / total_pairs. Does not
    apply a tie correction (tau-b) -- ties are rare on continuous cost
    values in practice, and tau-a is simpler to reason about; revisit if
    ties turn out to matter for your data.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            prod = (x[i] - x[j]) * (y[i] - y[j])
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# phi-recall
# ---------------------------------------------------------------------------

def phi_recall_curve(instance: cs.RMSInstance, baseline_results: List[dict],
                     fractions: List[float]) -> Tuple[List[dict], dict]:
    """
    For each fraction in `fractions`, simulate keeping only the top
    `fraction` of the baseline's own configs by Phi, and report:
      - kept_n: how many configs that keeps
      - corridor_recall: what share of the TRUE (1+epsilon) corridor
        would have survived the cut
      - winner_included: whether the true Phase-2 winner survives

    Uses the baseline's own results only -- no heuristic run needed.
    """
    ids = [r["config_idx"] for r in baseline_results]
    configs = [r["y"] for r in baseline_results]
    theta, phi, _ = st.compute_measures(instance, configs, ids)

    total_cost = ev.compute_total_cost(baseline_results, instance.alpha)
    C_star, threshold, corridor_entries = ev.compute_corridor(
        baseline_results, total_cost, instance.epsilon
    )
    corridor_ids = {r["config_idx"] for r in corridor_entries}
    winner = ev.select_phase2_winner(corridor_entries, phi)
    winner_id = winner["config_idx"] if winner else None

    sorted_ids = sorted(ids, key=lambda i: phi[i], reverse=True)

    rows = []
    for frac in fractions:
        keep_n = max(1, round(len(sorted_ids) * frac))
        kept = set(sorted_ids[:keep_n])
        recall = (len(kept & corridor_ids) / len(corridor_ids)
                 if corridor_ids else float("nan"))
        rows.append({
            "fraction": frac,
            "kept_n": keep_n,
            "corridor_recall": recall,
            "winner_included": (winner_id in kept) if winner_id is not None else None,
        })

    context = {"C_star": C_star, "corridor_size": len(corridor_ids), "winner_idx": winner_id}
    return rows, context


def print_phi_recall_table(rows: List[dict], context: dict):
    print(f"[Phi-recall] baseline C* = {context['C_star']:.4f}  "
          f"corridor size = {context['corridor_size']}  "
          f"winner config_idx = {context['winner_idx']}")
    print()
    print("| fraction | kept_n | corridor_recall | winner_included |")
    print("|---|---|---|---|")
    for r in rows:
        recall_str = f"{r['corridor_recall']:.1%}" if r["corridor_recall"] == r["corridor_recall"] else "n/a"
        print(f"| {r['fraction']:.0%} | {r['kept_n']} | {recall_str} | {r['winner_included']} |")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def compare_baseline_vs_heuristic(instance: cs.RMSInstance,
                                  baseline_results: List[dict],
                                  heuristic_results: List[dict]) -> dict:
    """
    Theta/Phi are computed once over the FULL feasible universe and
    shared by both evaluate() calls below -- never over baseline_results
    or heuristic_results' own (possibly truncated) config list. A
    partial-inversion run's results only cover ~30% of the universe;
    computing Phi from that subset alone would give a biased value for
    the exact same config_idx compared to the full-universe value
    orchestrate.py used internally when the run first picked its winner,
    silently disagreeing with the run's own original result.
    """
    universe_configs = cs.feasible_configs(instance)
    universe_ids = list(range(len(universe_configs)))
    theta, phi, _ = st.compute_measures(instance, universe_configs, universe_ids)

    b_out = ev.evaluate(instance, baseline_results, theta, phi)
    h_out = ev.evaluate(instance, heuristic_results, theta, phi)

    b_corridor_ids = {r["config_idx"] for r in b_out["corridor"]}
    h_corridor_ids = {r["config_idx"] for r in h_out["corridor"]}

    b_cost = ev.compute_total_cost(baseline_results, instance.alpha)
    h_cost = ev.compute_total_cost(heuristic_results, instance.alpha)

    common_ids = sorted(set(b_cost) & set(h_cost))
    x = [b_cost[i] for i in common_ids]
    y = [h_cost[i] for i in common_ids]

    winner_cost_gap_pct = None
    if h_out["winner"] is not None and b_out["C_star"] > 0:
        winner_cost_gap_pct = 100.0 * (h_out["winner"]["total_cost"] - b_out["C_star"]) / b_out["C_star"]

    baseline_winner_survives = None
    if b_out["winner"] is not None:
        baseline_winner_survives = b_out["winner"]["config_idx"] in h_corridor_ids

    return {
        "n_baseline":   len(baseline_results),
        "n_heuristic":  len(heuristic_results),
        "n_common":     len(common_ids),
        "baseline_C_star":  b_out["C_star"],
        "heuristic_C_star": h_out["C_star"],
        "corridor_jaccard": jaccard(b_corridor_ids, h_corridor_ids),
        "spearman_rho": spearman_rho(x, y),
        "kendall_tau":  kendall_tau(x, y),
        "baseline_winner_idx":  b_out["winner"]["config_idx"] if b_out["winner"] else None,
        "heuristic_winner_idx": h_out["winner"]["config_idx"] if h_out["winner"] else None,
        "baseline_winner_survives_in_heuristic_corridor": baseline_winner_survives,
        "winner_cost_gap_pct": winner_cost_gap_pct,
    }


def print_compare_report(result: dict):
    print(f"Baseline:  {result['n_baseline']} scheduled, C* = {result['baseline_C_star']:.4f}, "
          f"winner = {result['baseline_winner_idx']}")
    print(f"Heuristic: {result['n_heuristic']} scheduled, C* = {result['heuristic_C_star']:.4f}, "
          f"winner = {result['heuristic_winner_idx']}")
    print(f"Common configs (both scheduled): {result['n_common']}")
    print()
    print(f"Corridor overlap (Jaccard):        {result['corridor_jaccard']:.3f}")
    print(f"Cost rank correlation (Spearman):  {result['spearman_rho']:.3f}")
    print(f"Cost rank correlation (Kendall):   {result['kendall_tau']:.3f}")
    print(f"Baseline winner survives in heuristic corridor: "
          f"{result['baseline_winner_survives_in_heuristic_corridor']}")
    if result["winner_cost_gap_pct"] is not None:
        print(f"Heuristic winner's cost vs. true C*: +{result['winner_cost_gap_pct']:.2f}%")


# ---------------------------------------------------------------------------
# twopass-preview
# ---------------------------------------------------------------------------

def _baseline_elapsed_seconds(baseline_run_path: str) -> Optional[float]:
    """Best-effort pass-1 wall time from the baseline's own meta/summary."""
    import json
    import time as time_mod
    with open(baseline_run_path, encoding="utf-8") as f:
        data = json.load(f)
    meta, summary = data.get("meta", {}), data.get("summary")
    if not summary or "started_at" not in meta or "finished_at" not in summary:
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t0 = time_mod.mktime(time_mod.strptime(meta["started_at"], fmt))
        t1 = time_mod.mktime(time_mod.strptime(summary["finished_at"], fmt))
        return t1 - t0
    except ValueError:
        return None


def twopass_preview(instance: cs.RMSInstance, baseline_results: List[dict],
                    refine_corridor_factors: List[float],
                    refine_time_limit: Optional[float] = None,
                    pass1_elapsed_s: Optional[float] = None) -> List[dict]:
    """
    For each refine_corridor_factor, compute how many baseline configs
    would fall in the refinement subset (evaluation.refinement_subset,
    the exact function orchestrate.run_two_pass uses) -- entirely from
    the baseline's own recorded Q/C_config/perf, no Gurobi call needed.

    If refine_time_limit and pass1_elapsed_s are given, also projects the
    two-pass run's total wall time as pass1_elapsed_s + n_subset *
    refine_time_limit (pass 1 is assumed identical to the baseline's own
    pass, an exact match if run with the same time_limit_per_config).
    """
    total_cost = ev.compute_total_cost(baseline_results, instance.alpha)
    n_total = len(baseline_results)

    rows = []
    for factor in refine_corridor_factors:
        subset = ev.refinement_subset(baseline_results, total_cost, factor, instance.epsilon)
        n_subset = len(subset)
        row = {
            "refine_corridor_factor": factor,
            "n_subset": n_subset,
            "proportion": n_subset / n_total if n_total else float("nan"),
        }
        if refine_time_limit is not None:
            refine_s = n_subset * refine_time_limit
            row["projected_refine_s"] = refine_s
            if pass1_elapsed_s is not None:
                row["projected_total_s"] = pass1_elapsed_s + refine_s
        rows.append(row)
    return rows


def print_twopass_preview_table(rows: List[dict], pass1_elapsed_s: Optional[float]):
    if pass1_elapsed_s is not None:
        print(f"[Two-pass preview] pass-1 (baseline) wall time: "
              f"{pass1_elapsed_s/3600:.1f}h ({pass1_elapsed_s:.0f}s)")
    print()
    has_time = any("projected_total_s" in r for r in rows)
    if has_time:
        print("| refine_corridor_factor | n_subset | proportion | projected refine time | projected total |")
        print("|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['refine_corridor_factor']:.1f} | {r['n_subset']} | "
                  f"{r['proportion']:.1%} | {r['projected_refine_s']/3600:.1f}h | "
                  f"{r['projected_total_s']/3600:.1f}h |")
    else:
        print("| refine_corridor_factor | n_subset | proportion |")
        print("|---|---|---|")
        for r in rows:
            print(f"| {r['refine_corridor_factor']:.1f} | {r['n_subset']} | {r['proportion']:.1%} |")


# ---------------------------------------------------------------------------
# alpha-sweep
# ---------------------------------------------------------------------------

def alpha_sweep(instance: cs.RMSInstance, baseline_results: List[dict],
                alphas: List[float], top_k: int = 20,
                reference_alpha: Optional[float] = None) -> Tuple[List[dict], dict]:
    """
    Recompute C*/corridor/winner under each alpha in `alphas`, from a
    single baseline's Q/C_config/perf -- no Gurobi needed, since alpha
    only enters through total_cost = C_config + alpha/perf in
    evaluation.py. Theta/Phi don't depend on alpha at all, so they're
    computed once and reused across the whole sweep.

    Also tracks whether the winner's apparent stability is representative
    of the broader picture or a fluke specific to that one config: at a
    `reference_alpha` (defaults to the instance's own recorded alpha),
    the top `top_k` configs by cost are fixed as a reference set, and for
    every other alpha we report (a) how much that set's *membership*
    changes (Jaccard) and (b) whether the *relative order* among those
    same reference configs holds up under the new alpha (Spearman/
    Kendall on their cost values, reference vs current alpha) -- the same
    "does the ranking scramble" check as compare_baseline_vs_heuristic,
    just along the alpha axis instead of across methods.
    """
    ids = [r["config_idx"] for r in baseline_results]
    configs = [r["y"] for r in baseline_results]
    theta, phi, _ = st.compute_measures(instance, configs, ids)
    by_idx = {r["config_idx"]: r for r in baseline_results}

    if reference_alpha is None:
        reference_alpha = instance.alpha

    ref_total_cost = ev.compute_total_cost(baseline_results, reference_alpha)
    ref_topk_ids = sorted(ids, key=lambda i: ref_total_cost[i])[:top_k]
    ref_topk_set = set(ref_topk_ids)

    rows = []
    for a in alphas:
        total_cost = ev.compute_total_cost(baseline_results, a)
        C_star, threshold, corridor_entries = ev.compute_corridor(baseline_results, total_cost, instance.epsilon)
        winner = ev.select_phase2_winner(corridor_entries, phi)
        w = by_idx[winner["config_idx"]] if winner else None

        topk_a_set = set(sorted(ids, key=lambda i: total_cost[i])[:top_k])
        x = [ref_total_cost[i] for i in ref_topk_ids]
        y = [total_cost[i] for i in ref_topk_ids]

        rows.append({
            "alpha": a,
            "C_star": C_star,
            "corridor_size": len(corridor_entries),
            "winner_idx": winner["config_idx"] if winner else None,
            "winner_C_config": w["C_config"] if w else None,
            "winner_Q": w["Q"] if w else None,
            "winner_perf": w["perf"] if w else None,
            "topk_jaccard_vs_ref": jaccard(ref_topk_set, topk_a_set),
            "topk_spearman_vs_ref": spearman_rho(x, y),
            "topk_kendall_vs_ref": kendall_tau(x, y),
        })

    context = {"reference_alpha": reference_alpha, "top_k": top_k, "reference_topk_ids": ref_topk_ids}
    return rows, context


def find_winner_breakpoints(rows: List[dict]) -> List[dict]:
    """
    Group consecutive alpha_sweep rows sharing the same winner_idx into
    stability intervals: [alpha_low, alpha_high] over which the Phase-2
    winner doesn't change. A real alpha value sitting well inside a wide
    interval is a robust choice; one sitting near a boundary is fragile
    and worth flagging as such.
    """
    intervals = []
    current = None
    for r in rows:
        if current is None or r["winner_idx"] != current["winner_idx"]:
            current = {"winner_idx": r["winner_idx"], "alpha_low": r["alpha"], "alpha_high": r["alpha"]}
            intervals.append(current)
        else:
            current["alpha_high"] = r["alpha"]
    return intervals


def print_alpha_sweep_table(rows: List[dict], intervals: List[dict], context: dict):
    print(f"Reference alpha = {context['reference_alpha']:g}  "
          f"(top-{context['top_k']} stability measured against this)")
    print()
    print("| alpha | C* | corridor_size | winner_idx | winner_C_config | winner_Q | "
          "topK_jaccard | topK_spearman | topK_kendall |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['alpha']:g} | {r['C_star']:.4f} | {r['corridor_size']} | "
              f"{r['winner_idx']} | {r['winner_C_config']:.2f} | {r['winner_Q']:.2f} | "
              f"{r['topk_jaccard_vs_ref']:.3f} | {r['topk_spearman_vs_ref']:.3f} | "
              f"{r['topk_kendall_vs_ref']:.3f} |")
    print()
    print("Stability intervals (winner unchanged within each):")
    for iv in intervals:
        span = f"[{iv['alpha_low']:g}, {iv['alpha_high']:g}]"
        print(f"  winner={iv['winner_idx']:<6}  alpha in {span}")


# ---------------------------------------------------------------------------
# pareto
# ---------------------------------------------------------------------------

def pareto_frontier(baseline_results: List[dict]) -> List[Tuple[int, float, float]]:
    """
    Pareto frontier of (C_config, Q) -- lower is better on both axes.
    O(n log n): sort by C_config ascending, then scan keeping the running
    minimum Q; a point is on the frontier iff its Q beats every point
    that came before it in C_config order (the classic skyline algorithm).

    Returns [(config_idx, C_config, Q), ...] sorted by increasing C_config
    -- i.e. the frontier itself, trading off "cheaper setup" against
    "faster/better makespan" independently of any alpha.
    """
    points = [(r["config_idx"], r["C_config"], r["Q"]) for r in baseline_results]
    points.sort(key=lambda p: (p[1], p[2]))

    frontier = []
    best_q = float("inf")
    for idx, c, q in points:
        if q < best_q:
            frontier.append((idx, c, q))
            best_q = q
    return frontier


def print_pareto_table(frontier: List[Tuple[int, float, float]], n_total: int):
    print(f"Pareto frontier: {len(frontier)} config(s) out of {n_total} "
          f"({len(frontier)/n_total:.1%}) are non-dominated on (C_config, Q)")
    print()
    print("| config_idx | C_config | Q |")
    print("|---|---|---|")
    for idx, c, q in frontier:
        print(f"| {idx} | {c:.2f} | {q:.2f} |")


def plot_pareto(baseline_results: List[dict], frontier: List[Tuple[int, float, float]],
                alpha_sweep_rows: Optional[List[dict]], output_png: str):
    """
    Scatter of every scheduled config on (C_config, Q), the Pareto
    frontier connected as a line, and (if given) the winner at each swept
    alpha overlaid as labeled markers -- visualizes exactly what alpha
    trades off, and where each candidate winner sits relative to the
    trade-off curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_c = [r["C_config"] for r in baseline_results]
    all_q = [r["Q"] for r in baseline_results]
    frontier_c = [c for _, c, _ in frontier]
    frontier_q = [q for _, _, q in frontier]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_c, all_q, s=8, alpha=0.25, color="steelblue", label="all scheduled configs")
    ax.plot(frontier_c, frontier_q, color="crimson", marker="o", markersize=4,
           linewidth=1.5, label="Pareto frontier")

    if alpha_sweep_rows:
        by_idx = {r["config_idx"]: r for r in baseline_results}
        seen = set()
        for row in alpha_sweep_rows:
            wid = row["winner_idx"]
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            w = by_idx[wid]
            ax.annotate(f"a={row['alpha']:g}\n#{wid}", (w["C_config"], w["Q"]),
                       textcoords="offset points", xytext=(6, 6), fontsize=8, color="black")
            ax.scatter([w["C_config"]], [w["Q"]], s=40, color="black", zorder=5)

    ax.set_xlabel("C_config (setup cost)")
    ax.set_ylabel("Q (makespan)")
    ax.set_title("Cost vs. makespan trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", choices=["phi-recall", "compare", "twopass-preview", "alpha-sweep", "pareto"])
    parser.add_argument("instance_file")
    parser.add_argument("baseline_run")
    parser.add_argument("heuristic_run", nargs="?", default=None,
                        help="required for 'compare'")
    parser.add_argument("--fractions", default="0.05,0.1,0.2,0.3,0.5,0.7,1.0",
                        help="[phi-recall] comma-separated fractions to test")
    parser.add_argument("--refine-corridor-factors", default="1.0,2.0,3.0,5.0",
                        help="[twopass-preview] comma-separated refine_corridor_factor values to test")
    parser.add_argument("--refine-time-limit", type=float, default=None,
                        help="[twopass-preview] Gurobi time limit you're planning for the "
                             "refinement pass, to project wall time")
    parser.add_argument("--alphas", default=None,
                        help="[alpha-sweep, pareto] comma-separated alpha values to test")
    parser.add_argument("--top-k", type=int, default=20,
                        help="[alpha-sweep] size of the reference top-K set for stability tracking")
    parser.add_argument("--reference-alpha", type=float, default=None,
                        help="[alpha-sweep] alpha defining the reference top-K set "
                             "(default: the instance's own recorded alpha)")
    parser.add_argument("--output-png", default=None,
                        help="[pareto] save the cost-vs-makespan plot here (requires matplotlib)")

    args = parser.parse_args()
    instance = cs.load_instance_json(args.instance_file)

    if args.analysis == "phi-recall":
        baseline_results = orch.load_scheduling_results(args.baseline_run, instance)
        fractions = [float(f) for f in args.fractions.split(",")]
        rows, context = phi_recall_curve(instance, baseline_results, fractions)
        print_phi_recall_table(rows, context)

    elif args.analysis == "compare":
        if not args.heuristic_run:
            parser.error("'compare' requires a heuristic_run argument")
        baseline_results = orch.load_scheduling_results(args.baseline_run, instance)
        heuristic_results = orch.load_scheduling_results(args.heuristic_run, instance)
        result = compare_baseline_vs_heuristic(instance, baseline_results, heuristic_results)
        print_compare_report(result)

    elif args.analysis == "twopass-preview":
        baseline_results = orch.load_scheduling_results(args.baseline_run, instance)
        factors = [float(f) for f in args.refine_corridor_factors.split(",")]
        pass1_elapsed = _baseline_elapsed_seconds(args.baseline_run)
        rows = twopass_preview(instance, baseline_results, factors,
                               refine_time_limit=args.refine_time_limit,
                               pass1_elapsed_s=pass1_elapsed)
        print_twopass_preview_table(rows, pass1_elapsed)

    elif args.analysis == "alpha-sweep":
        if not args.alphas:
            parser.error("'alpha-sweep' requires --alphas")
        baseline_results = orch.load_scheduling_results(args.baseline_run, instance)
        alphas = sorted(float(a) for a in args.alphas.split(","))
        rows, context = alpha_sweep(instance, baseline_results, alphas,
                                    top_k=args.top_k, reference_alpha=args.reference_alpha)
        intervals = find_winner_breakpoints(rows)
        print_alpha_sweep_table(rows, intervals, context)

    elif args.analysis == "pareto":
        baseline_results = orch.load_scheduling_results(args.baseline_run, instance)
        frontier = pareto_frontier(baseline_results)
        print_pareto_table(frontier, len(baseline_results))
        if args.output_png:
            alpha_rows = None
            if args.alphas:
                alphas = sorted(float(a) for a in args.alphas.split(","))
                alpha_rows, _ = alpha_sweep(instance, baseline_results, alphas)
            plot_pareto(baseline_results, frontier, alpha_rows, args.output_png)
            print(f"\nSaved plot to {args.output_png}")


if __name__ == "__main__":
    main()