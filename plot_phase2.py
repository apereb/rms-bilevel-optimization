"""
plot_phase2.py
==============

Generates presentation-ready figures directly from a Phase-2 run, by
reusing the exact functions in config_space.py / structural.py /
evaluation.py / orchestrate.py (no re-implementation of the cost / Theta
/ Phi logic, so the plots are guaranteed consistent with the CLI tools'
own output).

Port of the pre-refactor plot_phase2.py (which imported from
legacy.explore_phase2 / legacy.rms_bilevel_gurobi) onto the current
4-module pipeline. Same three figures, same visual style. Bonus over the
original: works on ANY run file orchestrate.py can load -- full,
two-pass, partial-inversion, or a legacy single-file run -- via
orchestrate.load_scheduling_results' format auto-detection.

Theta/Phi are always computed over the FULL feasible universe
(config_space.feasible_configs), never over just the configs a given run
happened to schedule and persist -- Phi is neighbourhood-dependent, and
computing it from a partial-inversion run's own (truncated) file would
silently disagree with the value that run actually used internally to
pick its winner (a real bug caught and fixed elsewhere in this pipeline;
see compare_runs.py / heuristic_analysis.py / orchestrate.py's
run_recompute for the same fix). The scatter of "outside corridor" points
in fig1 is necessarily limited to whichever configs the run actually
scheduled (only Gurobi-solved configs have a Q/total_cost to plot) --
for a partial-inversion run this means fewer grey points than the true
universe, which is an inherent data limitation, not a bug.

Figures produced (saved as PNG, 300 dpi, in --outdir):

  1. fig1_cost_vs_phi.png
       Scatter of total_cost(y) vs Phi(y) for every config the run
       actually scheduled. Corridor configs highlighted, C* and
       (1+eps)C* threshold lines drawn, winner starred.

  2. fig2_top_corridor_theta.png
       Horizontal bar chart of the top-N corridor configs ranked by Phi,
       with Theta(y) split into its versatility and surplus-capacity
       components.

  3. fig3_neighbourhood_hist.png
       Histogram of |N(y, tau)| over all corridor configs.

Usage
-----
    python plot_phase2.py runs/r1/period1_baseline.meta.json instance_period1.json \
        --top 20 --outdir ./figures

    # override alpha/epsilon/tau/lambda/mu/delta for a what-if plot,
    # exactly like orchestrate.py recompute -- no Gurobi call either way:
    python plot_phase2.py runs/r1/period1_baseline.meta.json instance_period1.json \
        --epsilon 0.25 --outdir ./figures_eps025
"""

import argparse
import dataclasses
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config_space as cs
import evaluation as ev
import orchestrate as orch
import structural as st

# ---------------------------------------------------------------------
# Simple, presentation-friendly style (no seaborn dependency needed)
# ---------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

CORRIDOR_COLOR = "#2E7D32"      # green
OUTSIDE_COLOR = "#B0B0B0"       # grey
WINNER_COLOR = "#D81B60"        # magenta/red
LINE_COLOR = "#444444"


def fig_cost_vs_phi(results, total_cost, phi, corridor_ids, winner, C_star,
                     threshold, outpath):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    corridor_set = set(corridor_ids)

    xs_out = [total_cost[r["config_idx"]] for r in results if r["config_idx"] not in corridor_set]
    ys_out = [phi[r["config_idx"]] for r in results if r["config_idx"] not in corridor_set]

    xs_in = [total_cost[r["config_idx"]] for r in results if r["config_idx"] in corridor_set]
    ys_in = [phi[r["config_idx"]] for r in results if r["config_idx"] in corridor_set]

    ax.scatter(xs_out, ys_out, s=14, color=OUTSIDE_COLOR, alpha=0.5,
               label="Outside corridor", zorder=2)
    ax.scatter(xs_in, ys_in, s=22, color=CORRIDOR_COLOR, alpha=0.85,
               label=r"Admissible corridor $\mathcal{Y}_\epsilon$", zorder=3)

    ax.axvline(C_star, color=LINE_COLOR, linestyle="--", linewidth=1.2,
               label=r"$C^*$")
    ax.axvline(threshold, color=LINE_COLOR, linestyle=":", linewidth=1.2,
               label=r"$(1+\epsilon)C^*$")

    if winner is not None:
        wx = winner["total_cost"]
        wy = winner["Phi"]
        ax.scatter([wx], [wy], marker="*", s=400, color=WINNER_COLOR,
                   edgecolor="black", linewidth=0.6, zorder=5,
                   label=f"Selected config #{winner['config_idx']}")
        ax.annotate(
            f"  #{winner['config_idx']}\n  " + r"$\Phi$=" + f"{wy:.3f}",
            (wx, wy), textcoords="offset points", xytext=(10, 8),
            fontsize=9, color=WINNER_COLOR, fontweight="bold",
        )

    ax.set_xlabel("Total cost  " + r"$F(y)$")
    ax.set_ylabel(r"Flexibility  $\Phi(y)$")
    ax.set_title("Phase 2: cost vs. flexibility trade-off")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def fig_top_corridor_theta(corridor_ranked, n_operations, stage_ops, stage_demands,
                           top_n, outpath):
    top = corridor_ranked[:top_n]
    labels = [f"#{r['config_idx']}" for r in top]

    versatility = [st.machine_versatility_sum(r["y"], n_operations) for r in top]
    surplus = [st.surplus_capacity(r["y"], stage_ops, stage_demands) for r in top]

    y_pos = range(len(top))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(y_pos, versatility, color="#1E88E5", label="Versatility term")
    ax.barh(y_pos, surplus, left=versatility, color="#FFB300",
            label="Surplus capacity term")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # highest Phi at top
    ax.set_xlabel(r"$\Theta(y)$ = versatility + surplus capacity")
    ax.set_title(f"Top {len(top)} corridor configs by " + r"$\Phi(y)$" +
                 " — capability breakdown")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def fig_neighbourhood_hist(corridor_ids, neighbour_sets, outpath):
    sizes = [len(neighbour_sets[i]) for i in corridor_ids]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(sizes, bins=min(20, max(5, len(set(sizes)))),
            color=CORRIDOR_COLOR, alpha=0.85, edgecolor="white")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel(r"$|N(y,\tau)|$")
    ax.set_ylabel("Number of corridor configs")
    ax.set_title(r"Neighbourhood size distribution ($\tau$ fixed)")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_json")
    parser.add_argument("instance_file")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--lambda-coef", type=float, default=None, dest="lambda_coef")
    parser.add_argument("--mu-coef", type=float, default=None, dest="mu_coef")
    parser.add_argument("--delta-coef", type=float, default=None, dest="delta_coef")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--outdir", default="./figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    instance = cs.load_instance_json(args.instance_file)

    overrides = {}
    for field_name, cli_val in [
        ("alpha", args.alpha), ("epsilon", args.epsilon), ("tau", args.tau),
        ("lambda_coef", args.lambda_coef), ("mu_coef", args.mu_coef),
        ("delta_coef", args.delta_coef),
    ]:
        if cli_val is not None:
            overrides[field_name] = cli_val
    if overrides:
        instance = dataclasses.replace(instance, **overrides)

    results = orch.load_scheduling_results(args.run_json, instance)
    if not results:
        raise RuntimeError("The run file contains no feasible configurations.")

    total_cost = ev.compute_total_cost(results, instance.alpha)
    C_star, threshold, corridor_entries = ev.compute_corridor(results, total_cost, instance.epsilon)
    corridor_ids = [r["config_idx"] for r in corridor_entries]

    # Full-universe Theta/Phi -- never from `results`' own (possibly
    # truncated) config list. See module docstring.
    universe_configs = cs.feasible_configs(instance)
    universe_ids = list(range(len(universe_configs)))
    theta, phi, neighbour_sets = st.compute_measures(instance, universe_configs, universe_ids)

    winner_entry = ev.select_phase2_winner(corridor_entries, phi)
    winner = None
    if winner_entry is not None:
        winner = {
            **winner_entry,
            "total_cost": total_cost[winner_entry["config_idx"]],
            "Phi": phi[winner_entry["config_idx"]],
        }

    corridor_ranked = sorted(corridor_entries, key=lambda r: phi[r["config_idx"]], reverse=True)

    stage_ops = st.compute_stage_operations(instance)
    stage_demands = st.compute_stage_demands(instance, stage_ops)
    n_operations = instance.n_operations

    fig_cost_vs_phi(
        results, total_cost, phi, corridor_ids, winner, C_star, threshold,
        os.path.join(args.outdir, "fig1_cost_vs_phi.png"),
    )

    fig_top_corridor_theta(
        corridor_ranked, n_operations, stage_ops, stage_demands, args.top,
        os.path.join(args.outdir, "fig2_top_corridor_theta.png"),
    )

    fig_neighbourhood_hist(
        corridor_ids, neighbour_sets,
        os.path.join(args.outdir, "fig3_neighbourhood_hist.png"),
    )

    print(f"[plot_phase2] {len(results)} configs loaded, "
          f"{len(corridor_ids)} in corridor, winner = "
          f"{'#' + str(winner['config_idx']) if winner else 'none'}")
    print(f"[plot_phase2] Figures written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()