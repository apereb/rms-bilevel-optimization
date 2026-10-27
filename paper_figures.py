"""
Figures for the paper (Resolution method / Results), built from `full` baseline runs.

Step 1 -- collect per-configuration data (needs the project modules, no Gurobi call):

    python paper_figures.py collect instances/instance_period2.json \
        runs/r2/baseline/period2_baseline_v2.meta.json --out figures/inst2_data.json

Step 2 -- plots (need only the JSON files and matplotlib):

    # Fig. "corridor shift": cost vs Phi, baseline threshold vs shifted threshold at fraction f
    python paper_figures.py shift figures/inst2_data.json --fraction 0.3 --out figures/fig_shift.pdf

    # Fig. "bound tightness": F vs its lower bound, one panel per instance
    python paper_figures.py bound figures/inst1_data.json figures/inst2_data.json \
        figures/inst3_data.json --out figures/fig_bound.pdf

The collected JSON has one entry per configuration of the universe, indexed by
config_idx (null where the baseline has no value): F (= total_cost), LBF (lower
bound on F) and Phi.
"""

import argparse
import json
import os
from typing import List


# ---------------------------------------------------------------------------
# Step 1: collect (project modules, lazy imports)
# ---------------------------------------------------------------------------

def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _stub_gurobipy_if_missing() -> None:
    """
    Nothing here calls the solver, but scheduling.py imports gurobipy at module
    level. If gurobipy is not installed (a machine without Gurobi), register a
    minimal stand-in so that the import succeeds. Any attempt to actually solve
    would fail, which is fine: these scripts only read baseline results.
    """
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        import sys
        import types
        stub = types.ModuleType("gurobipy")
        stub.GRB = types.SimpleNamespace()
        stub.Env = stub.Model = None
        sys.modules["gurobipy"] = stub


def collect(instance_file: str, baseline_run: str, out_json: str) -> None:
    _stub_gurobipy_if_missing()
    import config_space as cs
    import evaluation as ev
    import lower_bounds as lbd
    import orchestrate as orch
    import scheduling as sch
    import structural as st

    instance = cs.load_instance_json(instance_file)
    results = orch.load_scheduling_results(baseline_run, instance)
    universe = cs.feasible_configs(instance)
    ids = list(range(len(universe)))

    total_cost = ev.compute_total_cost(results, instance.alpha)
    C_star, threshold, corridor = ev.compute_corridor(results, total_cost, instance.epsilon)
    _, phi, _ = st.compute_measures(instance, universe, ids)
    winner = ev.select_phase2_winner(corridor, phi)

    jobs, batch_size = sch.build_scheduled_jobs(instance)
    lbf = lbd.cost_lower_bounds(instance, universe, jobs, batch_size)

    data = {
        "instance_file": instance_file,
        "alpha": instance.alpha,
        "epsilon": instance.epsilon,
        "C_star": C_star,
        "winner_idx": winner["config_idx"] if winner else None,
        "F":   [total_cost.get(i) for i in ids],
        "LBF": [lbf[i] for i in ids],
        "Phi": [phi[i] for i in ids],
    }
    _ensure_parent_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"saved {len(ids)} configurations to {out_json}")


# ---------------------------------------------------------------------------
# Step 2: plots (no project imports)
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_shift(data: dict, fraction: float, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F, Phi = data["F"], data["Phi"]
    eps, C_star, w = data["epsilon"], data["C_star"], data["winner_idx"]
    ids = [i for i in range(len(F)) if F[i] is not None]

    order = sorted(ids, key=lambda i: (-Phi[i], i))
    keep_n = max(1, round(len(order) * fraction))
    kept = set(order[:keep_n])
    phi_cut = Phi[order[keep_n - 1]]
    C_star_f = min(F[i] for i in kept)
    thr, thr_f = (1 + eps) * C_star, (1 + eps) * C_star_f
    corridor = {i for i in ids if F[i] <= thr}

    removed = [i for i in ids if i not in kept]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # F on the horizontal axis, Phi on the vertical axis (same convention as the baseline figures)
    ax.scatter([F[i] for i in removed], [Phi[i] for i in removed], s=5, c="0.75",
               label=f"removed by the $\\Phi$ cut ($f={fraction*100:.0f}\\%$)", rasterized=True)
    ax.scatter([F[i] for i in kept], [Phi[i] for i in kept], s=5, c="tab:blue",
               label="kept", rasterized=True)
    cor = sorted(corridor)
    ax.scatter([F[i] for i in cor], [Phi[i] for i in cor], s=22, facecolors="none",
               edgecolors="tab:orange", linewidths=0.7, label="baseline corridor")
    if w is not None:
        ax.scatter([F[w]], [Phi[w]], marker="*", s=140, c="crimson", zorder=5,
                   label="baseline winner $y^\\star$")
    ax.axvline(thr, color="tab:orange", lw=1.2, label="$(1+\\varepsilon)C^*$")
    ax.axvline(thr_f, color="tab:green", lw=1.2, ls="--", label="$(1+\\varepsilon)C^*_f$ (shifted)")
    ax.axhline(phi_cut, color="0.3", lw=0.8, ls=":")
    ax.set_xlim(C_star * 0.98, max(thr, thr_f) * 1.2)
    ax.set_xlabel("$F(y)$")
    ax.set_ylabel("$\\Phi(y)$")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}  (C*={C_star:.4f}, C*_f={C_star_f:.4f}, "
          f"kept={keep_n}/{len(ids)})")


def plot_bound(datas: List[dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(datas)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    for k, (ax, d) in enumerate(zip(axes[0], datas), start=1):
        F, LB = d["F"], d["LBF"]
        eps, C_star = d["epsilon"], d["C_star"]
        ids = [i for i in range(len(F)) if F[i] is not None]
        thr = (1 + eps) * C_star
        cor = [i for i in ids if F[i] <= thr]
        rest = [i for i in ids if F[i] > thr]
        ax.scatter([LB[i] for i in rest], [F[i] for i in rest], s=4, c="0.7", rasterized=True)
        ax.scatter([LB[i] for i in cor], [F[i] for i in cor], s=4, c="tab:orange", rasterized=True)
        lo = min(min(LB[i] for i in ids), min(F[i] for i in ids))
        hi = max(max(LB[i] for i in ids), max(F[i] for i in ids))
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8)
        ratios = sorted(LB[i] / F[i] for i in ids if F[i] > 0)
        ax.set_title(f"Instance {k}  (median bound/$F$ = {ratios[len(ratios)//2]:.3f})",
                     fontsize=8)
        ax.set_xlabel("lower bound on $F(y)$")
        if k == 1:
            ax.set_ylabel("$F(y)$")
    fig.tight_layout()
    _ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect")
    p.add_argument("instance_file")
    p.add_argument("baseline_run")
    p.add_argument("--out", required=True)

    p = sub.add_parser("shift")
    p.add_argument("data_json")
    p.add_argument("--fraction", type=float, default=0.3)
    p.add_argument("--out", required=True)

    p = sub.add_parser("bound")
    p.add_argument("data_json", nargs="+")
    p.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "collect":
        collect(args.instance_file, args.baseline_run, args.out)
    elif args.cmd == "shift":
        plot_shift(_load(args.data_json), args.fraction, args.out)
    elif args.cmd == "bound":
        plot_bound([_load(p) for p in args.data_json], args.out)


if __name__ == "__main__":
    main()