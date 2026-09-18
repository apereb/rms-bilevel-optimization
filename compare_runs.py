"""
Side-by-side comparison of several pipeline runs (full baseline, two-pass,
partial-inversion, or old legacy runs) against the same instance.

Unlike just reading each run's own stored summary, this recomputes
Theta/Phi/cost/corridor for every run through structural.py/evaluation.py
-- the same code, every time. That matters for comparing an old legacy
run (unnormalized Theta) against new ones (normalized Theta): reading
stored summaries would compare apples to oranges; recomputing puts every
run on the same footing.

By default each run is evaluated under its *own* recorded alpha/epsilon/
tau (from its meta), so C*/corridor reflect what that run actually
targeted. Pass --alpha/--epsilon/--tau to force every run to be
re-evaluated under the same hyperparameters instead (e.g. to ask "which
heuristic gets closest to the true C*, under identical settings?").

Usage
-----
    python compare_runs.py instance.json baseline.meta.json two-pass.meta.json \
        partial-inversion.meta.json --output comparison.md

    # custom labels:
    python compare_runs.py instance.json baseline=runs/full.meta.json \
        heuristic=runs/pi.meta.json

    # old legacy run.json (single-file format) works too:
    python compare_runs.py instance.json legacy=old_run1.json

    # force a common epsilon across all runs for a fair comparison:
    python compare_runs.py instance.json a.meta.json b.meta.json --epsilon 0.15
"""

import argparse
import dataclasses
import os
import time
from typing import Dict, List, Optional

import config_space as cs
import evaluation as ev
import orchestrate as orch
import structural as st


def parse_run_arg(arg: str) -> tuple:
    """'label=path' -> (label, path); bare 'path' -> (basename w/o ext, path)."""
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label, path
    label = os.path.splitext(os.path.basename(arg))[0]
    return label, arg


def load_run_meta(run_path: str) -> dict:
    """
    Read a run's own meta dict, regardless of format: new JSONL-sidecar
    files have it at data["meta"]; legacy single-file runs have it there
    too (same key, both formats agree on this).
    """
    import json
    with open(run_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("meta", {})


def elapsed_seconds(meta: dict, summary: Optional[dict]) -> Optional[float]:
    """Best-effort wall time from meta['started_at'] to summary['finished_at']."""
    if not summary or "started_at" not in meta or "finished_at" not in summary:
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t0 = time.mktime(time.strptime(meta["started_at"], fmt))
        t1 = time.mktime(time.strptime(summary["finished_at"], fmt))
        return t1 - t0
    except ValueError:
        return None


def load_run_summary_if_present(run_path: str) -> Optional[dict]:
    import json
    with open(run_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("summary")


def evaluate_run(instance_base: cs.RMSInstance, run_path: str,
                 alpha: float, epsilon: float, tau: float,
                 universe_cache: Dict[float, tuple]) -> dict:
    """
    Load one run's scheduling results and recompute cost/corridor under
    the given (alpha, epsilon, tau) -- shared evaluation.py, so every run
    is judged by the same yardstick.

    Theta/Phi are recomputed over the FULL feasible universe
    (config_space.feasible_configs), never over just the subset of
    configs this particular run happened to schedule and persist. Phi is
    neighbourhood-dependent: a partial-inversion run's file only contains
    ~30% of the universe, and computing Phi from that subset alone gives
    a biased value for the exact same config_idx compared to computing
    it over the true full universe -- which is what orchestrate.py did
    internally when the run first picked its winner. Recomputing from a
    truncated file would silently disagree with the run's own original
    result. Theta has no such issue (it's config-local, no cross-config
    dependency), but Phi is recomputed alongside it here for simplicity
    and because both come from the same compute_measures() call.

    `universe_cache` memoizes the full-universe (theta, phi) by tau value
    across calls in the same process, since it only depends on tau (not
    alpha/epsilon) and is the same ~O(n^2) computation for every run
    sharing that tau -- avoids redoing it once per compared run.
    """
    run_meta = load_run_meta(run_path)
    run_summary = load_run_summary_if_present(run_path)

    instance = dataclasses.replace(instance_base, alpha=alpha, epsilon=epsilon, tau=tau)

    if tau not in universe_cache:
        universe_configs = cs.feasible_configs(instance)
        universe_ids = list(range(len(universe_configs)))
        universe_cache[tau] = st.compute_measures(instance, universe_configs, universe_ids)[:2]
    theta, phi = universe_cache[tau]

    results = orch.load_scheduling_results(run_path, instance)
    out = ev.evaluate(instance, results, theta, phi)

    out["mode"] = run_meta.get("mode", "?")
    out["n_scheduled"] = len(results)
    out["n_universe_hint"] = run_meta.get("limit")  # None for full/two-pass/partial-inversion universe runs
    out["elapsed_s"] = elapsed_seconds(run_meta, run_summary)
    out["run_meta"] = run_meta

    return out


def format_row(label: str, out: dict) -> List[str]:
    winner = out["winner"]
    elapsed = out["elapsed_s"]
    elapsed_str = f"{elapsed/3600:.1f}h" if elapsed is not None else "?"
    return [
        label,
        out["mode"],
        str(out["n_scheduled"]),
        f"{out['C_star']:.4f}",
        str(len(out["corridor"])),
        str(winner["config_idx"]) if winner else "-",
        f"{winner['total_cost']:.4f}" if winner else "-",
        f"{winner['Phi']:.4f}" if winner else "-",
        elapsed_str,
    ]


def print_markdown_table(rows: List[List[str]], headers: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instance_file")
    parser.add_argument("runs", nargs="+",
                        help="run files, as 'path' or 'label=path' "
                             "(new *.meta.json or legacy single-file run.json)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="force this alpha for every run (default: each "
                             "run's own recorded alpha)")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--output", default=None, help="save the table to this .md file")

    args = parser.parse_args()

    instance_base = cs.load_instance_json(args.instance_file)

    headers = ["run", "mode", "n_scheduled", "C*", "corridor", "winner_idx",
               "winner_cost", "winner_Phi", "wall_time"]
    rows = []
    universe_cache: Dict[float, tuple] = {}

    for run_arg in args.runs:
        label, path = parse_run_arg(run_arg)
        run_meta = load_run_meta(path)

        alpha = args.alpha if args.alpha is not None else run_meta.get("alpha", instance_base.alpha)
        epsilon = args.epsilon if args.epsilon is not None else run_meta.get("epsilon", instance_base.epsilon)
        tau = args.tau if args.tau is not None else run_meta.get("tau", instance_base.tau)

        out = evaluate_run(instance_base, path, alpha, epsilon, tau, universe_cache)
        rows.append(format_row(label, out))

    table = print_markdown_table(rows, headers)
    print(table)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()