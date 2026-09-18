"""
Cost evaluation and final selection for the RMS bi-level optimization
pipeline: total_cost(y), C*, the (1+epsilon) admissible corridor, and the
Phase-2 winner (max Phi within the corridor).

This module owns alpha and epsilon (and, transitively, everything that
depends on them) -- neither scheduling.py nor structural.py know about
these. It joins the two by config_idx:
  - scheduling results supply C_config(y), Q(y)/perf(y)
  - structural results supply Theta(y), Phi(y)

Both must use the same global config_idx convention for a given call
(see the config_ids parameters on scheduling.schedule_configs and
structural.compute_measures) -- this module does not re-derive or
validate that alignment, it trusts the ids it's given.
"""

from typing import Dict, List, Optional

from config_space import RMSInstance


def compute_total_cost(
    scheduling_results: List[dict],
    alpha: float,
) -> Dict[int, float]:
    """
    total_cost(y) = C_config(y) + alpha / perf(y)

    perf(y) is the representative throughput (batch_size / makespan),
    already computed by scheduling.schedule_configs. Configs with
    perf <= 0 (a degenerate/failed solve) get total_cost = +inf so they
    never win the corridor.
    """
    total_cost: Dict[int, float] = {}
    for r in scheduling_results:
        cid = r["config_idx"]
        tp = r["perf"]
        if tp > 0:
            total_cost[cid] = r["C_config"] + alpha / tp
        else:
            total_cost[cid] = float("inf")
    return total_cost


def compute_corridor(
    scheduling_results: List[dict],
    total_cost: Dict[int, float],
    epsilon: float,
) -> tuple:
    """
    C* = min total_cost over scheduling_results.
    threshold = (1 + epsilon) * C*.
    corridor = the scheduling_results entries whose total_cost <= threshold.

    Returns (C_star, threshold, corridor_entries).
    """
    if not scheduling_results:
        raise ValueError("compute_corridor: scheduling_results is empty.")

    C_star = min(total_cost[r["config_idx"]] for r in scheduling_results)
    threshold = (1.0 + epsilon) * C_star

    corridor_entries = [
        r for r in scheduling_results
        if total_cost[r["config_idx"]] <= threshold
    ]

    return C_star, threshold, corridor_entries


def select_phase2_winner(
    corridor_entries: List[dict],
    phi: Dict[int, float],
) -> Optional[dict]:
    """
    Phase-2 selection: among the corridor entries, the one maximizing
    Phi(y). Returns None if the corridor is empty.

    Raises KeyError if a corridor entry's config_idx has no matching Phi
    value -- almost always a sign that scheduling and structural were run
    over misaligned config_ids (see module docstring).
    """
    if not corridor_entries:
        return None

    return max(corridor_entries, key=lambda r: phi[r["config_idx"]])


def evaluate(
    instance: RMSInstance,
    scheduling_results: List[dict],
    theta: Dict[int, float],
    phi: Dict[int, float],
) -> dict:
    """
    Full evaluation pass: cost, corridor, and Phase-2 winner, using
    instance.alpha and instance.epsilon.

    Returns a dict with:
        total_cost        : Dict[config_idx, float]
        C_star            : float
        threshold         : float
        corridor           : List[dict]  (scheduling_results entries)
        winner            : dict | None  (a corridor entry, or None if
                            the corridor is empty)
    Each returned corridor entry keeps its scheduling fields (C_config,
    Q, perf, activity, gantt, solve_info) and gains "Theta" and "Phi" for
    convenience, without mutating the caller's original dicts.
    """
    total_cost = compute_total_cost(scheduling_results, instance.alpha)
    C_star, threshold, corridor_entries = compute_corridor(
        scheduling_results, total_cost, instance.epsilon
    )

    corridor = [
        {**r,
         "total_cost": total_cost[r["config_idx"]],
         "Theta": theta[r["config_idx"]],
         "Phi": phi[r["config_idx"]]}
        for r in corridor_entries
    ]

    winner = max(corridor, key=lambda r: r["Phi"]) if corridor else None

    return {
        "total_cost": total_cost,
        "C_star":     C_star,
        "threshold":  threshold,
        "corridor":   corridor,
        "winner":     winner,
    }


def refinement_subset(
    scheduling_results: List[dict],
    total_cost: Dict[int, float],
    refine_corridor_factor: float,
    epsilon: float,
) -> List[dict]:
    """
    For two-pass refinement: the entries within a *wider* corridor than
    the final one, i.e. total_cost <= (1 + refine_corridor_factor*epsilon)
    * provisional_best, meant to be re-scheduled with a larger time limit
    before the final evaluate() call.

    provisional_best is recomputed here from the pass-1 results passed
    in, so this can be called directly on schedule_configs' pass-1
    output.
    """
    provisional_best = min(total_cost[r["config_idx"]] for r in scheduling_results)
    refine_threshold = (1.0 + refine_corridor_factor * epsilon) * provisional_best

    return [
        r for r in scheduling_results
        if total_cost[r["config_idx"]] <= refine_threshold
    ]