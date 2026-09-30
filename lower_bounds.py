"""
Lower bounds on the Phase-1 quantities of the RMS bi-level pipeline.

Single responsibility: cheap, solver-free lower bounds on the makespan Q(y)
and on total_cost(y) = C_config(y) + alpha * Q(y) / batch_size. Used to
(1) certify C* with few scheduling calls and (2) restrict the set of
configurations that can belong to the admissible corridor (orchestrate.py,
mode `lb-exact`).

Depends only on config_space.py. Knows nothing about Gurobi, Theta/Phi or
the corridor.

Bound: job bound + Santos-Hunsucker-Deal (SHD) center-based stage bound.
Only jobs with a strictly positive processing time at a stage are counted
there (a 0-time no-op neither starts nor occupies a machine).
"""

from typing import Dict, List, Sequence

import config_space as cs


def makespan_lb(pt_matrix: Sequence[Sequence[float]],
                n_machines: Sequence[int]) -> float:
    """
    Lower bound on the makespan of a hybrid flow shop, processing times only.

    pt_matrix[s][j] : processing time of job j at stage s (0 = no-op).
    n_machines[s]   : machines actually available at stage s.

    Bounds (both valid for any schedule):
      - job bound   : max_j sum_s p[s][j]
      - stage bound : Santos-Hunsucker-Deal center bound,
        (sum of m smallest heads + load_s + sum of m smallest tails) / m,
        m = min(n_s, #jobs active at stage s); only jobs with p > 0 count.
        head_j(s) = sum of p at earlier stages,
        tail_j(s) = sum of p at later stages.
    """
    S = len(pt_matrix)
    J = len(pt_matrix[0]) if S else 0
    lb = 0.0
    for j in range(J):
        lb = max(lb, sum(pt_matrix[s][j] for s in range(S)))
    for s in range(S):
        idx = [j for j in range(J) if pt_matrix[s][j] > 0]
        if not idx:
            continue
        heads = sorted(sum(pt_matrix[t][j] for t in range(s)) for j in idx)
        tails = sorted(sum(pt_matrix[t][j] for t in range(s + 1, S)) for j in idx)
        load = sum(pt_matrix[s][j] for j in idx)
        m = min(n_machines[s], len(idx))
        stage_lb = (sum(heads[:m]) + load + sum(tails[:m])) / m
        lb = max(lb, stage_lb)
    return lb


def effective_machines(y, pt_matrix: Sequence[Sequence[float]]) -> List[int]:
    """Machines the MIP really uses (same rule as scheduling.schedule_configs)."""
    return [
        max(1, min(n, sum(1 for v in pt_matrix[s] if v > 0)))
        for s, (_, n) in enumerate(y)
    ]


def cost_lower_bounds(instance, configs: list, scheduled_jobs: list,
                      batch_size: int) -> Dict[int, float]:
    """
    LB_cost(y) = C_config(y) + alpha * LB_Q(y) / batch_size  <=  total_cost(y).

    Keys are config_idx = position in `configs` (the canonical universe from
    config_space.feasible_configs). `scheduled_jobs` and `batch_size` come
    from scheduling.build_scheduled_jobs(instance), so the bound sees exactly
    the batch the MIP schedules.
    """
    out: Dict[int, float] = {}
    for idx, y in enumerate(configs):
        pt = cs.build_pt_matrix(y, instance, scheduled_jobs)
        lb_q = makespan_lb(pt, effective_machines(y, pt))
        out[idx] = cs.config_setup_cost(y) + instance.alpha * lb_q / batch_size
    return out