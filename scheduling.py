"""
Lower-level scheduling oracle for the RMS bi-level optimization pipeline:
Gurobi hybrid-flowshop (HFS || Cmax) solver + the calling loop that runs
it over a set of upper-level configurations.

This module knows nothing about cost weighting (alpha), the admissible
corridor (epsilon), or the structural measures (Theta/Phi). It takes an
instance and a list of SystemConfig (the canonical config_idx of each
result is that list's position -- see config_space.feasible_configs and
structural.compute_measures, which use the same convention) and returns,
for every config in that list, the raw scheduling result: makespan Q(y),
setup cost C_config(y), throughput, and full schedule detail (activity +
gantt). evaluation.py is responsible for turning these into total_cost,
C*, and the corridor.

Because scheduling never restricts which configs get full detail
persisted (no "corridor only" shortcut), the same entry point serves
three run modes, distinguished only by which `configs` list is passed in:
  - full baseline run       : config_space.feasible_configs(instance)
  - two-pass refinement     : a second call on the subset near the
                              pass-1 provisional best, with a larger
                              time_limit_per_config
  - partial-inversion       : a first call on the subset that survived a
                              structural.compute_measures Phi threshold

Only representative-batch mode is supported (unit mode has been dropped
from the project).
"""

import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from config_space import (
    Job,
    RMSInstance,
    SystemConfig,
    build_pt_matrix,
    config_setup_cost,
    config_to_dict,
    config_label,
)


_GUROBI_STATUS_NAMES = {
    2:  "OPTIMAL",
    3:  "INFEASIBLE",
    4:  "INF_OR_UNBD",
    5:  "UNBOUNDED",
    9:  "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    13: "SUBOPTIMAL",
}


# ---------------------------------------------------------------------------
# Representative job batch
# ---------------------------------------------------------------------------

def build_scheduled_jobs(instance: RMSInstance) -> Tuple[List[Job], int]:
    """
    Build the representative job batch to schedule: instance.K copies of
    each active job, apportioned by demand share.

    Requires instance.K to be set (representative mode is the only mode
    supported -- unit mode was dropped from the project).
    """
    if instance.K is None:
        raise ValueError(
            "instance.K must be set (representative mode is the only "
            "supported mode; add \"K\": <int> to the instance JSON)."
        )

    active_jobs = [j for j in instance.jobs if j.demand > 0.0]
    if not active_jobs:
        raise ValueError(
            "Representative mode requires at least one job with demand > 0."
        )

    total_demand = sum(j.demand for j in active_jobs)
    scheduled_jobs: List[Job] = []
    for j in active_jobs:
        n_copies = math.ceil(instance.K * j.demand / total_demand)
        scheduled_jobs.extend([j] * n_copies)

    return scheduled_jobs, len(scheduled_jobs)


# ---------------------------------------------------------------------------
# Lower-level: Gurobi Hybrid Flowshop solver (HFS || Cmax)
#
# Classic disjunctive MIP formulation for flexible/hybrid flow shops --
# unchanged from the original: this is the part that didn't need to move,
# only the loop calling it did.
# ---------------------------------------------------------------------------

def solve_flowshop_gurobi(pt_matrix:  List[List[float]],
                          nb_jobs:    int,
                          nb_stages:  int,
                          n_machines: List[int],
                          time_limit: int = 5) -> Tuple[float, dict, list, dict]:
    """
    Solve the Hybrid Flowshop minimising makespan via Gurobi.

    Parameters
    ----------
    pt_matrix  : pt_matrix[s][j] = processing time of job j at stage s
    nb_jobs    : number of jobs
    nb_stages  : number of stages
    n_machines : n_machines[s] = number of identical parallel machines at stage s
    time_limit : Gurobi time limit in seconds

    Returns
    -------
    (makespan, activity, gantt, solve_info)
        makespan   : float
        activity   : activity[(j, s)] = {"machine": k, "start": t, "end": t}
        gantt      : gantt[s][k] = sorted list of {"job": j, "start": t, "end": t}
        solve_info : {"status": str, "gap": float|None, "runtime": float,
                      "obj_bound": float, "node_count": int}
    """
    real_jobs = [
        [j for j in range(nb_jobs) if pt_matrix[s][j] > 0]
        for s in range(nb_stages)
    ]
    nb_real = [len(real_jobs[s]) for s in range(nb_stages)]

    eff_machines = [
        max(1, min(n_machines[s], nb_real[s])) if nb_real[s] > 0 else 1
        for s in range(nb_stages)
    ]

    UB = sum(pt_matrix[s][j]
             for s in range(nb_stages)
             for j in range(nb_jobs)
             if pt_matrix[s][j] > 0) + 1.0
    BIGM = UB

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    with gp.Model(env=env) as model, env:
        model.Params.TimeLimit = time_limit
        model.Params.OutputFlag = 0

        C = {
            (j, s): model.addVar(lb=0.0, ub=UB, name=f"C_{j}_{s}")
            for j in range(nb_jobs) for s in range(nb_stages)
        }

        x = {}
        for s in range(nb_stages):
            for j in real_jobs[s]:
                for k in range(eff_machines[s]):
                    x[j, s, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{s}_{k}")

        z = {}
        for s in range(nb_stages):
            real = real_jobs[s]
            for a in range(len(real)):
                for b in range(a + 1, len(real)):
                    j1, j2 = real[a], real[b]
                    z[j1, j2, s] = model.addVar(vtype=GRB.BINARY, name=f"z_{j1}_{j2}_{s}")

        model.update()

        for j in range(nb_jobs):
            for s in range(nb_stages):
                dur = pt_matrix[s][j]
                if s == 0:
                    model.addConstr(C[j, s] >= dur, name=f"prec0_{j}_{s}")
                else:
                    model.addConstr(C[j, s] >= C[j, s - 1] + dur, name=f"prec_{j}_{s}")

        for s in range(nb_stages):
            for j in real_jobs[s]:
                model.addConstr(
                    gp.quicksum(x[j, s, k] for k in range(eff_machines[s])) == 1,
                    name=f"assign_{j}_{s}",
                )

        for s in range(nb_stages):
            real = real_jobs[s]
            for a in range(len(real)):
                for b in range(a + 1, len(real)):
                    j1, j2 = real[a], real[b]
                    p1, p2 = pt_matrix[s][j1], pt_matrix[s][j2]
                    zvar = z[j1, j2, s]
                    for k in range(eff_machines[s]):
                        same_machine_slack = BIGM * (2 - x[j1, s, k] - x[j2, s, k])
                        model.addConstr(
                            C[j1, s] - p1 >= C[j2, s] - BIGM * (1 - zvar) - same_machine_slack,
                            name=f"seq1_{j1}_{j2}_{s}_{k}",
                        )
                        model.addConstr(
                            C[j2, s] - p2 >= C[j1, s] - BIGM * zvar - same_machine_slack,
                            name=f"seq2_{j1}_{j2}_{s}_{k}",
                        )

        Cmax = model.addVar(lb=0.0, ub=UB, name="Cmax")
        for j in range(nb_jobs):
            model.addConstr(Cmax >= C[j, nb_stages - 1], name=f"cmax_{j}")
        model.setObjective(Cmax, GRB.MINIMIZE)

        model.optimize()

        status_name = _GUROBI_STATUS_NAMES.get(model.Status, f"STATUS_{model.Status}")

        if model.SolCount == 0:
            print(f"[WARN] Gurobi found no feasible solution "
                  f"(status={model.Status}) within time_limit={time_limit}s",
                  file=sys.stderr)
            makespan_val = UB
            gap = None
        else:
            makespan_val = Cmax.X
            try:
                gap = model.MIPGap * 100.0 if model.Status != GRB.OPTIMAL else 0.0
            except AttributeError:
                gap = None

        solve_info = {
            "status":     status_name,
            "gap":        gap,
            "runtime":    model.Runtime,
            "obj_bound":  model.ObjBound if model.SolCount > 0 else None,
            "node_count": int(model.NodeCount),
        }

        assignment = [[0] * nb_jobs for _ in range(nb_stages)]
        for s in range(nb_stages):
            for j in real_jobs[s]:
                for k in range(eff_machines[s]):
                    if x[j, s, k].X > 0.5:
                        assignment[s][j] = k
                        break

        activity = {}
        gantt = [[[] for _ in range(eff_machines[s])] for s in range(nb_stages)]
        for j in range(nb_jobs):
            for s in range(nb_stages):
                k = assignment[s][j]
                dur = pt_matrix[s][j]
                end = C[j, s].X if model.SolCount > 0 else 0.0
                start = end - dur
                activity[(j, s)] = {"machine": k, "start": start, "end": end}
                gantt[s][k].append({"job": j, "start": start, "end": end})

        for s in range(nb_stages):
            for k in range(eff_machines[s]):
                gantt[s][k].sort(key=lambda e: e["start"])

        return float(makespan_val), activity, gantt, solve_info


# ---------------------------------------------------------------------------
# Run output: text log + incremental, crash-safe JSON
# ---------------------------------------------------------------------------

class RunLogger:
    """Writes every line to stdout and, if configured, to a .log file."""

    def __init__(self, log_path: str = None):
        if log_path:
            parent = os.path.dirname(os.path.abspath(log_path))
            os.makedirs(parent, exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8") if log_path else None

    def write(self, line: str = ""):
        print(line)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


def activity_to_list(activity: dict, scheduled_jobs: List[Job]) -> list:
    """Flatten the {(j,s): {...}} activity dict into a JSON-friendly list."""
    return [
        {"job_id": scheduled_jobs[j].job_id, "job_index": j, "stage": s,
         "machine": a["machine"], "start": a["start"], "end": a["end"]}
        for (j, s), a in sorted(activity.items())
    ]


def gantt_to_list(gantt: list, scheduled_jobs: List[Job]) -> list:
    return [
        [
            [{"job_id": scheduled_jobs[e["job"]].job_id,
              "start": e["start"], "end": e["end"]} for e in machine]
            for machine in stage
        ]
        for stage in gantt
    ]


class JsonRunWriter:
    """
    Crash-safe, incremental writer for a scheduling run's output.

    Two files, not one:
      - <path_base>.configs.jsonl : one JSON object per line, one line
        per scheduled config. Appended and flushed after every config --
        O(1) per write, and every already-written line stays valid even
        if the process is killed mid-run.
      - <path_base>.meta.json     : small file with run metadata and,
        once the run finishes, the final summary. Rewritten in full on
        every set_summary() call, but this file never grows with the
        number of configs, so that stays cheap.

    This replaces an earlier version that rewrote one ever-growing JSON
    document (meta + all configs + summary) in full on every single
    config -- fine at ~525 corridor-only entries, but O(n^2) once every
    feasible config (thousands) gets full schedule detail persisted, as
    this pipeline now does by design. Measured directly: ~131s for 3024
    lightweight entries and climbing quadratically -- with real
    activity/gantt payloads on a ~4000-config run, that would dominate
    total run time on top of the actual Gurobi solving.
    """

    def __init__(self, path_base: str, meta: dict):
        self.path_base = path_base
        parent = os.path.dirname(os.path.abspath(path_base))
        os.makedirs(parent, exist_ok=True)
        self.configs_path = f"{path_base}.configs.jsonl"
        self.meta_path = f"{path_base}.meta.json"
        self._meta = meta
        self._summary = None
        self._configs_fh = open(self.configs_path, "w", encoding="utf-8")
        self._write_meta()

    def add_config_result(self, entry: dict):
        self._configs_fh.write(json.dumps(entry) + "\n")
        self._configs_fh.flush()

    def set_summary(self, summary: dict):
        """
        Attach a final summary dict (e.g. C_star/threshold/winner from
        evaluation.py) once the run is complete. Generic -- this module
        has no opinion on what the summary contains.
        """
        self._summary = summary
        self._write_meta()

    def _write_meta(self):
        tmp_path = self.meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": self._meta,
                "summary": self._summary,
                "configs_file": os.path.basename(self.configs_path),
            }, f, indent=2)
        os.replace(tmp_path, self.meta_path)

    def close(self):
        self._configs_fh.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def schedule_configs(
    instance: RMSInstance,
    configs: List[SystemConfig],
    config_ids: List[int] = None,
    time_limit_per_config: int = 5,
    logger: RunLogger = None,
    json_writer: JsonRunWriter = None,
    verbose: bool = True,
) -> Tuple[List[dict], dict]:
    """
    Run the Gurobi HFS oracle on every config in `configs`, in order.

    config_idx in each result is config_ids[i] (defaults to range(len(configs))
    if not given). Pass explicit config_ids whenever `configs` is not the
    full canonical universe -- e.g. for the partial-inversion heuristic,
    where `configs` is only the subset surviving a Phi filter computed by
    structural.compute_measures over the *full* universe: pass that
    subset's original global ids here so the two modules' config_idx
    values still refer to the same configs and can be joined downstream.

    Returns
    -------
    (results, run_info)
        results  : list of dicts, one per config, each with config_idx,
                   y, C_config, Q, perf, activity, gantt, solve_info.
                   No total_cost -- that's evaluation.py's job, since it
                   needs alpha, which this module doesn't know about.
        run_info : {"batch_size": int, "n_time_limit_hits": int,
                    "scheduled_jobs": List[Job]}
    """
    if config_ids is None:
        config_ids = list(range(len(configs)))
    elif len(config_ids) != len(configs):
        raise ValueError("config_ids must have the same length as configs.")

    scheduled_jobs, batch_size = build_scheduled_jobs(instance)

    results: List[dict] = []
    n_time_limit_hits = 0

    for pos, (idx, y) in enumerate(zip(config_ids, configs)):
        pt_matrix = build_pt_matrix(y, instance, scheduled_jobs)

        n_machines = [
            max(1, min(n, sum(1 for j in range(batch_size)
                              if pt_matrix[s][j] > 0)))
            for s, (_, n) in enumerate(y)
        ]

        makespan, activity, gantt, solve_info = solve_flowshop_gurobi(
            pt_matrix, batch_size, instance.nb_stages,
            n_machines=n_machines,
            time_limit=time_limit_per_config,
        )
        if solve_info["status"] != "OPTIMAL":
            n_time_limit_hits += 1

        cconfig = config_setup_cost(y)
        tp = batch_size / makespan if makespan > 0 else float("inf")

        entry = {
            "config_idx": idx,
            "y":          y,
            "C_config":   cconfig,
            "Q":          makespan,
            "perf":       tp,
            "activity":   activity,
            "gantt":      gantt,
            "solve_info": solve_info,
        }
        results.append(entry)

        if verbose and logger:
            flag = "" if solve_info["status"] == "OPTIMAL" else f"  [{solve_info['status']}]"
            logger.write(f"  [{pos+1:4d}/{len(configs)}] config_idx={idx}  {config_label(y)}"
                        f"  Cconfig={cconfig:.1f}  Cmax={makespan:.1f}"
                        f"  TP={tp:.4f}{flag}")

        if json_writer:
            json_writer.add_config_result({
                "config_idx":      idx,
                "y":               config_to_dict(y),
                "C_config":        cconfig,
                "Q":               makespan,
                "perf":            tp,
                "solver_status":   solve_info["status"],
                "mip_gap_pct":     solve_info["gap"],
                "solve_runtime_s": solve_info["runtime"],
                "node_count":      solve_info["node_count"],
                "activity":        activity_to_list(activity, scheduled_jobs),
                "gantt":           gantt_to_list(gantt, scheduled_jobs),
            })

    run_info = {
        "batch_size":        batch_size,
        "n_time_limit_hits": n_time_limit_hits,
        "scheduled_jobs":    scheduled_jobs,
    }

    return results, run_info