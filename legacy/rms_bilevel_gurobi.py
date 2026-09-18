"""
Exhaustive upper-level enumeration + Gurobi lower-level hybrid flowshop solver.

This is a Gurobi port of rms_bilevel.py, built to cross-check the Hexaly
lower-level solver against an independent MIP formulation. The upper-level
logic (enumeration, capability/capacity pre-filtering, two-pass refinement,
corridor + flexibility computation) is unchanged: only solve_flowshop_*
is replaced.

Data model
----------
Processing times are stored as p[op_id] on each MachineConfig (so the general
form p_{o,(i,k)}).
In the flowshop, each stage s imposes a unique operation on
each job j (job.operations[s] gives the op_id), so we recover the classic
p_{j,s}(y) by looking up mc.processing_times[op_id].

Each stage can now have n_s >= 1 identical parallel machines (Hybrid Flowshop).
n_s is a decision variable bounded by N_max (global), enumerated at the upper
level alongside the config choice. Setup cost scales as n_s * mc.setup_cost.

Lower-level formulation (Gurobi)
---------------------------------
Hexaly's version uses interval variables + list (partition) variables, which
have no direct Gurobi equivalent (Gurobi is a MIP solver, not a CP solver).
The Gurobi port uses the classic disjunctive MIP formulation for hybrid
flowshops instead:
  - continuous start-time variables C[j][s] (we track completion times)
  - precedence: C[j][s] >= C[j][s-1] + p[j][s]  (stage ordering per job)
  - binary assignment x[j][s][k]: job j is processed by machine k at stage s
  - assignment: sum_k x[j][s][k] = 1 for every real job j at every stage s
  - binary sequencing y[j][j'][s] (j != j', same stage): 1 iff j precedes j'
    on whichever machine they share; only created for job pairs that could
    plausibly share a machine (both assignable, i.e. always, since machines
    are interchangeable within a stage)
  - disjunctive ("big-M") constraints linking x, y and C so that two jobs
    assigned to the same machine at the same stage don't overlap
  - objective: minimise Cmax = max_j C[j][nb_stages-1]

This is the standard textbook MIP encoding for hybrid/parallel-machine
flowshops (see e.g. Pinedo, "Scheduling", ch. on flexible flow shops). It was
chosen over an "optional interval" style encoding because Gurobi has no
native interval/no-overlap global constraint the way a CP solver does -- the
big-M disjunctive formulation is the direct MIP analogue and is well-suited
to the problem size here (instances are small: ~20 jobs, few stages, 1-3
machine configs per stage), where MIP solvers handle the resulting binary
sequencing variables comfortably within short time limits.

Usage:
    python rms_bilevel_gurobi.py instance.json [time_limit_per_config]
"""

import gurobipy as gp
from gurobipy import GRB
import itertools
import sys
import os
import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Data Classes
# (identical to rms_bilevel.py -- kept duplicated so this file has no
# import-time dependency on the Hexaly version)

@dataclass
class Job:
    job_id: int
    operations: List[int]   # ordered list of op_ids this job requires
    demand: float = 0.0     # demand rate in parts/minute (0 = unconstrained)


@dataclass
class MachineConfig:
    machine_id: int
    config_id:  int
    setup_cost: float
    processing_times: Dict[int, float]          # op_id -> p_{o,(i,k)}
    modules: List[int] = field(default_factory=list)

    @property
    def capable_ops(self) -> Set[int]:
        return set(self.processing_times.keys())


@dataclass
class RMSInstance:
    jobs:         List[Job]
    nb_stages:    int
    stage_configs: List[List[MachineConfig]]    # stage_configs[s] = options at stage s
    N_max:        int   = 1                     # global upper bound on machines per stage
    K:            int   = None                  # representative batch size (None = unit mode)

    # Reconfiguration time coefficients
    lambda_coef: float = 3.0
    mu_coef:     float = 2.0
    delta_coef:  float = 1.0

    # Bi-level parameters
    alpha:   float = 1.0    # weight on Q(y) in objective
    epsilon: float = 0.05   # admissible degradation threshold
    tau:     float = 10.0   # max reconfiguration time for neighbourhood
    psi:     float = 1.0    # power index for operational capability
    time_scale: int = 1000  # kept for instance-file compatibility with the
                             # Hexaly version (Gurobi works in continuous
                             # time and does not require integer scaling),
                             # so this field is parsed but unused below.

    @property
    def nb_jobs(self) -> int:
        return len(self.jobs)


def load_instance_json(path: str) -> RMSInstance:
    """
    Load an RMS instance from a JSON file. Schema identical to the Hexaly
    version -- see rms_bilevel.py for the full JSON schema documentation.
    """
    with open(path) as f:
        d = json.load(f)

    jobs = [
        Job(job_id=j["job_id"], operations=j["operations"],
            demand=float(j.get("demand", 0.0)))
        for j in d["jobs"]
    ]

    stage_configs = []
    for s_list in d["stages"]:
        configs = []
        for c in s_list:
            pt = {int(op_id): float(t)
                  for op_id, t in c["processing_times"].items()}
            configs.append(MachineConfig(
                machine_id=c["machine_id"],
                config_id =c["config_id"],
                setup_cost=float(c["setup_cost"]),
                processing_times=pt,
                modules=c.get("modules", []),
            ))
        stage_configs.append(configs)

    return RMSInstance(
        jobs=jobs,
        nb_stages=d["nb_stages"],
        stage_configs=stage_configs,
        N_max=      d.get("N_max",       1),
        K=          d.get("K",           None),
        lambda_coef=d.get("lambda_coef", 3.0),
        mu_coef=    d.get("mu_coef",     2.0),
        delta_coef= d.get("delta_coef",  1.0),
        alpha=      d.get("alpha",       1.0),
        epsilon=    d.get("epsilon",     0.05),
        tau=        d.get("tau",         10.0),
        psi=        d.get("psi",         1.0),
        time_scale= d.get("time_scale",  1000),
    )


# ---------------------------------------------------------------------------
# System configuration
#
# A SystemConfig is a tuple of (MachineConfig, n_s) pairs, one per stage.
# y[s] = (mc, n_s)  where mc is the chosen MachineConfig and n_s >= 1 is the
# number of identical parallel machines at that stage.

SystemConfig = Tuple[Tuple[MachineConfig, int], ...]


def min_machines_required(mc: MachineConfig, s: int,
                          active_jobs: List[Job]) -> int:
    """
    Minimum n_s for (mc, stage s) to satisfy the aggregate capacity
    constraint across all active jobs whose operation mc supports.

    Uses aggregate workload: n_s >= ceil(sum_j d_j * p_{j,s})
    where the sum is over active jobs with op supported by mc.
    Works correctly for heterogeneous operations: each job contributes
    its own p_{j,s} looked up via its specific operation id.
    """
    total_load = 0.0
    for job in active_jobs:
        op = job.operations[s]
        if op in mc.processing_times:          # only jobs mc can handle
            total_load += job.demand * mc.processing_times[op]
    return math.ceil(total_load) if total_load > 0 else 1


def is_capable_for_stage(mc: MachineConfig, s: int, active_jobs: List[Job]) -> bool:
    """True iff mc can perform every active job's operation at stage s."""
    return all(job.operations[s] in mc.processing_times for job in active_jobs)


def enumerate_configs(instance: RMSInstance):
    """
    Lazily yield all (config, n_machines) combinations across stages.

    Pre-filtering, in order:
      1. Capability filter: drop mc entirely if it can't perform every
         active job's operation at this stage. This is the filter that
         matters most -- without it, incapable mc still generate N_max
         dead entries each, which is what blows up the raw count.
      2. Capacity filter: for surviving mc, only generate n in
         [n_min, N_max] where n_min is the aggregate workload lower bound.
    """
    active_jobs = [j for j in instance.jobs if j.demand > 0.0]
    if not active_jobs:
        active_jobs = instance.jobs

    stage_options = []
    for s in range(instance.nb_stages):
        options = []
        for mc in instance.stage_configs[s]:
            if not is_capable_for_stage(mc, s, active_jobs):
                continue
            n_min = min_machines_required(mc, s, active_jobs)
            for n in range(n_min, instance.N_max + 1):
                options.append((mc, n))
        stage_options.append(options)
    return itertools.product(*stage_options)


def count_raw_configs(instance: RMSInstance) -> int:
    """
    Return config count after pre-filtering (capability + aggregate
    capacity lower bound) but before full feasibility check.
    No memory cost -- pure arithmetic, mirrors enumerate_configs exactly.
    """
    active_jobs = [j for j in instance.jobs if j.demand > 0.0]
    if not active_jobs:
        active_jobs = instance.jobs
    total = 1
    for s in range(instance.nb_stages):
        n_options = sum(
            max(0, instance.N_max - min_machines_required(mc, s, active_jobs) + 1)
            for mc in instance.stage_configs[s]
            if is_capable_for_stage(mc, s, active_jobs)
        )
        total *= n_options
    return total


def config_setup_cost(y: SystemConfig) -> float:
    """Total setup cost = sum over stages of n_s * mc.setup_cost."""
    return sum(n * mc.setup_cost for mc, n in y)


def get_processing_time(y: SystemConfig, job: Job, stage: int) -> float:
    mc, _ = y[stage]
    op_id = job.operations[stage]
    return mc.processing_times[op_id]


def build_pt_matrix(y: SystemConfig, instance: RMSInstance,
                    jobs: List = None) -> List[List[float]]:
    """
    Build processing_time[stage][job] matrix, in source (unscaled) units.

    Unlike the Hexaly version, Gurobi's continuous variables do not require
    integer durations, so no time_scale rounding is applied here -- the
    matrix is used directly in source units throughout, and there is no
    scale/rescale step around the solver call.

    jobs: list of Job objects to schedule (defaults to instance.jobs).
    Raises KeyError for infeasible (stage, job) pairs.
    """
    if jobs is None:
        jobs = instance.jobs
    return [
        [get_processing_time(y, job, s) for job in jobs]
        for s in range(instance.nb_stages)
    ]


def validate_config(y: SystemConfig, instance: RMSInstance) -> bool:
    """
    Return True iff the configuration is feasible for all active jobs.

    Active jobs: demand > 0. If all jobs have demand=0 (legacy instances),
    all jobs are treated as active (backward compatibility).

    Two conditions per stage s:
      1. Capability : mc can perform every active job's operation at stage s
      2. Capacity   : n_s >= ceil(sum_j d_j * p_{j,s})  [aggregate workload]
                      Works for heterogeneous ops: each job uses its own
                      operation id and processing time.
    Note: configs generated by enumerate_configs already satisfy condition 2
    by construction (n >= n_min). This check acts as a safety net and also
    handles legacy instances bypassing enumerate_configs.
    """
    active_jobs = [j for j in instance.jobs if j.demand > 0.0]
    if not active_jobs:                        # legacy: all demand=0
        active_jobs = instance.jobs
    for s, (mc, n_s) in enumerate(y):
        total_load = 0.0
        for job in active_jobs:
            # 1. Capability check
            if job.operations[s] not in mc.processing_times:
                return False
            # 2. Accumulate aggregate workload (heterogeneous ops handled
            #    naturally: each job uses its own op and processing time)
            if job.demand > 0.0:
                total_load += job.demand * mc.processing_times[job.operations[s]]
        if n_s < math.ceil(total_load):
            return False
    return True


# ---------------------------------------------------------------------------
# Reconfiguration, neighbourhood, capability

def reconfiguration_time(y: SystemConfig, y2: SystemConfig,
                         instance: RMSInstance) -> float:
    total = 0.0
    for (mc1, _), (mc2, _) in zip(y, y2):
        s1 = set(mc1.modules)
        s2 = set(mc2.modules)
        added    = len(s2 - s1)
        removed  = len(s1 - s2)
        adjusted = len(s1 & s2)
        total += (instance.lambda_coef * added
                  + instance.mu_coef   * removed
                  + instance.delta_coef * adjusted)
    return total


def neighbourhood(y: SystemConfig, all_configs: List[SystemConfig],
                  instance: RMSInstance) -> List[SystemConfig]:
    return [
        y2 for y2 in all_configs
        if y2 is not y and reconfiguration_time(y, y2, instance) <= instance.tau
    ]


def operational_capability(y: SystemConfig, instance: RMSInstance) -> float:
    total = 0.0
    for mc, _ in y:
        real_capable = sum(op != 0 for op in mc.capable_ops)
        total += max(real_capable - 1, 0) ** instance.psi
    return total


def flexibility(y: SystemConfig,
                theta_cache: Dict[int, float],
                neighbour_ids: List[int]) -> float:
    theta_y   = theta_cache[id(y)]
    theta_sum = sum(theta_cache[n_id] for n_id in neighbour_ids)
    return (theta_y + theta_sum) / (len(neighbour_ids) + 1)


def count_feasible_configs(instance: RMSInstance) -> int:
    """
    Count configs that pass validate_config, without running any scheduling.
    Same enumeration as solve_bilevel, but no solver calls -- cheap to run
    up front to know how many configs will actually be scheduled.
    """
    count = 0
    for y in enumerate_configs(instance):
        if validate_config(y, instance):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Lower-level: Gurobi Hybrid Flowshop solver (HFS || Cmax)
#
# Classic disjunctive MIP formulation for flexible/hybrid flow shops:
#   - C[j][s]            : continuous completion time of job j at stage s
#   - x[j][s][k]         : binary, 1 iff job j runs on machine k at stage s
#   - z[j1][j2][s]       : binary, 1 iff job j1 precedes job j2 on whichever
#                          machine they end up sharing at stage s
#                          (only defined for j1 < j2, both real jobs at s)
#   - Precedence (per job, across stages):
#         C[j][s] >= C[j][s-1] + p[j][s]
#   - Assignment (per real job, per stage):
#         sum_k x[j][s][k] = 1
#   - Disjunctive / no-overlap (big-M) for every unordered pair of real jobs
#     sharing stage s, linked through x and z:
#         if x[j1][s][k] = x[j2][s][k] = 1 (same machine) then either
#         j1 finishes before j2 starts, or vice versa.
#   - Objective: minimise Cmax = max_j C[j][nb_stages-1]
#
# Zero-duration ("dummy") jobs are handled the same way as in the Hexaly
# version: excluded from assignment/disjunctive constraints, pinned to
# C[j][s] = C[j][s-1] (no-op stage), and reported on machine 0.

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
                 (source units; no integer scaling required)
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
                     status is one of OPTIMAL/TIME_LIMIT/INFEASIBLE/... (see
                     _GUROBI_STATUS_NAMES). gap is the MIP gap in % (None if
                     no incumbent). This is the key field for spotting
                     configs that hit the time limit without a proven-optimal
                     Cmax -- the Gurobi analogue of Hexaly's upper-bound
                     fallback.

    Mirrors the return shape of solve_flowshop_hexaly() in rms_bilevel.py
    (plus the extra solve_info diagnostics) so that solve_bilevel() only
    needs to unpack one extra value at the call site.
    """
    # ------------------------------------------------------------------
    # Pre-compute per-stage which jobs are real (p>0) vs dummy (p=0),
    # same rationale as the Hexaly version: dummy jobs add no load and
    # are excluded from assignment/disjunctive constraints entirely.
    real_jobs = [
        [j for j in range(nb_jobs) if pt_matrix[s][j] > 0]
        for s in range(nb_stages)
    ]
    nb_real = [len(real_jobs[s]) for s in range(nb_stages)]

    eff_machines = [
        max(1, min(n_machines[s], nb_real[s])) if nb_real[s] > 0 else 1
        for s in range(nb_stages)
    ]

    # Upper bound on makespan: all real jobs sequential, used as big-M and
    # as a variable upper bound (keeps the LP relaxation tight).
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

        # --------------------------------------------------------------
        # Completion-time variables (continuous), one per (job, stage).
        # Dummy (p=0) entries still get a variable for precedence
        # chaining, exactly like Hexaly's interval variables for p=0 jobs.
        C = {
            (j, s): model.addVar(lb=0.0, ub=UB, name=f"C_{j}_{s}")
            for j in range(nb_jobs) for s in range(nb_stages)
        }

        # --------------------------------------------------------------
        # Assignment variables x[j][s][k], only for real jobs.
        x = {}
        for s in range(nb_stages):
            for j in real_jobs[s]:
                for k in range(eff_machines[s]):
                    x[j, s, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{s}_{k}")

        # --------------------------------------------------------------
        # Sequencing variables z[j1][j2][s] for every unordered pair of
        # real jobs at stage s (j1 < j2 by job index). z=1 means j1 is
        # scheduled before j2 on their shared machine (if any).
        z = {}
        for s in range(nb_stages):
            real = real_jobs[s]
            for a in range(len(real)):
                for b in range(a + 1, len(real)):
                    j1, j2 = real[a], real[b]
                    z[j1, j2, s] = model.addVar(vtype=GRB.BINARY, name=f"z_{j1}_{j2}_{s}")

        model.update()

        # --------------------------------------------------------------
        # Precedence across stages (all jobs, including dummies):
        #   C[j][s] >= C[j][s-1] + p[j][s]
        # For s == 0: C[j][0] >= p[j][0]  (job starts at t=0 at earliest)
        for j in range(nb_jobs):
            for s in range(nb_stages):
                dur = pt_matrix[s][j]
                if s == 0:
                    model.addConstr(C[j, s] >= dur, name=f"prec0_{j}_{s}")
                else:
                    model.addConstr(C[j, s] >= C[j, s - 1] + dur, name=f"prec_{j}_{s}")
            # Dummy stages collapse to the previous completion time so they
            # don't drift the chain forward (zero duration => no-op).

        # --------------------------------------------------------------
        # Assignment: every real job goes to exactly one machine at stage s.
        for s in range(nb_stages):
            for j in real_jobs[s]:
                model.addConstr(
                    gp.quicksum(x[j, s, k] for k in range(eff_machines[s])) == 1,
                    name=f"assign_{j}_{s}",
                )

        # --------------------------------------------------------------
        # Disjunctive (no-overlap) constraints for every pair of real jobs
        # that *could* share a machine at stage s. Linked through a shared
        # "same machine" indicator derived from x via big-M:
        #
        #   For each candidate machine k:
        #     C[j1] - p[j1] >= C[j2]              - BigM*(1 - z) - BigM*(2 - x[j1,k] - x[j2,k])
        #     C[j2] - p[j2] >= C[j1]              - BigM*z       - BigM*(2 - x[j1,k] - x[j2,k])
        #
        # i.e. the ordering constraint is only enforced when both jobs are
        # actually on machine k; otherwise the two extra BigM terms relax it.
        for s in range(nb_stages):
            real = real_jobs[s]
            for a in range(len(real)):
                for b in range(a + 1, len(real)):
                    j1, j2 = real[a], real[b]
                    p1, p2 = pt_matrix[s][j1], pt_matrix[s][j2]
                    zvar = z[j1, j2, s]
                    for k in range(eff_machines[s]):
                        same_machine_slack = BIGM * (2 - x[j1, s, k] - x[j2, s, k])
                        # j1 before j2 on machine k
                        model.addConstr(
                            C[j1, s] - p1 >= C[j2, s] - BIGM * (1 - zvar) - same_machine_slack,
                            name=f"seq1_{j1}_{j2}_{s}_{k}",
                        )
                        # j2 before j1 on machine k
                        model.addConstr(
                            C[j2, s] - p2 >= C[j1, s] - BIGM * zvar - same_machine_slack,
                            name=f"seq2_{j1}_{j2}_{s}_{k}",
                        )

        # --------------------------------------------------------------
        # Objective: minimise makespan = max_j C[j][last_stage]
        Cmax = model.addVar(lb=0.0, ub=UB, name="Cmax")
        for j in range(nb_jobs):
            model.addConstr(Cmax >= C[j, nb_stages - 1], name=f"cmax_{j}")
        model.setObjective(Cmax, GRB.MINIMIZE)

        model.optimize()

        status_name = _GUROBI_STATUS_NAMES.get(model.Status, f"STATUS_{model.Status}")

        if model.SolCount == 0:
            # No incumbent found within the time limit -- mirrors the
            # caller's expectation of a numeric makespan; surface a large
            # sentinel rather than crashing, with a clear stderr warning.
            print(f"[WARN] Gurobi found no feasible solution "
                  f"(status={model.Status}) within time_limit={time_limit}s",
                  file=sys.stderr)
            makespan_val = UB
            gap = None
        else:
            makespan_val = Cmax.X
            # MIPGap is undefined once status is OPTIMAL in some Gurobi
            # versions; guard defensively and report 0.0 in that case.
            try:
                gap = model.MIPGap * 100.0 if model.Status != GRB.OPTIMAL else 0.0
            except AttributeError:
                gap = None

        solve_info = {
            "status":     status_name,
            "gap":        gap,             # MIP gap in %, None if no incumbent
            "runtime":    model.Runtime,   # seconds
            "obj_bound":  model.ObjBound if model.SolCount > 0 else None,
            "node_count": int(model.NodeCount),
        }

        # --------------------------------------------------------------
        # Extract schedule. Machine assignment comes from x; start/end
        # times come from C (start = C - duration).
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
# Run output: text log (human-readable, mirrors what used to go to stdout
# only) + incremental JSON (machine-readable, for later reloading/plotting/
# comparison against the Hexaly run).
#
# Both files are flushed after every write so a run can be tailed live
# (`tail -f run.log`) or killed/crashed mid-way without losing what was
# already computed -- there is no "write everything at the end" step.

class RunLogger:
    """Writes every line to stdout and, if configured, to a .log file."""

    def __init__(self, log_path: str = None):
        self._fh = open(log_path, "a", encoding="utf-8") if log_path else None

    def write(self, line: str = ""):
        print(line)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


def _config_to_dict(y: SystemConfig) -> list:
    """JSON-friendly representation of a SystemConfig."""
    return [
        {"stage": s, "machine_id": mc.machine_id, "config_id": mc.config_id,
         "n_machines": n, "setup_cost": mc.setup_cost,
         "capable_ops": sorted(mc.capable_ops),
         "modules": list(mc.modules)}
        for s, (mc, n) in enumerate(y)
    ]


def _activity_to_list(activity: dict, scheduled_jobs: list) -> list:
    """Flatten the {(j,s): {...}} activity dict into a JSON-friendly list."""
    return [
        {"job_id": scheduled_jobs[j].job_id, "job_index": j, "stage": s,
         "machine": a["machine"], "start": a["start"], "end": a["end"]}
        for (j, s), a in sorted(activity.items())
    ]


class JsonRunWriter:
    """
    Incremental JSON writer for a solve_bilevel run.

    Strategy: keep a Python dict in memory (it's just numbers/strings, not
    full schedules for every config -- see store_full_detail below) and
    rewrite the whole file after every config. Rewriting a small/medium
    JSON file every iteration is cheap compared to a Gurobi solve, and it
    keeps the file always valid (no risk of a truncated/unparsable JSON if
    the process is killed mid-write, unlike a naive append-only format).
    """

    def __init__(self, json_path: str, meta: dict):
        self.path = json_path
        self.data = {
            "meta": meta,
            "configs": [],       # one lightweight entry per feasible config
            "summary": None,     # filled in at the end (Phase 1/2 results)
        }
        self._flush()

    def add_config_result(self, entry: dict):
        self.data["configs"].append(entry)
        self._flush()

    def set_summary(self, summary: dict):
        self.data["summary"] = summary
        self._flush()

    def _flush(self):
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        # Atomic rename: never leaves a half-written run.json on disk, even
        # if the process is killed exactly during a flush.
        os.replace(tmp_path, self.path)


# ---------------------------------------------------------------------------
# Main bi-level solver
#
# Identical control flow to rms_bilevel.solve_bilevel, with two differences:
#   1. solve_flowshop_gurobi() replaces solve_flowshop_hexaly() (single call;
#      the accidental double-call present in the Hexaly version's current
#      file has been fixed here, not reproduced).
#   2. No time_scale conversion around the solver call, since Gurobi works
#      directly in source (continuous) units -- pt_matrix is already
#      unscaled, and Q/activity/gantt come back unscaled too.

def solve_bilevel(instance: RMSInstance,
                  time_limit_per_config: int = 5,
                  mode: str = "unit",
                  verbose: bool = True,
                  two_pass: bool = False,
                  refine_time_limit: int = 20,
                  refine_corridor_factor: float = 2.0,
                  output_base: str = None,
                  instance_path: str = None) -> dict:
    """
    output_base   : if given, results are written to '<output_base>.log'
                    (human-readable, same content as stdout) and
                    '<output_base>.json' (structured, incremental -- safe
                    to read while the run is still going, and safe even if
                    the run is killed mid-way: every config's result is
                    flushed to disk as soon as it's solved, so a run left
                    going overnight never loses progress if interrupted).
                    Parent directories are created if needed.
    instance_path : path to the instance JSON, stored in the run metadata
                    purely for traceability (which file produced this run).
    two_pass             : if True, run a fast first pass (time_limit_per_config),
                            then re-solve with refine_time_limit the configs that
                            fall within refine_corridor_factor * epsilon of the
                            pass-1 best, before computing the final corridor/Phi.
    refine_time_limit     : Gurobi time limit (s) used in the refinement pass.
    refine_corridor_factor: multiplier on epsilon for the refinement corridor,
                            wider than the final (1+epsilon) corridor to catch
                            configs that pass-1 under-resolved.
    """
    if output_base:
        os.makedirs(os.path.dirname(os.path.abspath(output_base)) or ".", exist_ok=True)

    log = RunLogger(f"{output_base}.log" if output_base else None)

    json_writer = None
    if output_base:
        json_writer = JsonRunWriter(f"{output_base}.json", meta={
            "instance_path": instance_path,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "solver": "gurobi",
            "mode": mode,
            "time_limit_per_config": time_limit_per_config,
            "two_pass": two_pass,
            "refine_time_limit": refine_time_limit if two_pass else None,
            "nb_stages": instance.nb_stages,
            "nb_jobs": instance.nb_jobs,
            "N_max": instance.N_max,
            "K": instance.K,
            "alpha": instance.alpha,
            "epsilon": instance.epsilon,
            "tau": instance.tau,
            "psi": instance.psi,
            "lambda_coef": instance.lambda_coef,
            "mu_coef": instance.mu_coef,
            "delta_coef": instance.delta_coef,
        })

    run_t0 = time.time()
    # ── Active jobs ────────────────────────────────────────────────────
    active_jobs = [j for j in instance.jobs if j.demand > 0.0]
    if not active_jobs:                        # legacy: all demand=0
        if mode == "representative":
            raise ValueError(
                "mode='representative' requires at least one job with demand > 0."
            )
        active_jobs = instance.jobs

    # ── Build job list for scheduler ───────────────────────────────────
    if mode == "unit":
        scheduled_jobs = active_jobs
        batch_size     = len(scheduled_jobs)
    elif mode == "representative":
        if instance.K is None:
            raise ValueError(
                "mode='representative' requires instance.K to be set "
                "(add \"K\": <int> to the JSON)."
            )
        total_demand   = sum(j.demand for j in active_jobs)
        scheduled_jobs = []
        for j in active_jobs:
            n_copies = math.ceil(instance.K * j.demand / total_demand)
            scheduled_jobs.extend([j] * n_copies)
        batch_size = len(scheduled_jobs)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'unit' or 'representative'.")

    nb_configs = count_raw_configs(instance)
    nb_feasible = count_feasible_configs(instance)

    if verbose:
        log.write(f"[RMS] Stages={instance.nb_stages}  Jobs={instance.nb_jobs}  "
                  f"N_max={instance.N_max}  mode={mode}  solver=gurobi")
        if mode == "representative":
            log.write(f"[RMS] K={instance.K}  batch_size={batch_size}  "
                      f"active_jobs={[j.job_id for j in active_jobs]}")
        else:
            log.write(f"[RMS] active_jobs={[j.job_id for j in active_jobs]}")
        log.write(f"[RMS] Raw configs (before feasibility filter): {nb_configs:,}")
        log.write(f"[RMS] Feasible configs (will be scheduled): {nb_feasible:,}")
        log.write(f"[RMS] alpha={instance.alpha}  epsilon={instance.epsilon}  "
                  f"tau={instance.tau}  psi={instance.psi}")
        log.write("-" * 64)

    all_results: List[dict] = []
    feasible_count = 0
    n_time_limit_hits = 0   # configs where Gurobi did not prove optimality

    for idx, y in enumerate(enumerate_configs(instance)):
        if not validate_config(y, instance):
            continue

        feasible_count += 1

        pt_matrix = build_pt_matrix(y, instance, scheduled_jobs)

        # Cap n_machines[s] to the number of jobs with p > 0 at stage s.
        # Same rationale as the Hexaly version: more machines than
        # non-zero-duration jobs is meaningless and can confuse the model.
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

        # No unit conversion needed here: pt_matrix was already built in
        # source units (see build_pt_matrix), so makespan/activity/gantt
        # come back in source units directly.

        cconfig = config_setup_cost(y)
        if mode == "unit":
            perf       = makespan
            total_cost = cconfig + instance.alpha * makespan
        else:  # representative
            tp         = batch_size / makespan if makespan > 0 else float("inf")
            perf       = tp
            total_cost = cconfig + instance.alpha / tp

        all_results.append({
            "config_idx": idx,
            "y":          y,
            "C_config":   cconfig,
            "Q":          makespan,       # raw makespan always stored
            "perf":       perf,           # makespan (unit) or TP (representative)
            "total_cost": total_cost,
            "activity":   activity,
            "gantt":      gantt,
            "solve_info": solve_info,
        })

        if verbose:
            flag = "" if solve_info["status"] == "OPTIMAL" else f"  [{solve_info['status']}]"
            if mode == "unit":
                log.write(f"  [{idx+1:3d}/{nb_configs}] {_config_label(y)}"
                          f"  Cconfig={cconfig:.1f}  Cmax={makespan:.1f}"
                          f"  Total={total_cost:.1f}{flag}")
            else:
                log.write(f"  [{idx+1:3d}/{nb_configs}] {_config_label(y)}"
                          f"  Cconfig={cconfig:.1f}  Cmax={makespan:.1f}"
                          f"  TP={perf:.4f}  Total={total_cost:.4f}{flag}")

        if json_writer:
            json_writer.add_config_result({
                "config_idx": idx,
                "y":          _config_to_dict(y),
                "C_config":   cconfig,
                "Q":          makespan,
                "perf":       perf,
                "total_cost": total_cost,
                "solver_status": solve_info["status"],
                "mip_gap_pct":   solve_info["gap"],
                "solve_runtime_s": solve_info["runtime"],
                "node_count":    solve_info["node_count"],
            })

    if not all_results:
        if json_writer:
            json_writer.set_summary({"error": "No feasible configuration found."})
        log.close()
        raise RuntimeError("No feasible configuration found.")

    # ── Two-pass refinement ──────────────────────────────────────────────
    if two_pass:
        provisional_best = min(r["total_cost"] for r in all_results)
        refine_threshold = (1 + refine_corridor_factor * instance.epsilon) * provisional_best

        to_refine = [r for r in all_results if r["total_cost"] <= refine_threshold]

        if verbose:
            log.write(f"[Two-pass] Pass-1 best total_cost = {provisional_best:.4f}")
            log.write(f"[Two-pass] Refinement threshold (1+{refine_corridor_factor}*eps)*best "
                      f"= {refine_threshold:.4f}")
            log.write(f"[Two-pass] Re-solving {len(to_refine)} config(s) with "
                      f"time_limit={refine_time_limit}s")

        for r in to_refine:
            y = r["y"]
            pt_matrix = build_pt_matrix(y, instance, scheduled_jobs)
            n_machines = [
                max(1, min(n, sum(1 for j in range(batch_size)
                                  if pt_matrix[s][j] > 0)))
                for s, (_, n) in enumerate(y)
            ]

            makespan, activity, gantt, solve_info = solve_flowshop_gurobi(
                pt_matrix, batch_size, instance.nb_stages,
                n_machines=n_machines,
                time_limit=refine_time_limit,
            )

            cconfig = config_setup_cost(y)
            if mode == "unit":
                perf       = makespan
                total_cost = cconfig + instance.alpha * makespan
            else:
                tp         = batch_size / makespan if makespan > 0 else float("inf")
                perf       = tp
                total_cost = cconfig + instance.alpha / tp

            old_cost = r["total_cost"]
            r["Q"]          = makespan
            r["perf"]       = perf
            r["total_cost"] = total_cost
            r["activity"]   = activity
            r["gantt"]      = gantt
            r["solve_info"] = solve_info
            r["refined"]    = True

            if verbose and abs(total_cost - old_cost) > 1e-6:
                log.write(f"    [Refined] {_config_label(y)}  "
                          f"Total {old_cost:.4f} -> {total_cost:.4f}")

        if verbose:
            log.write("-" * 64)

    # Phase 1 — find C* and corridor
    C_star       = min(r["total_cost"] for r in all_results)
    threshold    = (1 + instance.epsilon) * C_star
    near_optimal = [r for r in all_results if r["total_cost"] <= threshold]

    if verbose:
        log.write(f"[Phase 1] C* = {C_star:.4f}")
        log.write(f"[Phase 1] Corridor threshold (1+eps)*C* = {threshold:.4f}")
        log.write(f"[Phase 1] {len(near_optimal)} config(s) in the corridor")

    # Pre-compute Theta(y) and neighbourhoods for all feasible configs
    feasible_configs = [r["y"] for r in all_results]

    theta_cache: Dict[int, float] = {
        id(r["y"]): operational_capability(r["y"], instance)
        for r in all_results
    }
    neighbour_map: Dict[int, List[int]] = {
        id(r["y"]): [id(n) for n in neighbourhood(r["y"], feasible_configs, instance)]
        for r in all_results
    }

    for r in all_results:
        y         = r["y"]
        r["Theta"] = theta_cache[id(y)]
        r["Phi"]   = flexibility(y, theta_cache, neighbour_map[id(y)])

    # Phase 2 — maximise Phi within corridor
    best_entry = max(near_optimal, key=lambda r: r["Phi"])

    if verbose:
        log.write("[Phase 2] Flexibility scores inside corridor:")
        for r in near_optimal:
            marker = " <- BEST" if r is best_entry else ""
            log.write(f"    {_config_label(r['y'])}"
                      f"  Theta={r['Theta']:.3f}  Phi={r['Phi']:.3f}{marker}")
        log.write("[Result] Optimal system configuration (Phase 2 winner):")
        for s, (mc, n) in enumerate(best_entry["y"]):
            log.write(f"    Stage {s}: machine {mc.machine_id}  "
                      f"config {mc.config_id}  "
                      f"n_machines={n}  "
                      f"capable ops={sorted(mc.capable_ops)}  "
                      f"setup cost={mc.setup_cost} (total={n*mc.setup_cost:.1f})")
        log.write(f"  C_config    = {best_entry['C_config']:.2f}")
        log.write(f"  Makespan    = {best_entry['Q']:.2f}")
        if mode == "representative":
            log.write(f"  Throughput  = {best_entry['perf']:.4f} parts/min")
        log.write(f"  Total cost  = {best_entry['total_cost']:.4f}")
        log.write(f"  Theta(y)    = {best_entry['Theta']:.4f}")
        log.write(f"  Phi(y)      = {best_entry['Phi']:.4f}")
        log.write("  Schedule (activity view):")
        log.write(f"    {'Inst':>4} {'Type':>4} {'Stage':>6} {'Machine':>8} {'Start':>7} {'End':>7}")
        for (j, s), a in sorted(best_entry["activity"].items()):
            log.write(
                f"    {j:>4} {scheduled_jobs[j].job_id:>4} "
                f"{s:>6} {a['machine']:>8} "
                f"{a['start']:>7.3f} {a['end']:>7.3f}"
            )
        log.write("  Schedule (Gantt view — per machine):")
        for s, (mc, n) in enumerate(best_entry["y"]):
            eff_n = len(best_entry["gantt"][s])   # actual machines used by solver
            for k in range(eff_n):
                jobs_on_machine = best_entry["gantt"][s][k]
                jobs_str = "  ".join(
                    f"J{e['job']}(T{scheduled_jobs[e['job']].job_id})"
                    f"[{e['start']:.3f}-{e['end']:.3f}]"
                    for e in jobs_on_machine
                )
                log.write(f"    Stage {s} machine {k}: {jobs_str}")
            if eff_n < n:
                log.write(f"    Stage {s}: {n - eff_n} additional machine(s) "
                          f"reserved for capacity but unused in this schedule")
        log.write(f"Feasible configs: {feasible_count:,}")

    run_duration_s = time.time() - run_t0

    if verbose:
        log.write("-" * 64)
        log.write(f"[Run] Total wall time: {run_duration_s:.1f}s "
                  f"({run_duration_s/60:.1f} min)")
        log.write(f"[Run] {n_time_limit_hits}/{feasible_count} config(s) did NOT "
                  f"reach a Gurobi-proven optimum within time_limit_per_config="
                  f"{time_limit_per_config}s -- see solver_status/mip_gap_pct "
                  f"per config in the JSON output for details.")

    if json_writer:
        # Full schedule detail (activity + gantt) is only kept for the
        # final corridor, not for all ~thousands of configs -- see the
        # discussion with the user: the corridor is what's actually
        # inspected afterwards, so this keeps the JSON file light while
        # still being exhaustive on the numbers (cost/Cmax/Theta/Phi/
        # solver status) for every single config above.
        corridor_detail = [
            {
                "config_idx": r["config_idx"],
                "y":          _config_to_dict(r["y"]),
                "C_config":   r["C_config"],
                "Q":          r["Q"],
                "perf":       r["perf"],
                "total_cost": r["total_cost"],
                "Theta":      r["Theta"],
                "Phi":        r["Phi"],
                "is_phase2_best": r is best_entry,
                "solve_info": r["solve_info"],
                "refined":    r.get("refined", False),
                "activity":   _activity_to_list(r["activity"], scheduled_jobs),
                "gantt": [
                    [
                        [{"job_id": scheduled_jobs[e["job"]].job_id,
                          "start": e["start"], "end": e["end"]} for e in machine]
                        for machine in stage
                    ]
                    for stage in r["gantt"]
                ],
            }
            for r in near_optimal
        ]

        json_writer.set_summary({
            "finished_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_duration_s":    run_duration_s,
            "feasible_count":    feasible_count,
            "n_time_limit_hits": n_time_limit_hits,
            "C_star":            C_star,
            "threshold":         threshold,
            "phase2_best_config_idx": best_entry["config_idx"],
            "phase2_best_y":     _config_to_dict(best_entry["y"]),
            "corridor_size":     len(near_optimal),
            "corridor_detail":   corridor_detail,
        })

    log.close()

    return {
        "C_star":       C_star,
        "threshold":    threshold,
        "near_optimal": near_optimal,
        "phase2_best":  best_entry,
        "all_results":  all_results,
    }


# ---------------------------------------------------------------------------
# Helpers

def _config_label(y: SystemConfig) -> str:
    return " | ".join(
        f"s{s}->m{mc.machine_id}k{mc.config_id}x{n}"
        for s, (mc, n) in enumerate(y)
    )


# ---------------------------------------------------------------------------
# CLI entry-point

def main():
    if len(sys.argv) < 2:
        print("Usage: python rms_bilevel_gurobi.py instance.json [time_limit] "
              "[unit|representative] [--two-pass] [refine_time_limit] "
              "[--output path/to/run_basename]")
        sys.exit(1)

    two_pass = "--two-pass" in sys.argv
    args     = [a for a in sys.argv[1:] if a != "--two-pass"]

    output_base = None
    if "--output" in args:
        i = args.index("--output")
        if i + 1 >= len(args):
            print("Error: --output requires a path argument (e.g. --output results/run1)")
            sys.exit(1)
        output_base = args[i + 1]
        args = args[:i] + args[i + 2:]   # drop --output and its value from positional args

    instance_file = args[0]
    time_limit    = int(args[1]) if len(args) >= 2 else 5
    mode          = args[2] if len(args) >= 3 else "unit"
    refine_time_limit = int(args[3]) if len(args) >= 4 else 20

    instance = load_instance_json(instance_file)
    solve_bilevel(instance, time_limit_per_config=time_limit,
                  mode=mode, verbose=True,
                  two_pass=two_pass,
                  refine_time_limit=refine_time_limit,
                  output_base=output_base,
                  instance_path=instance_file)

    if output_base:
        print(f"\n[Run] Results written to {output_base}.log and {output_base}.json")


if __name__ == "__main__":
    main()