"""
Exhaustive upper-level enumeration + Hexaly lower-level hybrid flowshop solver.

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

Usage:
    python rms_bilevel.py instance.json [time_limit_per_config]
"""

import hexaly.optimizer
import itertools
import sys
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Data Classes

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
    time_scale: int = 1000  # internal units per source unit (Hexaly requires
                             # integer interval lengths; this scales fractional
                             # processing times, e.g. derived from parts/hour,
                             # into integers without precision loss)


    @property
    def nb_jobs(self) -> int:
        return len(self.jobs)


def load_instance_json(path: str) -> RMSInstance:
    """
    Load an RMS instance from a JSON file.

    JSON schema
    -----------
    {
        "nb_stages": <int>,
        "N_max":       <int>,     // optional, default 1
        "lambda_coef": <float>,   // optional, default 3
        "mu_coef":     <float>,   // optional, default 2
        "delta_coef":  <float>,   // optional, default 1
        "alpha":       <float>,   // optional, default 1
        "epsilon":     <float>,   // optional, default 0.05
        "tau":         <float>,   // optional, default 10
        "psi":         <float>,   // optional, default 1
        "time_scale":  <int>,     // optional, default 1000

        "jobs": [
            { "job_id": 0, "operations": [0, 1, 2] },
            ...
        ],

        "stages": [
            [
                {
                    "machine_id": 0,
                    "config_id":  0,
                    "setup_cost": 10.0,
                    "processing_times": {"0": 4.0, "2": 6.0},
                    "modules": [0, 1, 2]
                },
                ...
            ],
            ...
        ]
    }
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
# A SystemConfig is now a tuple of (MachineConfig, n_s) pairs, one per stage.
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
         matters most — without it, incapable mc still generate N_max
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
    No memory cost — pure arithmetic, mirrors enumerate_configs exactly.
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
                    jobs: List = None) -> List[List[int]]:
    """
    Build processing_time[stage][job] matrix, in internal (scaled) units.

    Hexaly's interval/length model requires integer durations. Source data
    (e.g. processing times derived from parts/hour demand rates) can be
    fractional, so values are scaled by instance.time_scale and rounded
    to the nearest integer before being passed to the solver. Reconversion
    to source units happens on the way out, in solve_bilevel.

    jobs: list of Job objects to schedule (defaults to instance.jobs).
    Raises KeyError for infeasible (stage, job) pairs.
    """
    if jobs is None:
        jobs = instance.jobs
    scale = instance.time_scale
    return [
        [int(round(get_processing_time(y, job, s) * scale)) for job in jobs]
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
# Reconfiguration, neighbourhood, capability (unchanged in logic,
# updated to unpack (mc, n) tuples)

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
        nb_capable = len(mc.capable_ops)
        total += max(nb_capable - 1, 0) ** instance.psi
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
    Same enumeration as solve_bilevel, but no Hexaly calls — cheap to run
    up front to know how many configs will actually be scheduled.
    """
    count = 0
    for y in enumerate_configs(instance):
        if validate_config(y, instance):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Lower-level: Hexaly Hybrid Flowshop solver (HFS || Cmax)
#
# Formulation based on the JSSP interval + list approach:
#   - One interval variable act[j][s] per (job, stage) with fixed duration
#   - Precedence: end(act[j][s-1]) <= start(act[j][s])  for all j, s > 0
#   - n_s list variables at stage s (one per physical machine)
#   - Partition constraint: every job appears on exactly one machine at each stage
#   - Disjunctive: jobs on the same machine don't overlap (via list ordering)
#   - Objective: Cmax = max over j of end(act[j][last_stage])

def solve_flowshop_hexaly(pt_matrix:  List[List[int]],
                          nb_jobs:    int,
                          nb_stages:  int,
                          n_machines: List[int],
                          time_limit: int = 5) -> Tuple[float, List[List[int]]]:
    """
    Solve the Hybrid Flowshop minimising makespan via Hexaly.

    Parameters
    ----------
    pt_matrix  : pt_matrix[s][j] = processing time of job j at stage s
    nb_jobs    : number of jobs
    nb_stages  : number of stages
    n_machines : n_machines[s] = number of identical parallel machines at stage s
    time_limit : Hexaly time limit in seconds

    Returns
    -------
    (makespan, assignment)
        makespan   : float
        assignment : assignment[s][j] = index of physical machine (0-based)
                     assigned to job j at stage s
    """
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        optimizer.param.verbosity = 0
        model = optimizer.model

        # ------------------------------------------------------------------
        # Pre-compute per-stage which jobs are real (p>0) vs dummy (p=0).
        # Zero-duration jobs are excluded from machine lists and disjunctive
        # constraints entirely — they impose no load and their presence in
        # the partition causes Hexaly to fail even with generous time limits.
        # They still have interval variables (for precedence chaining) but
        # are pinned to start=0, end=0 and assigned to machine 0 by default.

        real_jobs = [
            [j for j in range(nb_jobs) if pt_matrix[s][j] > 0]
            for s in range(nb_stages)
        ]
        nb_real = [len(real_jobs[s]) for s in range(nb_stages)]

        # ------------------------------------------------------------------
        # Upper bound: worst-case makespan = all real jobs sequentially
        UB = sum(pt_matrix[s][j]
                 for s in range(nb_stages)
                 for j in range(nb_jobs)
                 if pt_matrix[s][j] > 0) + 1

        # ------------------------------------------------------------------
        # Interval variables — one per (job, stage)
        act = [
            [model.interval(0, UB) for s in range(nb_stages)]
            for j in range(nb_jobs)
        ]

        # Fix duration of each interval to the processing time
        for j in range(nb_jobs):
            for s in range(nb_stages):
                model.constraint(model.eq(model.length(act[j][s]), pt_matrix[s][j]))

        # ------------------------------------------------------------------
        # Machine list variables — only over real jobs at each stage.
        # n_machines[s] is already capped upstream, but we also cap here
        # to nb_real[s] as a safety net.
        eff_machines = [
            max(1, min(n_machines[s], nb_real[s])) if nb_real[s] > 0 else 1
            for s in range(nb_stages)
        ]

        # machine_lists[s][k] partitions only the real jobs at stage s.
        # For all-dummy stages (nb_real[s]==0) we create no lists —
        # model.list(0) is illegal in Hexaly.
        machine_lists = [
            [model.list(nb_real[s]) for k in range(eff_machines[s])]
            if nb_real[s] > 0 else []
            for s in range(nb_stages)
        ]

        # ------------------------------------------------------------------
        # Partition constraint over real jobs only
        for s in range(nb_stages):
            if nb_real[s] > 0:
                model.constraint(model.partition(*machine_lists[s]))

        # ------------------------------------------------------------------
        # Precedence constraints: stage ordering per job (all jobs)
        for j in range(nb_jobs):
            for s in range(1, nb_stages):
                model.constraint(
                    model.geq(model.start(act[j][s]), model.end(act[j][s - 1]))
                )

        # ------------------------------------------------------------------
        # Disjunctive constraints over real jobs only.
        # machine_lists[s][k] contains indices into real_jobs[s], so we
        # use real_jobs[s][i] to look up the actual job index for act.

        for s in range(nb_stages):
            if nb_real[s] == 0:
                continue

            real = real_jobs[s]
            # Arrays indexed by position in real_jobs[s]
            starts = model.array([model.start(act[real[r]][s]) for r in range(nb_real[s])])
            ends   = model.array([model.end(act[real[r]][s])   for r in range(nb_real[s])])

            for k in range(eff_machines[s]):
                lst = machine_lists[s][k]

                sequence_fn = model.lambda_function(
                    lambda i: model.geq(
                        starts[lst[i]],
                        ends[lst[i - 1]]
                    )
                )
                model.constraint(
                    model.and_(
                        model.range(1, model.count(lst)),
                        sequence_fn
                    )
                )

        # ------------------------------------------------------------------
        # Objective: minimise makespan
        all_ends = [model.end(act[j][nb_stages - 1]) for j in range(nb_jobs)]
        makespan = model.max(*all_ends)
        model.minimize(makespan)
        model.close()

        optimizer.param.time_limit = time_limit
        optimizer.solve()

        # ------------------------------------------------------------------
        # Extract schedule from solution
        #
        # activity[(j, s)] = {"machine": k, "start": t, "end": t}
        # gantt[s][k]      = sorted list of {"job": j, "start": t, "end": t}

        activity = {}
        gantt    = [[[] for _ in range(eff_machines[s])] for s in range(nb_stages)]

        # First pass: record machine assignment from list variables.
        # machine_lists[s][k] contains indices into real_jobs[s].
        # Dummy jobs (p=0) are assigned to machine 0 by default.
        assignment = [[0] * nb_jobs for _ in range(nb_stages)]
        for s in range(nb_stages):
            if nb_real[s] == 0:
                continue           # all-dummy stage: no lists, all default to machine 0
            for k in range(eff_machines[s]):
                for r in machine_lists[s][k].value:
                    j = real_jobs[s][r]     # map back to actual job index
                    assignment[s][j] = k

        # Second pass: extract start/end from interval variables
        for j in range(nb_jobs):
            for s in range(nb_stages):
                k       = assignment[s][j]
                iv      = act[j][s].value  # HxIntervalValue
                t_start = int(iv.start())
                t_end   = int(iv.end())
                activity[(j, s)] = {"machine": k, "start": t_start, "end": t_end}
                gantt[s][k].append({"job": j, "start": t_start, "end": t_end})

        # Sort each machine's job list by start time
        for s in range(nb_stages):
            for k in range(eff_machines[s]):
                gantt[s][k].sort(key=lambda x: x["start"])

        return float(makespan.value), activity, gantt


# ---------------------------------------------------------------------------
# Main bi-level solver

def solve_bilevel(instance: RMSInstance,
                  time_limit_per_config: int = 5,
                  mode: str = "unit",
                  verbose: bool = True,
                  two_pass: bool = False,
                  refine_time_limit: int = 20,
                  refine_corridor_factor: float = 2.0) -> dict:
    """
    two_pass             : if True, run a fast first pass (time_limit_per_config),
                            then re-solve with refine_time_limit the configs that
                            fall within refine_corridor_factor * epsilon of the
                            pass-1 best, before computing the final corridor/Phi.
    refine_time_limit     : Hexaly time limit (s) used in the refinement pass.
    refine_corridor_factor: multiplier on epsilon for the refinement corridor,
                            wider than the final (1+epsilon) corridor to catch
                            configs that pass-1 under-resolved.
    """
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
        print(f"[RMS] Stages={instance.nb_stages}  Jobs={instance.nb_jobs}  "
              f"N_max={instance.N_max}  mode={mode}")
        if mode == "representative":
            print(f"[RMS] K={instance.K}  batch_size={batch_size}  "
                  f"active_jobs={[j.job_id for j in active_jobs]}")
        else:
            print(f"[RMS] active_jobs={[j.job_id for j in active_jobs]}")
        print(f"[RMS] Raw configs (before feasibility filter): {nb_configs:,}")
        print(f"[RMS] Feasible configs (will be scheduled): {nb_feasible:,}")
        print(f"[RMS] alpha={instance.alpha}  epsilon={instance.epsilon}  "
              f"tau={instance.tau}  psi={instance.psi}")
        print("-" * 64)

    all_results: List[dict] = []
    feasible_count = 0

    for idx, y in enumerate(enumerate_configs(instance)):
        if not validate_config(y, instance):
            continue

        feasible_count += 1

        pt_matrix  = build_pt_matrix(y, instance, scheduled_jobs)

        # Cap n_machines[s] to the number of jobs with p > 0 at stage s.
        # Having more machines than non-zero jobs causes Hexaly's partition
        # constraints to behave unexpectedly (returns upper bound 100000).
        # This covers both all-dummy stages (cap=0 -> 1) and partially-dummy
        # stages (e.g. 1 real job, n=2 -> capped to 1).
        n_machines = [
            max(1, min(n, sum(1 for j in range(batch_size)
                              if pt_matrix[s][j] > 0)))
            for s, (_, n) in enumerate(y)
        ]

        makespan, activity, gantt = solve_flowshop_hexaly(
            pt_matrix, batch_size, instance.nb_stages,
            n_machines=n_machines,
            time_limit=time_limit_per_config,
        )

        # Convert back from internal (scaled) units to source units.
        # solve_flowshop_hexaly stays unit-agnostic; this is the one place
        # where scaled units re-enter the RMS model's native unit system.
        scale = instance.time_scale
        makespan = makespan / scale
        activity = {
            k: {"machine": v["machine"],
                "start": v["start"] / scale,
                "end":   v["end"] / scale}
            for k, v in activity.items()
        }
        gantt = [
            [
                [{"job": e["job"], "start": e["start"] / scale, "end": e["end"] / scale}
                 for e in machine_jobs]
                for machine_jobs in stage_gantt
            ]
            for stage_gantt in gantt
        ]

        cconfig = config_setup_cost(y)
        if mode == "unit":
            perf       = makespan
            total_cost = cconfig + instance.alpha * makespan
        else:  # representative
            tp         = batch_size / makespan if makespan > 0 else float("inf")
            perf       = tp
            total_cost = cconfig + instance.alpha / tp
        makespan = makespan  # keep raw makespan for reporting

        all_results.append({
            "config_idx": idx,
            "y":          y,
            "C_config":   cconfig,
            "Q":          makespan,       # raw makespan always stored
            "perf":       perf,           # makespan (unit) or TP (representative)
            "total_cost": total_cost,
            "activity":   activity,
            "gantt":      gantt,
        })

        if verbose:
            if mode == "unit":
                print(f"  [{idx+1:3d}/{nb_configs}] {_config_label(y)}"
                      f"  Cconfig={cconfig:.1f}  Cmax={makespan:.1f}"
                      f"  Total={total_cost:.1f}")
            else:
                print(f"  [{idx+1:3d}/{nb_configs}] {_config_label(y)}"
                      f"  Cconfig={cconfig:.1f}  Cmax={makespan:.1f}"
                      f"  TP={perf:.4f}  Total={total_cost:.4f}")

    if not all_results:
        raise RuntimeError("No feasible configuration found.")

    # ── Two-pass refinement ──────────────────────────────────────────────
    if two_pass:
        provisional_best = min(r["total_cost"] for r in all_results)
        refine_threshold = (1 + refine_corridor_factor * instance.epsilon) * provisional_best

        to_refine = [r for r in all_results if r["total_cost"] <= refine_threshold]

        if verbose:
            print(f"[Two-pass] Pass-1 best total_cost = {provisional_best:.4f}")
            print(f"[Two-pass] Refinement threshold (1+{refine_corridor_factor}*eps)*best "
                  f"= {refine_threshold:.4f}")
            print(f"[Two-pass] Re-solving {len(to_refine)} config(s) with "
                  f"time_limit={refine_time_limit}s")

        for r in to_refine:
            y = r["y"]
            pt_matrix = build_pt_matrix(y, instance, scheduled_jobs)
            n_machines = [
                max(1, min(n, sum(1 for j in range(batch_size)
                                  if pt_matrix[s][j] > 0)))
                for s, (_, n) in enumerate(y)
            ]

            makespan, activity, gantt = solve_flowshop_hexaly(
                pt_matrix, batch_size, instance.nb_stages,
                n_machines=n_machines,
                time_limit=refine_time_limit,
            )

            scale = instance.time_scale
            makespan = makespan / scale
            activity = {
                k: {"machine": v["machine"],
                    "start": v["start"] / scale,
                    "end":   v["end"] / scale}
                for k, v in activity.items()
            }
            gantt = [
                [
                    [{"job": e["job"], "start": e["start"] / scale, "end": e["end"] / scale}
                     for e in machine_jobs]
                    for machine_jobs in stage_gantt
                ]
                for stage_gantt in gantt
            ]

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
            r["refined"]    = True

            if verbose and abs(total_cost - old_cost) > 1e-6:
                print(f"    [Refined] {_config_label(y)}  "
                      f"Total {old_cost:.4f} -> {total_cost:.4f}")

        if verbose:
            print("-" * 64)

    # Phase 1 — find C* and corridor
    C_star       = min(r["total_cost"] for r in all_results)
    threshold    = (1 + instance.epsilon) * C_star
    near_optimal = [r for r in all_results if r["total_cost"] <= threshold]

    if verbose:
        print(f"[Phase 1] C* = {C_star:.4f}")
        print(f"[Phase 1] Corridor threshold (1+eps)*C* = {threshold:.4f}")
        print(f"[Phase 1] {len(near_optimal)} config(s) in the corridor")

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
        print("[Phase 2] Flexibility scores inside corridor:")
        for r in near_optimal:
            marker = " <- BEST" if r is best_entry else ""
            print(f"    {_config_label(r['y'])}"
                  f"  Theta={r['Theta']:.3f}  Phi={r['Phi']:.3f}{marker}")
        print("[Result] Optimal system configuration (Phase 2 winner):")
        for s, (mc, n) in enumerate(best_entry["y"]):
            print(f"    Stage {s}: machine {mc.machine_id}  "
                  f"config {mc.config_id}  "
                  f"n_machines={n}  "
                  f"capable ops={sorted(mc.capable_ops)}  "
                  f"setup cost={mc.setup_cost} (total={n*mc.setup_cost:.1f})")
        print(f"  C_config    = {best_entry['C_config']:.2f}")
        print(f"  Makespan    = {best_entry['Q']:.2f}")
        if mode == "representative":
            print(f"  Throughput  = {best_entry['perf']:.4f} parts/min")
        print(f"  Total cost  = {best_entry['total_cost']:.4f}")
        print(f"  Theta(y)    = {best_entry['Theta']:.4f}")
        print(f"  Phi(y)      = {best_entry['Phi']:.4f}")
        print("  Schedule (activity view):")
        print(f"    {"Job":>4} {"Stage":>6} {"Machine":>8} {"Start":>7} {"End":>7}")
        for (j, s), a in sorted(best_entry["activity"].items()):
            print(f"    {scheduled_jobs[j].job_id:>4} {s:>6} {a['machine']:>8} "
                  f"{a['start']:>7} {a['end']:>7}")
        print("  Schedule (Gantt view — per machine):")
        for s, (mc, n) in enumerate(best_entry["y"]):
            eff_n = len(best_entry["gantt"][s])   # actual machines used by solver
            for k in range(eff_n):
                jobs_on_machine = best_entry["gantt"][s][k]
                jobs_str = "  ".join(
                    f"j{instance.jobs[e['job']].job_id}[{e['start']:.3f}-{e['end']:.3f}]"
                    for e in jobs_on_machine
                )
                print(f"    Stage {s} machine {k}: {jobs_str}")
            if eff_n < n:
                print(f"    Stage {s}: {n - eff_n} additional machine(s) "
                      f"reserved for capacity but unused in this schedule")
        print(f"Feasible configs: {feasible_count:,}")

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
        print("Usage: python rms_bilevel.py instance.json [time_limit] [unit|representative] [--two-pass] [refine_time_limit]")
        sys.exit(1)

    two_pass = "--two-pass" in sys.argv
    args     = [a for a in sys.argv[1:] if a != "--two-pass"]

    instance_file = args[0]
    time_limit    = int(args[1]) if len(args) >= 2 else 5
    mode          = args[2] if len(args) >= 3 else "unit"
    refine_time_limit = int(args[3]) if len(args) >= 4 else 20

    instance = load_instance_json(instance_file)
    solve_bilevel(instance, time_limit_per_config=time_limit,
                  mode=mode, verbose=True,
                  two_pass=two_pass,
                  refine_time_limit=refine_time_limit)


if __name__ == "__main__":
    main()