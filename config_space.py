"""
Instance data model, loading, and configuration-space enumeration for the
RMS bi-level optimization pipeline.

This module owns everything that both the scheduling oracle (scheduling.py)
and the structural measures (structural.py) need as a common foundation:
the instance data model, JSON loading, and the enumeration/validation of
the upper-level configuration space (SystemConfig).

It has no dependency on Gurobi and no knowledge of Theta/Phi or of
alpha/epsilon/tau/lambda/mu/delta beyond carrying their values (loaded
from the instance file, used downstream by structural.py and
evaluation.py).
"""

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    jobs:          List[Job]
    nb_stages:     int
    stage_configs: List[List[MachineConfig]]    # stage_configs[s] = options at stage s
    N_max:         int = 1                      # global upper bound on machines per stage
    K:             int = None                   # representative batch size

    # Reconfiguration effort coefficients.
    #
    # Retooling (module-level; applies only when the machine CONFIG k
    # changes at a stage): lambda = add a module, mu = remove one,
    # delta = re-adjust a shared one. Model requires lambda > mu > delta > 0.
    lambda_coef: float = 3.0
    mu_coef:     float = 2.0
    delta_coef:  float = 1.0

    # Install/decommission (fleet-size level; applies whenever n_s
    # changes, independently of any config change): kappa_add = install
    # one already-qualified unit, kappa_rem = decommission one. Model
    # requires kappa_add > kappa_rem > 0. Deliberately separate from
    # lambda/mu/delta: installing an extra unit of a machine type already
    # in use needs no new tooling qualification, only physical mounting,
    # wiring and commissioning.
    kappa_add: float = 1.5
    kappa_rem: float = 1.0

    # Bi-level parameters
    alpha:   float = 1.0    # weight on Q(y) in objective
    epsilon: float = 0.05   # admissible degradation threshold
    tau:     float = 10.0   # max reconfiguration time for neighbourhood
    psi:     float = 1.0    # power index for operational capability

    @property
    def nb_jobs(self) -> int:
        return len(self.jobs)

    @property
    def n_operations(self) -> int:
        """
        |O|: size of the instance's full operation catalogue, i.e. the
        union of capable_ops across every MachineConfig at every stage.

        This is the systematic replacement for what used to be a
        hardcoded constant (e.g. 19 = 20 - 1 for one specific instance):
        it reflects *all* operations the system is capable of performing
        across the whole configuration space, not just the operations
        actually required by active jobs. Operation 0 (dummy/no-op) is
        never a key of processing_times, so it is excluded automatically.

        Used by structural.py to normalize machine versatility (theta),
        which is meant to capture capability beyond what's strictly
        required -- i.e. alternatives available in case the *nature* of
        demand changes, as distinct from surplus capacity, which captures
        adaptability to changes in demand *quantity*.
        """
        ops: Set[int] = set()
        for stage_configs in self.stage_configs:
            for mc in stage_configs:
                ops.update(mc.capable_ops)
        return len(ops)


def load_instance_json(path: str) -> RMSInstance:
    """
    Load an RMS instance from a JSON file.
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
        # Defaulted so instance files predating the install/decommission
        # term load unchanged; override per-instance by adding the keys.
        kappa_add=  d.get("kappa_add",   1.5),
        kappa_rem=  d.get("kappa_rem",   1.0),
        alpha=      d.get("alpha",       1.0),
        epsilon=    d.get("epsilon",     0.05),
        tau=        d.get("tau",         10.0),
        psi=        d.get("psi",         1.0),
    )


# ---------------------------------------------------------------------------
# System configuration
#
# A SystemConfig is a tuple of (MachineConfig, n_s) pairs, one per stage.
# y[s] = (mc, n_s)  where mc is the chosen MachineConfig and n_s >= 1 is the
# number of identical parallel machines at that stage.
# ---------------------------------------------------------------------------

SystemConfig = Tuple[Tuple[MachineConfig, int], ...]


def active_jobs_of(instance: RMSInstance) -> List[Job]:
    """
    Jobs with demand > 0. Falls back to all jobs if none have positive
    demand (legacy instances where demand was never populated).
    """
    active = [j for j in instance.jobs if j.demand > 0.0]
    return active if active else instance.jobs


def min_machines_required(mc: MachineConfig, s: int,
                          active_jobs: List[Job]) -> int:
    """
    Minimum n_s for (mc, stage s) to satisfy the aggregate capacity
    constraint across all active jobs whose operation mc supports.

    Uses aggregate workload: n_s >= ceil(sum_j d_j * p_{j,s})
    where the sum is over active jobs with op supported by mc.
    """
    total_load = 0.0
    for job in active_jobs:
        op = job.operations[s]
        if op in mc.processing_times:
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
         active job's operation at this stage.
      2. Capacity filter: for surviving mc, only generate n in
         [n_min, N_max] where n_min is the aggregate workload lower bound.
    """
    active_jobs = active_jobs_of(instance)

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
    Config count after pre-filtering (capability + aggregate capacity
    lower bound) but before full feasibility check. Pure arithmetic,
    mirrors enumerate_configs exactly, no memory cost.
    """
    active_jobs = active_jobs_of(instance)
    total = 1
    for s in range(instance.nb_stages):
        n_options = sum(
            max(0, instance.N_max - min_machines_required(mc, s, active_jobs) + 1)
            for mc in instance.stage_configs[s]
            if is_capable_for_stage(mc, s, active_jobs)
        )
        total *= n_options
    return total


def validate_config(y: SystemConfig, instance: RMSInstance) -> bool:
    """
    Return True iff the configuration is feasible for all active jobs.

    Two conditions per stage s:
      1. Capability : mc can perform every active job's operation at stage s
      2. Capacity   : n_s >= ceil(sum_j d_j * p_{j,s})  [aggregate workload]

    Note: configs generated by enumerate_configs already satisfy condition 2
    by construction (n >= n_min). This check acts as a safety net and also
    handles configs built outside enumerate_configs.
    """
    active_jobs = active_jobs_of(instance)
    for s, (mc, n_s) in enumerate(y):
        total_load = 0.0
        for job in active_jobs:
            if job.operations[s] not in mc.processing_times:
                return False
            if job.demand > 0.0:
                total_load += job.demand * mc.processing_times[job.operations[s]]
        if n_s < math.ceil(total_load):
            return False
    return True


def feasible_configs(instance: RMSInstance) -> List[SystemConfig]:
    """
    All configs that pass validate_config, materialized as a list.

    This is the single canonical entry point that scheduling.py,
    structural.py and any orchestration script should use to get "the"
    configuration space for an instance -- guarantees every consumer
    enumerates and filters the same way.
    """
    return [y for y in enumerate_configs(instance) if validate_config(y, instance)]


def count_feasible_configs(instance: RMSInstance) -> int:
    """Count configs that pass validate_config, without materializing them."""
    count = 0
    for y in enumerate_configs(instance):
        if validate_config(y, instance):
            count += 1
    return count


def config_setup_cost(y: SystemConfig) -> float:
    """Total setup cost = sum over stages of n_s * mc.setup_cost."""
    return sum(n * mc.setup_cost for mc, n in y)


def get_processing_time(y: SystemConfig, job: Job, stage: int) -> float:
    mc, _ = y[stage]
    op_id = job.operations[stage]
    return mc.processing_times[op_id]


def build_pt_matrix(y: SystemConfig, instance: RMSInstance,
                    jobs: List[Job] = None) -> List[List[float]]:
    """
    Build processing_time[stage][job] matrix, in source (unscaled) units.

    jobs: list of Job objects to schedule (defaults to instance.jobs).
    Raises KeyError for infeasible (stage, job) pairs.
    """
    if jobs is None:
        jobs = instance.jobs
    return [
        [get_processing_time(y, job, s) for job in jobs]
        for s in range(instance.nb_stages)
    ]


def config_label(y: SystemConfig) -> str:
    return " | ".join(
        f"s{s}->m{mc.machine_id}k{mc.config_id}x{n}"
        for s, (mc, n) in enumerate(y)
    )


def config_to_dict(y: SystemConfig) -> list:
    """JSON-friendly representation of a SystemConfig."""
    return [
        {"stage": s, "machine_id": mc.machine_id, "config_id": mc.config_id,
         "n_machines": n, "setup_cost": mc.setup_cost,
         "capable_ops": sorted(mc.capable_ops),
         "modules": list(mc.modules)}
        for s, (mc, n) in enumerate(y)
    ]