"""
Structural (scheduling-independent) measures for the RMS bi-level
optimization pipeline: operational capability Theta(y), neighbourhood
flexibility Phi(y), and reconfiguration time / neighbourhood.

None of these depend on the lower-level scheduling result (Q(y), the
Gantt/activity detail) -- they are pure functions of an instance and a
configuration's structure (which MachineConfig is chosen at each stage,
how many machines, and the instance's job/demand data). This is what
makes it possible to compute Phi for the *entire* feasible configuration
space before ever calling the scheduling oracle, and to filter on it
up front (see the partial-inversion experiment in the orchestration
layer).

Theta(y) has two components:
  - machine versatility: alternatives available if the *nature* of
    demand changes (capability beyond what's strictly required at the
    assigned stage), normalized by the instance's full operation
    catalogue size (n_operations, from config_space.RMSInstance).
  - surplus capacity: adaptability to changes in demand *quantity*
    (how much spare throughput a stage has relative to current demand).

Formula reference: this module implements the normalized version
(explore_phase2.py in the original codebase), confirmed as the
authoritative definition going forward. The older unnormalized version
(rms_bilevel_gurobi.py) is not reproduced here.

Configuration identity: every function here that returns a per-config
result is keyed by config_idx (the config's position in whatever
List[SystemConfig] the caller passed in -- canonically
config_space.feasible_configs(instance)), never by id(y). This makes
results serializable and stable across processes, unlike the original
id(y)-keyed caches.
"""

from typing import Dict, List, Tuple

from config_space import Job, MachineConfig, RMSInstance, SystemConfig


# ---------------------------------------------------------------------------
# Machine versatility (demand-nature component of Theta)
# ---------------------------------------------------------------------------

def machine_versatility(mc: MachineConfig, n_operations: int) -> float:
    """
    theta_{i,k} = (real_capable - 1) / (n_operations - 1)

    real_capable is the number of non-dummy operations mc can perform.
    The '-1' removes the operation currently required at the assigned
    stage, so this measures capability *beyond* what's strictly needed --
    i.e. alternatives available if the nature of demand were to change.

    n_operations is the instance's full operation catalogue size
    (RMSInstance.n_operations), not a hardcoded constant.
    """
    real_capable = sum(op != 0 for op in mc.capable_ops)
    return max(real_capable - 1, 0) / (n_operations - 1)


def machine_versatility_sum(y: SystemConfig, n_operations: int) -> float:
    """Sum theta_{i_s,k_s} over all stages of configuration y."""
    return sum(machine_versatility(mc, n_operations) for mc, _ in y)


# ---------------------------------------------------------------------------
# Stage operation and demand (instance-level, config-independent --
# computed once per instance, not once per config)
# ---------------------------------------------------------------------------

def compute_stage_operations(instance: RMSInstance) -> List[int]:
    """
    Recover the unique operation o_s associated with each stage s.

    Some stages may be dummy/no-op stages: all active jobs have operation 0
    there. In that case the entry for that stage is 0 and callers should
    treat it as inactive (zero demand, zero surplus contribution).

    This depends only on the instance (jobs, demand), not on any specific
    configuration -- compute once per instance and reuse across every
    config's surplus_capacity() call, rather than recomputing per config.
    """
    stage_ops = []
    for s in range(instance.nb_stages):
        operations = {
            job.operations[s]
            for job in instance.jobs
            if job.demand > 0.0 and job.operations[s] != 0
        }
        if not operations:
            stage_ops.append(0)
        elif len(operations) == 1:
            stage_ops.append(next(iter(operations)))
        else:
            raise ValueError(
                f"Stage {s} does not have exactly one unique non-zero "
                f"operation. Found: {sorted(operations)}"
            )
    return stage_ops


def compute_stage_demands(instance: RMSInstance, stage_ops: List[int]) -> List[float]:
    """
    D_{o_s}: aggregate demand associated with each stage's operation.
    Dummy stages (stage_ops[s] == 0) get 0.0 by definition.
    """
    stage_demands = []
    for s, o_s in enumerate(stage_ops):
        if o_s == 0:
            stage_demands.append(0.0)
            continue
        total = sum(
            job.demand
            for job in instance.jobs
            if job.demand > 0.0 and job.operations[s] == o_s
        )
        stage_demands.append(total)
    return stage_demands


# ---------------------------------------------------------------------------
# Surplus capacity (demand-quantity component of Theta)
# ---------------------------------------------------------------------------

def surplus_capacity(y: SystemConfig, stage_ops: List[int],
                     stage_demands: List[float]) -> float:
    """
    sum_s sigma_s(y), where:

        sigma_s(y) = n_s * r_{i_s,o_s}^{k_s} / D_{o_s} - 1
        r_{i_s,o_s}^{k_s} = 1 / p_{o_s,(i_s,k_s)}

    Dummy/no-op stages (stage_ops[s] == 0) or stages with zero aggregate
    demand contribute 0. Not normalized or capped, unlike machine
    versatility.
    """
    total = 0.0

    for s, (mc, n) in enumerate(y):
        o_s = stage_ops[s]
        if o_s == 0:
            continue

        D_os = stage_demands[s]
        if D_os <= 0.0:
            continue

        if o_s not in mc.processing_times:
            raise ValueError(
                f"Stage {s}: machine (id={mc.machine_id}, "
                f"config={mc.config_id}) cannot perform operation {o_s}."
            )

        p = mc.processing_times[o_s]
        if p <= 0.0:
            raise ValueError(
                f"Stage {s}: non-positive processing time p={p} "
                f"for operation {o_s}."
            )

        r = 1.0 / p
        sigma = n * r / D_os - 1.0
        if -1e-10 < sigma < 0.0:
            sigma = 0.0

        total += sigma

    return total


# ---------------------------------------------------------------------------
# Combined operational capability Theta
# ---------------------------------------------------------------------------

def operational_capability(y: SystemConfig, n_operations: int,
                           stage_ops: List[int],
                           stage_demands: List[float]) -> float:
    """
    Theta(y) = sum_s theta_{i_s,k_s} + sum_s sigma_s(y)

    Versatility term normalized by (n_operations - 1); surplus-capacity
    term deliberately not normalized or bounded.
    """
    versatility = machine_versatility_sum(y, n_operations)
    surplus = surplus_capacity(y, stage_ops, stage_demands)
    return versatility + surplus


# ---------------------------------------------------------------------------
# Reconfiguration time and neighbourhood
# ---------------------------------------------------------------------------

def same_machine_types(y1: SystemConfig, y2: SystemConfig) -> bool:
    """
    True iff y1 and y2 use the same machine TYPE (machine_id, i) at every
    stage -- i.e. they differ only in machine configuration (config_id,
    k) and/or in the number of deployed units (n_s).

    A MachineConfig is made of two kinds of modules: principal modules,
    which define the machine type i, and auxiliary modules, which define
    the configuration k. Only auxiliary modules are listed in
    MachineConfig.modules -- principal modules are not swappable in this
    model, so a reconfiguration can never change a stage's machine type.
    Configurations differing in machine type at any stage are therefore
    mutually unreachable, whatever the effort budget tau.
    """
    return all(mc1.machine_id == mc2.machine_id for (mc1, _), (mc2, _) in zip(y1, y2))


def reconfiguration_time(y1: SystemConfig, y2: SystemConfig,
                         instance: RMSInstance) -> float:
    """
    Directional reconfiguration effort T_rec(y1 -> y2), summed over stages:

        T_rec = sum_s [ T_retool,s + T_count,s ]

    Retooling (module-level), guarded by a type-change indicator:

        T_retool,s = 1[(i_s,k_s) != (i_s',k_s')]
                     * min(n_s, n_s')
                     * [ lambda*A_s + mu*R_s + delta*J_s ]

        A_s = |M' \\ M|   modules added
        R_s = |M \\ M'|   modules removed
        J_s = |M & M'|   modules shared and re-adjusted

    The indicator is essential: without it, a stage whose machine-config
    pair is UNCHANGED would still score delta*|M| (full overlap counted
    as re-adjustment) even though nothing physically happens there. That
    matters precisely because n_s may vary independently of (i_s,k_s):
    a pure change in unit count must not be priced as retooling.

    The min(n_s, n_s') factor scales retooling to the machines that
    actually SURVIVE the transition: added units arrive already qualified
    for the target config (that is what kappa_add pays for), and units
    about to be decommissioned are not retooled first.

    Install/decommission (fleet-size level), applied independently of any
    type change:

        T_count,s = kappa_add * max(n_s' - n_s, 0)
                  + kappa_rem * max(n_s - n_s', 0)

    Returns +inf when the two configurations do not share the same
    machine type at every stage (see same_machine_types): such a pair is
    unreachable by reconfiguration, not merely expensive.

    Asymmetric by construction (lambda != mu, kappa_add != kappa_rem):
    T_rec(y1 -> y2) generally differs from T_rec(y2 -> y1), so
    neighbourhoods are directional.
    """
    if not same_machine_types(y1, y2):
        return float("inf")

    total = 0.0
    for (mc1, n1), (mc2, n2) in zip(y1, y2):
        # --- Retooling term (only if the machine-config pair changed) ---
        if (mc1.machine_id, mc1.config_id) != (mc2.machine_id, mc2.config_id):
            modules_1 = set(mc1.modules)
            modules_2 = set(mc2.modules)
            added = len(modules_2 - modules_1)
            removed = len(modules_1 - modules_2)
            adjusted = len(modules_1 & modules_2)
            per_unit = (instance.lambda_coef * added
                        + instance.mu_coef * removed
                        + instance.delta_coef * adjusted)
            total += min(n1, n2) * per_unit

        # --- Install / decommission term (always, independently) ---
        total += (instance.kappa_add * max(n2 - n1, 0)
                  + instance.kappa_rem * max(n1 - n2, 0))

    return total


# ---------------------------------------------------------------------------
# Main entry point: Theta / Phi / neighbourhoods for a whole config list
# ---------------------------------------------------------------------------

def compute_measures(
    instance: RMSInstance,
    configs: List[SystemConfig],
    config_ids: List[int] = None,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, List[int]]]:
    """
    Compute Theta(y), Phi(y), and neighbourhoods for every configuration
    in `configs`, keyed by config_ids[i] (defaults to range(len(configs))
    if not given, i.e. plain position in the list).

    Pass explicit config_ids whenever `configs` is not the full canonical
    universe (e.g. config_space.feasible_configs(instance)) -- for
    instance, if you materialize the universe once, keep each config's
    position in *that* list as its global id, and pass the matching
    subset of ids whenever you call this on a filtered subset. This is
    what lets results here stay joinable, by config_idx, with a
    scheduling.schedule_configs() call made over a *different* subset of
    the same universe (e.g. the partial-inversion heuristic, which
    schedules only the configs surviving a Phi filter computed here over
    the full universe).

    Neighbourhood (DIRECTIONAL -- configs reachable FROM y):

        N(y, tau) = { y' != y : T_rec(y -> y') <= instance.tau }

    T_rec is asymmetric (lambda != mu, kappa_add != kappa_rem), so
    y' in N(y,tau) does NOT imply y in N(y',tau); ordered pairs are
    evaluated in both directions separately.

    Flexibility:   Phi(y) = [Theta(y) + sum_{y' in N(y,tau)} Theta(y')]
                            / [|N(y,tau)| + 1]

    Complexity: O(n^2) ordered-pair comparisons in the worst case, but
    configs are first bucketed by their per-stage machine-type signature
    (the tuple of machine_id over stages). Reconfiguration can never
    change a stage's machine type, so only configs sharing a signature
    can ever be neighbours -- comparisons are confined within buckets,
    which is what keeps the directional O(n^2) tractable at ~4000
    configs.
    """
    if config_ids is None:
        config_ids = list(range(len(configs)))
    elif len(config_ids) != len(configs):
        raise ValueError("config_ids must have the same length as configs.")

    n_operations = instance.n_operations
    stage_ops = compute_stage_operations(instance)
    stage_demands = compute_stage_demands(instance, stage_ops)

    theta: Dict[int, float] = {
        cid: operational_capability(y, n_operations, stage_ops, stage_demands)
        for cid, y in zip(config_ids, configs)
    }

    neighbour_sets: Dict[int, List[int]] = {cid: [] for cid in config_ids}

    # Bucket by machine-type signature: only same-signature configs can
    # ever be mutually reachable (see same_machine_types).
    buckets: Dict[tuple, List[int]] = {}
    for pos, y in enumerate(configs):
        signature = tuple(mc.machine_id for mc, _ in y)
        buckets.setdefault(signature, []).append(pos)

    for positions in buckets.values():
        for i in positions:
            for j in positions:
                if i == j:
                    continue
                # Directional: i -> j only. The reverse direction is
                # evaluated separately when the loop reaches (j, i).
                if reconfiguration_time(configs[i], configs[j], instance) <= instance.tau:
                    neighbour_sets[config_ids[i]].append(config_ids[j])

    phi: Dict[int, float] = {}
    for cid in config_ids:
        neighbours = neighbour_sets[cid]
        theta_sum = theta[cid] + sum(theta[n] for n in neighbours)
        phi[cid] = theta_sum / (len(neighbours) + 1)

    return theta, phi, neighbour_sets