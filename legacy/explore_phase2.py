"""
Phase-2 explorer for rms_bilevel_gurobi.py runs.

Purpose
-------
Recompute everything downstream of the saved Q(y) values:

    - total_cost(y)
    - C*
    - the (1 + epsilon) admissible corridor
    - machine versatility
    - surplus capacity
    - combined operational capability Theta(y)
    - neighbourhood N(y, tau)
    - flexibility Phi(y)

for NEW values of alpha / epsilon / tau / lambda / mu / delta,
WITHOUT calling Gurobi again.

Q(y) and C_config(y) are read directly from the run's JSON output.
The lower-level scheduling problem is never re-solved.

The lower-level makespan Q(y) is therefore treated as a fixed input:
changing alpha or epsilon changes the upper-level selection, but does not
change any scheduling result.

Why this script reloads the instance file
------------------------------------------
The run JSON's `y` field stores the machine/configuration identifiers,
capable operations and modules for traceability. The authoritative source
for a MachineConfig, especially its module set, is the instance file.

The instance is therefore reloaded and a
(stage, machine_id, config_id) -> MachineConfig lookup is built.

This also makes the script compatible with older run JSON files that may
not have stored all MachineConfig information.

Operational capability
-----------------------
The operational capability measure follows the report definition:

    theta_{i,k}
        = (sum_o phi_{i,o}^k - 1) / (|O| - 1)

with |O| = 20, hence a denominator of 19.

The "-1" removes the operation currently required at the assigned stage.
Thus theta measures capability beyond what is strictly required.

The combined measure is:

    Theta(y)
        = sum_s theta_{i_s,k_s}
        + sum_s sigma_s(y)

where

    sigma_s(y)
        = n_s r_{i_s,o_s}^{k_s} / D_{o_s} - 1

and

    r_{i_s,o_s}^{k_s} = 1 / p_{o_s,(i_s,k_s)}.

The versatility term is normalized by 19, while surplus capacity is not
normalized or capped.

The instances used with this script are constructed so that each stage
corresponds to one operation in the common job sequence. Jobs that do not
require an operation at a given stage use operation 0. The script therefore
recovers o_s as the unique non-zero operation appearing at stage s.

Neighbourhood and flexibility
-----------------------------
Two feasible configurations y and y' are neighbours when

    T_rec(y, y') <= tau.

The flexibility score is then:

    Phi(y) =
        [Theta(y) + sum_{y' in N(y,tau)} Theta(y')]
        / [|N(y,tau)| + 1].

Usage
-----
    python explore_phase2.py run1.json instance.json

Override any desired upper-level parameter:

    python explore_phase2.py run1.json instance.json \
        --alpha 5 \
        --epsilon 0.15 \
        --tau 3 \
        --lambda-coef 3.0 \
        --mu-coef 2.0 \
        --delta-coef 1.0 \
        --top 20

Any constant not passed on the command line defaults to the value saved
in run1.json's meta block. For older run JSON files where lambda/mu/delta
were not saved, the instance-file values are used as fallback.
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from legacy.rms_bilevel_gurobi import (
    load_instance_json,
    MachineConfig,
    RMSInstance,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The model contains 20 operations.
#
# theta_{i,k} = (number of capable operations - 1) / (20 - 1)
#
# The denominator is therefore intentionally 19.
N_OPERATIONS = 20
N_ADDITIONAL_OPERATIONS = N_OPERATIONS - 1


# ---------------------------------------------------------------------------
# Lightweight configuration representation
# ---------------------------------------------------------------------------

@dataclass
class ConfigEntry:
    """
    Lightweight representation of one feasible upper-level configuration
    loaded from the run JSON.

    The actual MachineConfig objects are reconstructed from the instance
    file using (stage, machine_id, config_id).
    """

    config_idx: int
    y_label: str

    # One MachineConfig per stage.
    mc_list: List[MachineConfig]

    # Number of deployed machines at each stage.
    n_list: List[int]

    # Values saved by the original Gurobi run.
    C_config: float
    Q: float
    perf: float

    # Solver diagnostics from the original run.
    solver_status: str
    mip_gap_pct: float


# ---------------------------------------------------------------------------
# Instance / configuration loading
# ---------------------------------------------------------------------------

def build_mc_lookup(
    instance: RMSInstance,
) -> Dict[Tuple[int, int, int], MachineConfig]:
    """
    Build a lookup:

        (stage, machine_id, config_id) -> MachineConfig

    from the authoritative instance file.
    """
    lookup = {}

    for s, configs in enumerate(instance.stage_configs):
        for mc in configs:
            lookup[(s, mc.machine_id, mc.config_id)] = mc

    return lookup


def load_configs(
    run_json_path: str,
    instance: RMSInstance,
) -> Tuple[List[ConfigEntry], dict]:
    """
    Load every configuration stored in the run JSON and resolve its
    MachineConfig objects against the instance file.

    Returns
    -------
    entries
        All feasible configurations stored in the run JSON.
    meta
        Metadata saved by the original run.
    """

    with open(run_json_path, encoding="utf-8") as f:
        run_data = json.load(f)

    mc_lookup = build_mc_lookup(instance)

    entries = []

    for c in run_data["configs"]:
        mc_list = []
        n_list = []

        for stage_entry in c["y"]:
            key = (
                stage_entry["stage"],
                stage_entry["machine_id"],
                stage_entry["config_id"],
            )

            if key not in mc_lookup:
                raise KeyError(
                    f"Config (stage={key[0]}, machine={key[1]}, "
                    f"config={key[2]}) from {run_json_path} "
                    f"was not found in the instance file. "
                    f"Are you sure this run JSON matches this instance?"
                )

            mc_list.append(mc_lookup[key])
            n_list.append(stage_entry["n_machines"])

        entries.append(
            ConfigEntry(
                config_idx=c["config_idx"],
                y_label=" | ".join(
                    f"s{s}->m{mc.machine_id}k{mc.config_id}x{n}"
                    for s, (mc, n) in enumerate(
                        zip(mc_list, n_list)
                    )
                ),
                mc_list=mc_list,
                n_list=n_list,
                C_config=c["C_config"],
                Q=c["Q"],
                perf=c["perf"],
                solver_status=c["solver_status"],
                mip_gap_pct=c["mip_gap_pct"],
            )
        )

    return entries, run_data["meta"]


# ---------------------------------------------------------------------------
# Cost recomputation
# ---------------------------------------------------------------------------

def recompute_total_cost(
    entries: List[ConfigEntry],
    mode: str,
    alpha: float,
) -> Dict[int, float]:
    """
    Recompute total_cost(y) for every configuration using a new alpha.

    The saved Q(y) values are treated as fixed.

    Unit mode:
        total_cost = C_config + alpha * Q

    Representative mode:
        total_cost = C_config + alpha / TP

    where TP = batch_size / Q was already stored as `perf` in the run JSON.
    """

    total_cost = {}

    for e in entries:
        if mode == "unit":
            total_cost[e.config_idx] = (
                e.C_config + alpha * e.Q
            )

        elif mode == "representative":
            tp = e.perf

            if tp > 0:
                total_cost[e.config_idx] = (
                    e.C_config + alpha / tp
                )
            else:
                total_cost[e.config_idx] = float("inf")

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                "Expected 'unit' or 'representative'."
            )

    return total_cost


# ---------------------------------------------------------------------------
# Reconfiguration time
# ---------------------------------------------------------------------------

def reconfiguration_time(
    e1: ConfigEntry,
    e2: ConfigEntry,
    lambda_coef: float,
    mu_coef: float,
    delta_coef: float,
) -> float:
    """
    Compute the reconfiguration time T_rec(e1, e2).

    For each stage:

        added    = |A_2 \\ A_1|
        removed  = |A_1 \\ A_2|
        adjusted = |A_1 intersection A_2|

    and:

        T_rec =
            lambda * added
          + mu     * removed
          + delta  * adjusted.
    """

    total = 0.0

    for mc1, mc2 in zip(e1.mc_list, e2.mc_list):
        modules_1 = set(mc1.modules)
        modules_2 = set(mc2.modules)

        added = len(modules_2 - modules_1)
        removed = len(modules_1 - modules_2)
        adjusted = len(modules_1 & modules_2)

        total += (
            lambda_coef * added
            + mu_coef * removed
            + delta_coef * adjusted
        )

    return total


# ---------------------------------------------------------------------------
# Machine versatility
# ---------------------------------------------------------------------------

def machine_versatility(mc: MachineConfig) -> float:
    """
    Compute theta_{i,k} for one machine-configuration pair.

        theta_{i,k}
            = (sum_o phi_{i,o}^k - 1) / (|O| - 1)

    The model contains 20 operations, hence the denominator is 19.

    The '-1' removes the operation currently required at the assigned
    stage, so the measure represents additional operational capability.

    Operation 0 is treated as a dummy/no-operation marker and is excluded
    from the number of real capable operations.
    """

    real_capable = sum(
        op != 0
        for op in mc.capable_ops
    )

    return max(real_capable - 1, 0) / N_ADDITIONAL_OPERATIONS


def machine_versatility_sum(e: ConfigEntry) -> float:
    """Sum theta_{i_s,k_s} over all stages of configuration e."""

    return sum(
        machine_versatility(mc)
        for mc in e.mc_list
    )


# ---------------------------------------------------------------------------
# Stage operation and demand
# ---------------------------------------------------------------------------

def stage_operation(
    instance: RMSInstance,
    stage: int,
) -> int:
    """
    Recover the unique operation o_s associated with stage s.

    Some stages may be dummy/no-op stages: all active jobs have operation 0
    there, meaning no real work is performed at this stage. In that case this
    function returns 0 and the caller should treat the stage as inactive.

    For genuine production stages, the active jobs must all share the same
    non-zero operation, which is the standard RMS convention used here.
    """

    operations = {
        job.operations[stage]
        for job in instance.jobs
        if job.demand > 0.0
        and job.operations[stage] != 0
    }

    if not operations:
        return 0

    if len(operations) != 1:
        raise ValueError(
            f"Stage {stage} does not have exactly one unique non-zero "
            f"operation. Found: {sorted(operations)}"
        )

    return next(iter(operations))


def stage_demand(
    instance: RMSInstance,
    stage: int,
    operation: int,
) -> float:
    """
    Compute D_{o_s}, the aggregate demand associated with operation o_s
    at stage s.

    If the stage is a dummy/no-op stage, operation 0 is returned and the
    demand is zero by definition.
    """

    if operation == 0:
        return 0.0

    return sum(
        job.demand
        for job in instance.jobs
        if job.demand > 0.0
        and job.operations[stage] == operation
    )


# ---------------------------------------------------------------------------
# Surplus capacity
# ---------------------------------------------------------------------------

def surplus_capacity(
    e: ConfigEntry,
    instance: RMSInstance,
) -> float:
    """
    Compute the total relative surplus capacity:

        sum_s sigma_s(y)

    where

        sigma_s(y)
            = n_s * r_{i_s,o_s}^{k_s} / D_{o_s} - 1

    and

        r_{i_s,o_s}^{k_s}
            = 1 / p_{o_s,(i_s,k_s)}.

    Dummy/no-op stages contribute zero because they have no real operation.
    """

    total = 0.0

    for s, (mc, n) in enumerate(
        zip(e.mc_list, e.n_list)
    ):
        o_s = stage_operation(instance, s)

        if o_s == 0:
            continue

        D_os = stage_demand(
            instance,
            s,
            o_s,
        )

        if D_os <= 0.0:
            continue

        if o_s not in mc.processing_times:
            raise ValueError(
                f"Configuration {e.config_idx}: machine configuration "
                f"(machine={mc.machine_id}, config={mc.config_id}) "
                f"cannot perform operation {o_s} at stage {s}."
            )

        p = mc.processing_times[o_s]

        if p <= 0.0:
            raise ValueError(
                f"Configuration {e.config_idx}: non-positive processing "
                f"time p={p} for operation {o_s} at stage {s}."
            )

        r = 1.0 / p
        sigma = n * r / D_os - 1.0

        if sigma < 0.0 and sigma > -1e-10:
            sigma = 0.0

        total += sigma

    return total


# ---------------------------------------------------------------------------
# Combined operational capability Theta
# ---------------------------------------------------------------------------

def operational_capability(
    e: ConfigEntry,
    instance: RMSInstance,
) -> float:
    """
    Compute the combined operational capability Theta(y):

        Theta(y)
            = sum_s theta_{i_s,k_s}
            + sum_s sigma_s(y).

    The versatility term is normalized by 19.

    The surplus-capacity term is deliberately not normalized or bounded.
    """

    versatility = machine_versatility_sum(e)

    surplus = surplus_capacity(
        e,
        instance,
    )

    return versatility + surplus


# ---------------------------------------------------------------------------
# Neighbourhood and flexibility
# ---------------------------------------------------------------------------

def compute_measures(
    entries: List[ConfigEntry],
    instance: RMSInstance,
    tau: float,
    lambda_coef: float,
    mu_coef: float,
    delta_coef: float,
):
    """
    Compute Theta(y), Phi(y), and neighbourhoods for every feasible
    configuration.

    Neighbourhood:

        N(y,tau)
            = { y' != y : T_rec(y,y') <= tau }

    Flexibility:

        Phi(y)
            = [Theta(y) + sum_{y' in N(y,tau)} Theta(y')]
              / [|N(y,tau)| + 1].
    """

    # --------------------------------------------------------------
    # Theta(y)
    # --------------------------------------------------------------
    theta = {
        e.config_idx: operational_capability(
            e,
            instance,
        )
        for e in entries
    }

    # --------------------------------------------------------------
    # Neighbourhoods
    #
    # This is O(n^2) over the feasible configurations.
    # For the current instances (~3000 configs), this is acceptable.
    # --------------------------------------------------------------
    neighbour_sets: Dict[int, List[int]] = {
        e.config_idx: []
        for e in entries
    }

    for i, e1 in enumerate(entries):
        for e2 in entries[i + 1:]:
            t_rec = reconfiguration_time(
                e1,
                e2,
                lambda_coef,
                mu_coef,
                delta_coef,
            )

            if t_rec <= tau:
                neighbour_sets[e1.config_idx].append(
                    e2.config_idx
                )
                neighbour_sets[e2.config_idx].append(
                    e1.config_idx
                )

    # --------------------------------------------------------------
    # Phi(y)
    # --------------------------------------------------------------
    phi = {}

    for e in entries:
        neighbours = neighbour_sets[e.config_idx]

        theta_sum = (
            theta[e.config_idx]
            + sum(theta[n] for n in neighbours)
        )

        phi[e.config_idx] = (
            theta_sum / (len(neighbours) + 1)
        )

    return theta, phi, neighbour_sets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "run_json",
        help="JSON output produced by rms_bilevel_gurobi.py",
    )

    parser.add_argument(
        "instance_file",
        help="Original RMS instance JSON",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override epsilon",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override neighbourhood threshold tau",
    )

    parser.add_argument(
        "--lambda-coef",
        type=float,
        default=None,
        dest="lambda_coef",
        help="Override lambda reconfiguration coefficient",
    )

    parser.add_argument(
        "--mu-coef",
        type=float,
        default=None,
        dest="mu_coef",
        help="Override mu reconfiguration coefficient",
    )

    parser.add_argument(
        "--delta-coef",
        type=float,
        default=None,
        dest="delta_coef",
        help="Override delta reconfiguration coefficient",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of corridor configurations to print, ranked by Phi",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load instance and saved run
    # ------------------------------------------------------------------

    instance = load_instance_json(
        args.instance_file
    )

    entries, meta = load_configs(
        args.run_json,
        instance,
    )

    if not entries:
        raise RuntimeError(
            "The run JSON contains no feasible configurations."
        )

    # ------------------------------------------------------------------
    # Recover parameters.
    #
    # Explicit command-line values have priority.
    # Otherwise use values saved in the original run.
    # For lambda/mu/delta, older JSON files may not contain the values,
    # so fall back to the instance.
    # ------------------------------------------------------------------

    if args.alpha is not None:
        alpha = args.alpha
    else:
        alpha = meta["alpha"]

    if args.epsilon is not None:
        epsilon = args.epsilon
    else:
        epsilon = meta["epsilon"]

    if args.tau is not None:
        tau = args.tau
    else:
        tau = meta["tau"]

    if args.lambda_coef is not None:
        lambda_coef = args.lambda_coef
    else:
        lambda_coef = meta.get(
            "lambda_coef",
            instance.lambda_coef,
        )

    if args.mu_coef is not None:
        mu_coef = args.mu_coef
    else:
        mu_coef = meta.get(
            "mu_coef",
            instance.mu_coef,
        )

    if args.delta_coef is not None:
        delta_coef = args.delta_coef
    else:
        delta_coef = meta.get(
            "delta_coef",
            instance.delta_coef,
        )

    mode = meta["mode"]

    # ------------------------------------------------------------------
    # Information about representative mode.
    #
    # It is not actually needed for recomputing total_cost because
    # `perf` (throughput) was already stored in the run JSON.
    # We nevertheless recover batch_size for informative output.
    # ------------------------------------------------------------------

    batch_size = None

    if mode == "representative":
        if meta.get("K") is None:
            raise ValueError(
                "Representative mode requires K in the run metadata."
            )

        active = [
            j
            for j in instance.jobs
            if j.demand > 0.0
        ]

        total_demand = sum(
            j.demand
            for j in active
        )

        batch_size = sum(
            math.ceil(
                meta["K"] * j.demand / total_demand
            )
            for j in active
        )

    elif mode != "unit":
        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Expected 'unit' or 'representative'."
        )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print(
        f"[Explore] Run: {args.run_json}  "
        f"Instance: {args.instance_file}"
    )

    print(
        f"[Explore] {len(entries)} feasible configs loaded  "
        f"mode={mode}"
    )

    if mode == "representative":
        print(
            f"[Explore] K={meta.get('K')}  "
            f"representative batch size={batch_size}"
        )

    print(
        f"[Explore] Using: "
        f"alpha={alpha}  "
        f"epsilon={epsilon}  "
        f"tau={tau}  "
        f"lambda={lambda_coef}  "
        f"mu={mu_coef}  "
        f"delta={delta_coef}"
    )

    if any(
        k not in meta
        for k in (
            "lambda_coef",
            "mu_coef",
            "delta_coef",
        )
    ):
        print(
            "[Explore] NOTE: this run.json predates saving "
            "lambda/mu/delta in meta. Values not explicitly passed "
            "on the command line are taken from the instance file."
        )

    print("-" * 64)

    # ------------------------------------------------------------------
    # Recompute total cost and admissible corridor
    #
    # Q(y) remains exactly the value obtained in the original run.
    # ------------------------------------------------------------------

    total_cost = recompute_total_cost(
        entries,
        mode,
        alpha,
    )

    C_star = min(
        total_cost.values()
    )

    threshold = (
        (1.0 + epsilon)
        * C_star
    )

    corridor = [
        e
        for e in entries
        if total_cost[e.config_idx] <= threshold
    ]

    print(
        "[Saved lower-level results] "
        "Q(y) and C_config(y) are read directly from the run JSON."
    )

    print(
        f"[Corridor] C* = {C_star:.4f}"
    )

    print(
        f"[Corridor] threshold = "
        f"(1+epsilon)*C* = {threshold:.4f}"
    )

    print(
        f"[Corridor] {len(corridor)} config(s) "
        f"in the corridor "
        f"(out of {len(entries)} feasible)"
    )

    # ------------------------------------------------------------------
    # Compute Theta / neighbourhood / Phi
    # ------------------------------------------------------------------

    theta, phi, neighbour_sets = compute_measures(
        entries,
        instance,
        tau,
        lambda_coef,
        mu_coef,
        delta_coef,
    )

    # ------------------------------------------------------------------
    # Rank only configurations in the admissible corridor.
    # ------------------------------------------------------------------

    corridor_ranked = sorted(
        corridor,
        key=lambda e: phi[e.config_idx],
        reverse=True,
    )

    winner = (
        corridor_ranked[0]
        if corridor_ranked
        else None
    )

    # ------------------------------------------------------------------
    # Report corridor
    # ------------------------------------------------------------------

    print("-" * 64)

    print(
        f"[Phase 2] Top "
        f"{min(args.top, len(corridor_ranked))} "
        f"corridor config(s) by Phi:"
    )

    for e in corridor_ranked[:args.top]:
        marker = (
            " <- BEST"
            if e is winner
            else ""
        )

        print(
            f"    [{e.config_idx:4d}] "
            f"{e.y_label}  "
            f"cost={total_cost[e.config_idx]:.4f}  "
            f"Q={e.Q:.3f}  "
            f"Theta={theta[e.config_idx]:.3f}  "
            f"Phi={phi[e.config_idx]:.3f}  "
            f"|N(y,tau)|="
            f"{len(neighbour_sets[e.config_idx])}"
            f"{marker}"
        )

    # ------------------------------------------------------------------
    # Detailed winner
    # ------------------------------------------------------------------

    if winner is not None:
        winner_idx = winner.config_idx

        versatility = machine_versatility_sum(
            winner
        )

        surplus = surplus_capacity(
            winner,
            instance,
        )

        print("-" * 64)

        print(
            "[Result] Phase-2 winner under these constants:"
        )

        print(
            f"    {winner.y_label}"
        )

        print(
            f"    config_idx={winner.config_idx}"
        )

        print(
            f"    status={winner.solver_status}  "
            f"gap={winner.mip_gap_pct}"
        )

        print(
            f"    C_config={winner.C_config:.4f}"
        )

        print(
            f"    Q={winner.Q:.4f}"
        )

        print(
            f"    total_cost="
            f"{total_cost[winner_idx]:.4f}"
        )

        print(
            f"    Machine versatility sum="
            f"{versatility:.4f}"
        )

        print(
            f"    Surplus capacity sum="
            f"{surplus:.4f}"
        )

        print(
            f"    Theta="
            f"{theta[winner_idx]:.4f}"
        )

        print(
            f"    Phi="
            f"{phi[winner_idx]:.4f}"
        )

        print(
            f"    |N(y,tau)|="
            f"{len(neighbour_sets[winner_idx])}"
        )

        print()

        print(
            "    NOTE: full schedule (activity/gantt) for this "
            "configuration is only available if it was inside "
            "the ORIGINAL run's corridor (saved in corridor_detail)."
        )

        print(
            "    If this winner differs from the original run's "
            "winner, or if it was not in the original corridor, "
            "its schedule was not saved and would require re-solving "
            "that configuration with Gurobi."
        )

    else:
        print("-" * 64)

        print(
            "[Result] The admissible corridor is empty."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()