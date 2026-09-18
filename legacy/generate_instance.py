"""
generate_instance.py
====================
Generate random RMS instances for rms_bilevel.py.

Usage:
    python generate_instance.py [nb_jobs] [nb_stages] [nb_configs_per_stage]
                                [seed] [output.json]
"""

import json
import random
import sys


def generate(
    nb_jobs: int = 4,
    nb_stages: int = 3,
    nb_configs_per_stage: int = 3,
    nb_machines_per_stage: int = 2,
    seed: int = 42,
    alpha: float = 1.0,
    epsilon: float = 0.15,
    tau: float = 8.0,
    psi: float = 1.0,
    lambda_coef: float = 3.0,
    mu_coef: float = 2.0,
    delta_coef: float = 1.0,
    pt_lo: int = 2,
    pt_hi: int = 10,
    cost_lo: int = 5,
    cost_hi: int = 20,
    nb_modules: int = 8,
    extra_ops_per_config: int = None,
    flexibility_cost_penalty: int = 2,
    flexibility_time_penalty: float = 0.5,
) -> dict:
    """
    Generate structured RMS instances.

    Properties:
    - guaranteed feasibility
    - nontrivial capability diversity
    - correlated module neighborhoods
    - flexibility incurs cost/performance penalties
    """
    rng = random.Random(seed)

    if extra_ops_per_config is None:
        extra_ops_per_config = nb_jobs

    # ------------------------------------------------------------------
    # Jobs
    # job j requires operation s*nb_jobs+j at stage s
    # ------------------------------------------------------------------
    jobs = [
        {
            "job_id": j,
            "operations": [s * nb_jobs + j for s in range(nb_stages)],
        }
        for j in range(nb_jobs)
    ]

    all_op_ids = list(range(nb_stages * nb_jobs))

    # ------------------------------------------------------------------
    # Build stages
    # ------------------------------------------------------------------
    stages = []
    machine_counter = 0

    for s in range(nb_stages):
        required_ops = [s * nb_jobs + j for j in range(nb_jobs)]
        other_ops = [op for op in all_op_ids if op not in required_ops]

        configs_this_stage = []

        # Create machine pool for this stage
        machine_ids = []
        for _ in range(nb_machines_per_stage):
            machine_ids.append(machine_counter)
            machine_counter += 1

        # Base module structure per machine
        machine_base_modules = {}
        for machine_id in machine_ids:
            machine_base_modules[machine_id] = set(
                rng.sample(
                    range(nb_modules),
                    rng.randint(2, max(2, nb_modules // 2))
                )
            )

        # EXACT number of configs per stage
        for ki in range(nb_configs_per_stage):
            machine_id = machine_ids[ki % nb_machines_per_stage]
            base_modules = machine_base_modules[machine_id]

            # ----------------------------------------------------------
            # Correlated module mutations
            # ----------------------------------------------------------
            modules = set(base_modules)

            n_mutations = rng.randint(0, 2)

            for _ in range(n_mutations):
                if rng.random() < 0.5 and len(modules) < nb_modules:
                    available = list(set(range(nb_modules)) - modules)
                    if available:
                        modules.add(rng.choice(available))
                elif len(modules) > 1:
                    modules.remove(rng.choice(list(modules)))

            modules = sorted(modules)

            # ----------------------------------------------------------
            # Capability generation
            # ----------------------------------------------------------
            nb_extra = rng.randint(
                0,
                min(extra_ops_per_config, len(other_ops))
            )

            extra_ops = rng.sample(other_ops, nb_extra) if nb_extra > 0 else []
            supported_ops = required_ops + extra_ops

            flex_level = len(extra_ops)

            # ----------------------------------------------------------
            # Processing times
            # More flexible => slightly slower on average
            # ----------------------------------------------------------
            pt = {}
            for op_id in supported_ops:
                base_time = rng.randint(pt_lo, pt_hi)
                adjusted_time = base_time + flexibility_time_penalty * flex_level
                pt[str(op_id)] = float(round(adjusted_time, 2))

            # ----------------------------------------------------------
            # Cost
            # More flexible => more expensive
            # ----------------------------------------------------------
            base_cost = rng.randint(cost_lo, cost_hi)
            setup_cost = float(
                base_cost + flexibility_cost_penalty * flex_level
            )

            configs_this_stage.append({
                "machine_id": machine_id,
                "config_id": ki,
                "setup_cost": setup_cost,
                "processing_times": pt,
                "modules": modules,
            })

        stages.append(configs_this_stage)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    total_configs = 1
    for stage in stages:
        total_configs *= len(stage)

    instance = {
        "nb_stages": nb_stages,
        "lambda_coef": lambda_coef,
        "mu_coef": mu_coef,
        "delta_coef": delta_coef,
        "alpha": alpha,
        "epsilon": epsilon,
        "tau": tau,
        "psi": psi,
        "jobs": jobs,
        "stages": stages,
        "_meta": {
            "nb_jobs": nb_jobs,
            "nb_stages": nb_stages,
            "nb_configs_per_stage": nb_configs_per_stage,
            "nb_machines_per_stage": nb_machines_per_stage,
            "total_system_configs": total_configs,
            "seed": seed,
        },
    }

    return instance


def main():
    args = sys.argv[1:]

    nb_jobs = int(args[0]) if len(args) > 0 else 4
    nb_stages = int(args[1]) if len(args) > 1 else 3
    nb_configs_per_stage = int(args[2]) if len(args) > 2 else 3
    seed = int(args[3]) if len(args) > 3 else 42
    output = args[4] if len(args) > 4 else "instance_gen.json"

    inst = generate(
        nb_jobs=nb_jobs,
        nb_stages=nb_stages,
        nb_configs_per_stage=nb_configs_per_stage,
        seed=seed,
    )

    with open(output, "w") as f:
        json.dump(inst, f, indent=2)

    meta = inst["_meta"]

    print(f"Generated instance -> {output}")
    print(
        f"Jobs={meta['nb_jobs']}  "
        f"Stages={meta['nb_stages']}  "
        f"Configs/stage={meta['nb_configs_per_stage']}  "
        f"Machines/stage={meta['nb_machines_per_stage']}"
    )
    print(f"Total system configurations: {meta['total_system_configs']}")
    print(f"Seed: {meta['seed']}")


if __name__ == "__main__":
    main()