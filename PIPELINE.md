# RMS Bi-Level Optimization Pipeline — Documentation

This document describes a pipeline for solving a two-phase bi-level
optimization problem for reconfigurable manufacturing system (RMS)
configuration design: enumerate candidate machine configurations, solve a
lower-level hybrid-flowshop scheduling problem for each, select a
near-optimal-cost corridor, and pick the most flexible configuration
within that corridor.

It explains what the code does, how it is
structured, and how to use it.



## Table of contents

0. [Prerequisites](#0-prerequisites)
1. [Problem overview](#1-problem-overview)
2. [Architecture](#2-architecture)
3. [Data model and instance file format](#3-data-model-and-instance-file-format)
4. [The optimization model](#4-the-optimization-model)
5. [Core modules](#5-core-modules)
6. [Running a pipeline: `orchestrate.py`](#6-running-a-pipeline-orchestratepy)
7. [Run output format](#7-run-output-format)
8. [Analysis tools](#8-analysis-tools)
9. [Plotting](#9-plotting)
10. [Testing](#10-testing)
11. [Known limitations and methodological caveats](#11-known-limitations-and-methodological-caveats)
12. [Worked example](#12-worked-example)

---

## 0. Prerequisites

The pipeline requires the following environment and dependencies.

### Software requirements

- **Python**: Python 3.14.3 or later.
- **Gurobi**: Gurobi 13.0.2, with the corresponding Python package `gurobipy`.
- **Gurobi license**: a valid Gurobi license is required for the optimization-based modes.

### Python dependencies

In addition to the standard Python library, the project requires the following external packages:

- **Gurobi Python API** (`gurobipy`) 13.0.2
- **Matplotlib**, used by `plot_phase2.py` for plotting.

These packages must be installed in the Python environment used to run the project. They can be installed with:

    pip install gurobipy==13.0.2 matplotlib


### Verification

The Python environment and Gurobi installation can be tested with:

    python -c "import gurobipy as gp; print(gp.gurobi.version())"

A successful installation should print:

    (13, 0, 2)

A valid Gurobi license must also be available before running the optimization-based modes.

All commands in this documentation assume that they are executed from the project root directory.

## 1. Problem overview

Given an RMS instance (jobs, stages, candidate machine configurations at
each stage), the pipeline answers: which system configuration should we
deploy?

The decision happens in two phases:

- **Phase 1 (cost)**: among all feasible configurations, find the one(s)
  with the lowest total cost (setup cost plus a term penalizing poor
  throughput/makespan). Rather than keeping only the single cheapest
  configuration, an admissible corridor of near-optimal
  configurations is kept: all configurations within `(1+epsilon)` of the
  true minimum cost `C*`.
- **Phase 2 (flexibility)**: among the corridor, pick the configuration
  that maximizes a flexibility score `Phi`, which rewards configurations
  that are both individually capable (`Theta`) and sit in a
  neighbourhood of other capable configurations reachable at low
  reconfiguration effort. The idea: prefer a configuration that is not
  only cheap now, but easy to pivot away from later if conditions change.

Evaluating cost requires solving a real scheduling problem (a hybrid
flexible flow-shop, solved with Gurobi) for every candidate
configuration, this is the expensive part. Evaluating flexibility is
purely structural (depends only on the instance and the configuration's
properties, never on the scheduling result) and is comparatively free.
This asymmetry is central to how the pipeline is organized and to the
faster heuristics it offers (Section 6).

## 2. Architecture

Four modules form the foundation, each with a single responsibility and
no circular dependencies:

```
config_space.py   instance loading, data model, configuration-space
                   enumeration and feasibility. No Gurobi, no cost/
                   flexibility logic.

structural.py      Theta (operational capability) and Phi (flexibility),
                   and the reconfiguration-effort model that defines
                   neighbourhoods. Depends only on config_space.py.
                   Never touches Gurobi or cost.

scheduling.py      the Gurobi hybrid-flowshop solver and the loop that
                   runs it over a set of configurations. Depends only on
                   config_space.py. Knows nothing about cost weighting
                   (alpha) or flexibility.

evaluation.py      total cost, the admissible corridor, and Phase-2
                   selection. Joins scheduling.py's and structural.py's
                   outputs by a shared config index. Owns alpha/epsilon.
```

On top of these, four command-line tools:

```
orchestrate.py          runs a full pipeline end to end, in one of five
                        modes (Section 6).

heuristic_analysis.py   post-hoc analysis: how good is a candidate
                        heuristic setting, how sensitive is the result
                        to a parameter, where does the cost/makespan
                        trade-off frontier sit (Section 8).

compare_runs.py         a quick side-by-side summary table across
                        several runs (Section 8).

plot_phase2.py          the three standard figures (cost-vs-flexibility
                        scatter, capability breakdown, neighbourhood-size
                        histogram) for a given run (Section 9).
```

A key design choice threading through the whole pipeline: every module
that computes `Theta`/`Phi` does so over the full feasible universe of
configurations, never over just the subset a particular run happened to
schedule. `Phi` depends on a configuration's neighbourhood, and a
partial run (e.g. `partial-inversion`, Section 6) only ever schedules a
fraction of the universe: computing `Phi` from that fraction alone would
yield a different value than the baseline. Every tool that reloads a run's results
(`orchestrate.py recompute`, `compare_runs.py`, `heuristic_analysis.py`)
reconstructs the full universe via `config_space.feasible_configs()`
before computing `Theta`/`Phi`.

## 3. Data model and instance file format

Three dataclasses in `config_space.py`:

```python
@dataclass
class Job:
    job_id: int
    operations: List[int]   # one operation id per stage; 0 = no-op at that stage (in practice dummy 0-time operation)
    demand: float = 0.0     # demand rate; 0 = job inactive for this instance

@dataclass
class MachineConfig:
    machine_id: int          # machine TYPE (basic modules, not swappable)
    config_id:  int          # machine CONFIGURATION (auxiliary modules, swappable)
    setup_cost: float
    processing_times: Dict[int, float]   # op_id -> processing time
    modules: List[int]                    # AUXILIARY modules only (see Section 4.3)

@dataclass
class RMSInstance:
    jobs: List[Job]
    nb_stages: int
    stage_configs: List[List[MachineConfig]]   # stage_configs[s] = options at stage s
    N_max: int = 1        # global upper bound on machines per stage
    K: int = None         # representative batch size
    lambda_coef, mu_coef, delta_coef: float    # retooling coefficients
    kappa_add, kappa_rem: float                # install/decommission coefficients
    alpha, epsilon, tau: float                 # bi-level parameters
    psi: float             # unused legacy field, kept only for old instance files' compatibility; no current formula reads it
```

A `SystemConfig` (the decision variable `y`) is a tuple of
`(MachineConfig, n)` pairs, one per stage, where `n` is the number of
identical parallel machines deployed.

### Instance JSON schema

```json
{
  "nb_stages": 5,
  "N_max": 3,
  "K": 20,
  "alpha": 5,
  "epsilon": 0.15,
  "tau": 3,
  "lambda_coef": 0.5,
  "mu_coef": 0.4,
  "delta_coef": 0.1,
  "kappa_add": 1.5,
  "kappa_rem": 1.0,
  "jobs": [
    {"job_id": 0, "demand": 0.333333, "operations": [2, 12, 17, 0, 0]}
  ],
  "stages": [
    [
      {"machine_id": 1, "config_id": 1, "setup_cost": 10.0,
       "processing_times": {"2": 0.5, "12": 0.3}, "modules": [10, 11]}
    ]
  ]
}
```

Notes:

- `demand` and `processing_times` must share the same unit of time
  (e.g. both in minutes), nothing in the code converts between them;
  they are multiplied directly wherever capacity is checked.
- `kappa_add`/`kappa_rem` default to `1.5`/`1.0` if omitted, so older
  instance files without them still load; add them explicitly to new
  instance files for clarity and to override the defaults.
- `modules` lists only auxiliary modules (those defining
  `config_id`). Principal modules (those defining `machine_id`) are
  never listed and are not swappable (see Section 4.3).

## 4. The optimization model

### 4.1 Cost and corridor

```
total_cost(y) = C_config(y) + alpha / perf(y)
```

`C_config(y)` is the setup cost (sum of `n * setup_cost` over stages);
`perf(y)` is the representative throughput (`batch_size / makespan`).
`C*` is the minimum `total_cost` over the evaluated configurations; the
admissible corridor is every configuration with
`total_cost(y) <= (1+epsilon) * C*`.

### 4.2 Operational capability `Theta` and flexibility `Phi`

```
Theta(y) = versatility(y) + surplus_capacity(y)
```

- **Versatility**: for each stage, `(real_capable_ops - 1) / (|O| - 1)`,
  where `|O|` is the full operation catalogue of the instance (the union
  of every machine's capable operations, not just those a job currently
  requires). Captures capability beyond what's strictly needed: the
  alternatives available if the nature of demand changed.
- **Surplus capacity**: `n * r / D - 1` per stage, where `r = 1/p` is the
  machines processing rate (computed from its processing time `p`) and `D` the aggregate demand. Captures
  adaptability to changes in demand quantity. Not normalized.

```
Phi(y) = [Theta(y) + sum_{y' in N(y,tau)} Theta(y')] / [|N(y,tau)| + 1]
```

`N(y,tau)` is `y`'s reconfiguration neighbourhood (below): the
configurations reachable from `y` at a reconfiguration effort at most
`tau`. `Phi` is the average capability of `y` and everything it can
cheaply pivot to.

### 4.3 Reconfiguration effort and neighbourhoods

A `MachineConfig` is defined by two kinds of modules: basic
modules, which determine the machine type (`machine_id`), and
auxiliary modules, which determine the configuration
(`config_id`). Only auxiliary modules are represented in the data
(`MachineConfig.modules`); principal modules are not swappable. As a
consequence: two configurations can only be reconfiguration-reachable
from one another if they use the same `machine_id` at every stage (that is not a computational limitation, but a modeling choice). If
any stage's machine type differs, `T_rec = +infinity`.

Given that constraint, the reconfiguration effort from `y` to `y'` is a
sum over stages of two independent terms:

```
T_rec(y -> y') = sum_s [ T_retool,s(y,y') + T_count,s(y,y') ]
```

**Retooling** (module-level; only applies when the machine config
changes at a stage, i.e. `(machine_id, config_id)` differs; a stage
where nothing changes contributes 0, not `delta * |modules|`):

```
T_retool,s = 1[(i_s,k_s) != (i_s',k_s')] * min(n_s, n_s') *
             [ lambda*A_s + mu*R_s + delta*J_s ]
```

`A_s`/`R_s`/`J_s` are the counts of modules added / removed / shared
between the two configurations' module sets. The `min(n_s, n_s')` factor
scales retooling to only the units that survive the transition: added
units arrive already qualified for the target configuration (that's
what `kappa_add` pays for), and decommissioned units are not retooled
first.

**Install/decommission** (fleet-size level; applies whenever the unit
count changes, independently of any configuration change):

```
T_count,s = kappa_add * max(n_s' - n_s, 0) + kappa_rem * max(n_s - n_s', 0)
```

`T_rec` is directional (asymmetric): `lambda != mu` and
`kappa_add != kappa_rem` in general, so the effort to go from `y` to
`y'` differs from the reverse. Neighbourhoods are therefore directional
too: `y' in N(y,tau)` does not imply `y in N(y',tau)`.

`structural.compute_measures` groups configurations by their per-stage
machine-type signature before comparing pairs. Since only
same-signature configurations can ever be mutually reachable, this
confines the (still directional, still pairwise) comparison to within
each group, which is what keeps it tractable at several thousand
configurations.

## 5. Core modules

### `config_space.py`

- `load_instance_json(path)`  parses the instance JSON into an
  `RMSInstance`.
- `feasible_configs(instance)`  builds the set of feasible `SystemConfig`s for an instance. Every downstream module
  should use this rather than re-deriving the config space.
- `enumerate_configs` / `validate_config` / `count_raw_configs` /
  `count_feasible_configs`  the enumeration and feasibility-filtering
  machinery `feasible_configs` is built from.
- `config_setup_cost`, `build_pt_matrix`, `config_label`,
  `config_to_dict`  small helpers used throughout the pipeline for
  cost computation, building the scheduler's input matrix, and
  human/JSON-friendly representations of a `SystemConfig`.

### `structural.py`

- `compute_measures(instance, configs, config_ids=None)`  Returns `(theta, phi, neighbour_sets)`, each a dict keyed
  by `config_ids[i]` (defaults to position in `configs` if not given).
  Pass explicit `config_ids` whenever `configs` is not the full universe (e.g. a filtered subset), so results stay joinable
  by a common id with whatever else was computed over the true universe
  or a different subset.
- `reconfiguration_time(y1, y2, instance)`  Section 4.3's formula.
- `same_machine_types(y1, y2)`  the machine-type reachability check.
- `machine_versatility_sum`, `surplus_capacity`, `operational_capability`
   the `Theta` components, exposed separately because `plot_phase2.py`
  needs the breakdown for its capability-composition figure.

### `scheduling.py`

- `schedule_configs(instance, configs, config_ids=None,
  time_limit_per_config=5, logger=None, json_writer=None)`  runs the
  Gurobi hybrid-flowshop solver over every config in `configs`, in
  order. Returns `(results, run_info)`; each result has `config_idx`,
  `y`, `C_config`, `Q` (makespan), `perf`, `activity`, `gantt`,
  `solve_info`  but no `total_cost` (that needs `alpha`, which this
  module never sees).
- `solve_flowshop_gurobi`  the actual MIP (classic disjunctive
  formulation for hybrid/parallel-machine flow shops).
- `RunLogger` / `JsonRunWriter`  human-readable logging and
  crash-safe incremental persistence (Section 7).
- `build_scheduled_jobs(instance)`  builds the representative job batch
  (`K` copies apportioned by demand share) that actually gets scheduled.

### `evaluation.py`

- `compute_total_cost(scheduling_results, alpha)`,
  `compute_corridor(scheduling_results, total_cost, epsilon)`,
  `select_phase2_winner(corridor_entries, phi)`  the three cost/
  selection primitives, usable independently.
- `evaluate(instance, scheduling_results, theta, phi)`  the full
  Phase-1 + Phase-2 pass in one call; returns a dict with
  `total_cost`, `C_star`, `threshold`, `corridor` (list of merged
  entries, each gaining `total_cost`/`Theta`/`Phi`), and `winner`.
- `refinement_subset(...)`  the cost-threshold subset used by
  `two-pass`'s refinement step (Section 6).

## 6. Running a pipeline: `orchestrate.py`

```
python orchestrate.py <mode> <instance.json> [options]
```

### `full`

The baseline: schedule every feasible configuration once, at a fixed
Gurobi time limit.

```
python orchestrate.py full instance.json --time-limit 15 --output runs/baseline
```

Before committing to a run that can take hours, `--limit N` schedules
only a random sample of `N` configurations and projects the full run's
duration from the measured per-config average:

```
python orchestrate.py full instance.json --time-limit 15 --limit 30 --output runs/timing_test
```

### `two-pass`

A cheap first pass (small `--time-limit`) over everything, then a
second, more thorough pass (`--refine-time-limit`) only on the subset
whose pass-1 cost falls within `(1 + refine-corridor-factor * epsilon)`
of the pass-1 provisional best. Both passes are genuine new Gurobi
solves, nothing is reused from any prior run.

```
python orchestrate.py two-pass instance.json --time-limit 5 --refine-time-limit 15 --refine-corridor-factor 1.5 --output runs/twopass
```

`heuristic_analysis.py twopass-preview` (Section 8) estimates the
refinement subset size and projected total wall time from an existing
`full` run's cost data (useful for picking a sane `--refine-corridor-factor`).

### `partial-inversion`

Computes `Phi` over the full universe first (cheap, no Gurobi), keeps
only the top `--phi-top-fraction` of configurations by `Phi`, and
schedules only those.

```
python orchestrate.py partial-inversion instance.json --time-limit 15 --phi-top-fraction 0.3 --output runs/partial
```

This is a heuristic with a real, instance-dependent risk: the true
cost-optimal configuration can have low `Phi` and be excluded outright.
Use `heuristic_analysis.py phi-recall` (Section 8) against an existing
`full` run to check how much of the true
corridor a given fraction would actually retain.

### `simulate-partial-inversion`

Replicates `partial-inversion`'s result without calling Gurobi:
reuses an existing `full` run's own already-solved `Q` values for
whichever configurations fall in the top fraction by `Phi`.

```
python orchestrate.py simulate-partial-inversion instance.json --baseline-run runs/baseline.meta.json --phi-top-fraction 0.3 --output runs/partial_sim
```

This is an approximation, not a substitute for a real run: reusing
`Q` values ignores the small solver-to-solver stochasticity a genuine
re-solve would show (observed empirically: real re-solves agree with
the baseline at Spearman ~0.999, this simulation reproduces it at
exactly 1.000). Useful for fast, free exploration of several candidate
fractions.

### `recompute`

Reloads a previous run's results and recomputes cost/corridor/selection
under different `alpha`/`epsilon`/`tau`/`lambda`/`mu`/`delta` No
Gurobi call, since none of these parameters affect the already-solved
`Q` values.

```
python orchestrate.py recompute instance.json --run-json runs/baseline.meta.json --epsilon 0.25 --output runs/baseline_eps025
```

Works on any run file `orchestrate.py` can load: one it produced
itself, or a legacy single-file run from an earlier version of this
codebase (format auto-detected).

## 7. Run output format

Every mode except `recompute`'s bare evaluation produces two files at
`<output_base>`:

- **`<output_base>.log`**  human-readable progress log, one line per
  scheduled configuration plus a final summary block.
- **`<output_base>.meta.json`**  small JSON with `meta` (the run's
  settings: mode, alpha/epsilon/tau/..., time limits) and, once the run
  finishes, `summary` (`C_star`, `threshold`, `corridor_size`,
  `phase2_best_config_idx`, `phase2_best_y`). Also has a `configs_file`
  key pointing to the sidecar below.
- **`<output_base>.configs.jsonl`**  one JSON object per line, one line
  per scheduled configuration, with the full detail (`config_idx`, `y`,
  `C_config`, `Q`, `perf`, solver diagnostics, `activity`, `gantt`).
  Append-only and flushed after every line: crash-safe (every
  already-written line stays valid even if the process is killed
  mid-run).

`orchestrate.load_scheduling_results(path, instance)` reloads either
this format or a legacy single-file run (auto-detected by the presence
of a `configs` list vs. a `configs_file` pointer), reconstructing each
entry's `SystemConfig` against the instance file. It deduplicates by
`config_idx`, keeping the last-written entry necessary because
`two-pass` legitimately writes some configs twice (once in pass 1, again
if refined in pass 2) to the same sidecar file.

## 8. Analysis tools

### `heuristic_analysis.py`

Five sub-commands, each `python heuristic_analysis.py <analysis> instance.json <run(s)> [options]`:

- **`phi-recall`**  needs only a `full` (or `two-pass`) run. For a swept
  list of `--fractions`, reports what share of the true corridor and
  whether the true winner would survive keeping only that fraction of
  configurations by `Phi`. Answers "how safe is a given
  `phi-top-fraction` for `partial-inversion`?" without spending any
  Gurobi time.
- **`compare`**  needs two actually executed runs (can't be
  simulated). Reports corridor Jaccard overlap, Spearman/Kendall rank
  correlation of `total_cost` restricted to configurations both runs
  scheduled, whether the first run's winner survives in the second's
  corridor, and the second run's winner's cost gap vs. the first run's
  `C*`.
- **`twopass-preview`**  needs only a `full` run. For a swept list of
  `--refine-corridor-factors`, reports the resulting refinement subset
  size/proportion and (with `--refine-time-limit`) a projected total
  wall time for a real `two-pass` run.
- **`alpha-sweep`**  needs only a `full` run. Recomputes `C*`/corridor/
  winner for each `--alphas` value (no Gurobi). Also tracks, against a
  `--reference-alpha` (default: the instance's own recorded alpha), how
  much the top-`--top-k` set by cost changes (Jaccard) and whether its
  internal ranking holds up (Spearman/Kendall). This catches a subtler
  problem than "does the winner change": the winner can look stable
  across a range of alpha while the rest of the ranking underneath
  reshuffles substantially.
- **`pareto`**  needs only a `full` run. Computes the Pareto frontier
  of `(C_config, Q)` (an `O(n log n)` skyline algorithm) and, with
  `--output-png` and `--alphas`, plots it with each swept alpha's winner
  overlaid (requires `matplotlib`).

### `compare_runs.py`

A simpler, faster overview across any number of runs at once:

```
python compare_runs.py instance.json baseline=runs/baseline.meta.json partial=runs/partial.meta.json twopass=runs/twopass.meta.json
```

Prints one row per run (`C*`, corridor size, winner, winner cost/`Phi`,
wall time). By default each run is evaluated under its own recorded
`alpha`/`epsilon`/`tau`; pass `--alpha`/`--epsilon`/`--tau` to force a
common value across all of them instead. Less detailed than `heuristic_analysis.py compare` (no rank
correlation), but handles many runs in one table.

## 9. Plotting

```
python plot_phase2.py <run.meta.json> instance.json --top 20 --outdir figures
```

Produces three figures from a single run, reusing the same
`evaluation.py`/`structural.py` logic as the CLI tools (never
reimplemented):

1. **`fig1_cost_vs_phi.png`**  scatter of `total_cost` vs. `Phi` for
   every configuration the run scheduled; corridor highlighted, `C*` and
   `(1+epsilon)C*` threshold lines drawn, winner starred.
2. **`fig2_top_corridor_theta.png`**  horizontal bar chart of the top
   `--top` corridor configurations by `Phi`, with `Theta` split into its
   versatility and surplus-capacity components.
3. **`fig3_neighbourhood_hist.png`**  histogram of neighbourhood sizes
   `|N(y,tau)|` over the corridor.

Same `--alpha`/`--epsilon`/`--tau`/`--lambda-coef`/`--mu-coef`/
`--delta-coef` override flags as `orchestrate.py recompute`, for
what-if figures without any new Gurobi call. Works on any run format
`orchestrate.load_scheduling_results` can load.

## 10. Testing

```
pytest tests/
```

Covers `config_space.py`, `structural.py` (including the reconfiguration
model's type-change guard, install/decommission term, asymmetry, and
machine-type restriction), `evaluation.py`, `scheduling.py`,
`orchestrate.py`'s four modes, and `heuristic_analysis.py`'s rank
correlation primitives and analyses on a small, fully self-contained
synthetic instance defined in `tests/conftest.py`. Tests requiring
`gurobipy` skip cleanly (`pytest.importorskip`) if it is not installed;
everything else runs regardless.

An optional regression suite (`tests/test_real_data_regression.py`)
checks `config_space.py`/`evaluation.py` against a real historical run,
activated via an environment variable rather than a hardcoded path:

```
RMS_TEST_DATA_DIR=/path/to/data pytest tests/test_real_data_regression.py
```

## 11. Known limitations and methodological caveats

- **`partial-inversion`'s reliability is instance-dependent.** On some
  instances it agrees closely with the full baseline; on others, the
  true cost-optimal configuration has low `Phi` and gets excluded
  outright, so the reported winner is not merely a worse approximation
  of the same choice but potentially a genuinely different one. Always
  check `phi-recall` against a real `full` run before trusting a
  `partial-inversion` result on a new instance, and do not assume a
  fraction that worked well on one instance will work equally well on
  another.
- **The selected winner is not, in general, Pareto-optimal** on
  `(C_config, Q)`. This is expected, not a bug: Phase 2 selects by
  `argmax Phi` within the cost corridor, not by joint cost/throughput
  optimality. A reader comparing the winner against a Pareto-frontier
  plot should not be surprised to find it strictly dominated on that
  2D projection.
- **The result can be sensitive to `alpha`.** Observed on real
  instances: the winner can change at nearly every tested `alpha` value
  around the one actually used, and the wider ranking beneath a
  seemingly stable winner can still reshuffle substantially
  (`alpha-sweep`'s top-K stability columns exist specifically to catch
  this). Do not treat a single `alpha` value's result as self-evidently
  robust without checking a sweep.
- **`kappa_add`/`kappa_rem` are currently defaulted, not empirically
  calibrated.** They were derived by analogy from `lambda`/`mu` under an
  assumption about how many modules a typical machine holds. Revisit if
  either more principled reasoning or real data becomes available.

## 12. Worked example

Starting from nothing but an instance file:

```bash
# 1. Time a small sample before committing to a full run
python orchestrate.py full instance.json --time-limit 15 --limit 30 --output runs/timing_test

# 2. Run the baseline
python orchestrate.py full instance.json --time-limit 15 --output runs/baseline

# 3. Check how safe a partial-inversion fraction would be, for free
python heuristic_analysis.py phi-recall instance.json runs/baseline.meta.json --fractions 0.1,0.2,0.3,0.5

# 4. Run the heuristics
python orchestrate.py partial-inversion instance.json --time-limit 15 --phi-top-fraction 0.3 --output runs/partial
python orchestrate.py two-pass instance.json --time-limit 5 --refine-time-limit 15 --output runs/twopass

# 5. Compare
python heuristic_analysis.py compare instance.json runs/baseline.meta.json runs/partial.meta.json
python heuristic_analysis.py compare instance.json runs/baseline.meta.json runs/twopass.meta.json
python compare_runs.py instance.json baseline=runs/baseline.meta.json partial=runs/partial.meta.json twopass=runs/twopass.meta.json

# 6. Sensitivity and trade-off analysis
python heuristic_analysis.py alpha-sweep instance.json runs/baseline.meta.json --alphas 1,2,3,4,5,6,7,8,10,15,20 --top-k 20
python heuristic_analysis.py pareto instance.json runs/baseline.meta.json --alphas 1,5,8,20 --output-png pareto.png

# 7. Figures
python plot_phase2.py runs/baseline.meta.json instance.json --top 20 --outdir figures
```