# Bi-level Configuration Design for Reconfigurable Manufacturing Systems

## Overview

This project implements a bi-level optimization framework for configuration design in Reconfigurable Manufacturing Systems (RMS). It combines exhaustive upper-level enumeration with a Hexaly lower-level flowshop solver to optimize machine configurations while considering flexibility and reconfiguration costs.

## Project Structure

### Files

- **`generate_instance.py`** – Random instance generator for RMS problems
  - Generates flowshop instances with multiple jobs, stages, and machine configurations
  - Supports customizable parameters (number of jobs, stages, configurations, operation capabilities)
  - Exports instances to JSON format

- **`rms_bilevel.py`** – Main bi-level optimization solver
  - Upper-level: Configuration design (machine-config assignment to stages)
  - Lower-level: Flowshop scheduling using Hexaly optimizer
  - Evaluates flexibility corridors and reconfiguration times
  - Optimizes makespan and setup costs

- **`instance_small.json`** – Sample problem instance
  - 4 jobs, 3 stages
  - Multiple machine configurations per stage with varying processing times
  - Example of the data format used by the solver

## Key Concepts

### Problem Model

- **Reconfigurable Manufacturing Systems (RMS)**: Systems where machines can be configured in different ways to perform various operations
- **Bi-level Optimization**: 
  - **Upper level**: Select which machine configuration to assign to each manufacturing stage
  - **Lower level**: Schedule jobs through stages, finding optimal flowshop schedules for given configurations
- **Flexibility**: Machine configs can handle operations beyond their required set (from neighboring stages)
- **Reconfiguration Time**: Cost of changing machine configurations, parameterized by module changes

### Data Model

- **Operations**: Global unique identifiers; each job requires specific operations at each stage (flowshop constraint)
- **Processing Times**: `p_{o,(i,k)}` – time for operation `o` on machine `i` in config `k`
- **Machine Configuration**: Set of modules that define capability; processing times dict specifies which operations are feasible
- **Flexible Capability**: A config can handle required operations plus extra operations from neighboring stages

### Parameters

- `lambda_coef`, `mu_coef`, `delta_coef` – Reconfiguration time coefficients (Eq. 6)
- `alpha` – Weight on lower-level objective in upper-level problem (Eq. 2)
- `epsilon` – Admissible degradation threshold for flexibility (Eq. 12)
- `tau` – Maximum reconfiguration time for the neighborhood (Eq. 7)
- `psi` – Power index for operational capability metric (Eq. 9)

## Usage

### Generate an Instance

```bash
python generate_instance.py [nb_jobs] [nb_stages] [nb_configs_per_stage] [seed] [output.json]
```

**Example:**
```bash
python generate_instance.py 5 4 3 42 my_instance.json
```

**Default parameters** (if not specified):
- `nb_jobs=4`, `nb_stages=3`, `nb_configs_per_stage=3`, `seed=42`, `output=instance.json`

### Solve an Instance

```bash
python rms_bilevel.py instance.json [time_limit_per_config]
```

**Example:**
```bash
python rms_bilevel.py instance_small.json 60
```

## Dependencies

- **Python 3.7+**
- **Hexaly Optimizer** – Install via pip:
  ```bash
  pip install hexaly
  ```

## Instance JSON Format

```json
{
  "nb_stages": 3,
  "lambda_coef": 3.0,
  "mu_coef": 2.0,
  "delta_coef": 1.0,
  "alpha": 1.0,
  "epsilon": 0.05,
  "tau": 8.0,
  "psi": 1.0,
  "jobs": [
    {"job_id": 0, "operations": [0, 3, 6]},
    ...
  ],
  "stages": [
    [
      {
        "machine_id": 0,
        "config_id": 0,
        "setup_cost": 10.0,
        "processing_times": {"0": 4.0, "1": 6.0, "2": 3.0},
        "modules": [0, 1, 2]
      },
      ...
    ],
    ...
  ]
}
```

## Notes

- Operation IDs are **global and unique** across all stages
- In the flowshop model, each stage `s` has a dedicated block of op_ids: `[s*nb_jobs .. (s+1)*nb_jobs - 1]`
- The same data structure supports future job-shop extensions (where operation ordering is job-specific)
- Processing times and capability are unified: a config can perform operation `o` iff `o` is a key in its `processing_times` dict

## References

- Bi-level optimization framework for RMS configuration design
- Flexibility corridors for robustness analysis
- Hexaly optimizer for lower-level flowshop scheduling
