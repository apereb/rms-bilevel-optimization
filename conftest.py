"""
Shared fixtures for the RMS bi-level pipeline test suite.

`tiny_instance` / `tiny_instance_path` are the same synthetic 2-job,
2-stage instance used throughout development to validate the refactor
against the original scripts: small enough to solve instantly, but with
enough structure (heterogeneous processing times, module overlap between
MachineConfigs, a non-trivial tau) to exercise every code path.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_space as cs


def _build_tiny_instance_dict() -> dict:
    return {
        "jobs": [
            {"job_id": 1, "operations": [1, 2], "demand": 0.5},
            {"job_id": 2, "operations": [1, 2], "demand": 0.3},
        ],
        "nb_stages": 2,
        "stages": [
            [
                {"machine_id": 1, "config_id": 1, "setup_cost": 10.0,
                 "processing_times": {"1": 0.5, "2": 0.3}, "modules": [1, 2]},
                {"machine_id": 2, "config_id": 1, "setup_cost": 8.0,
                 "processing_times": {"1": 0.6, "2": 0.4}, "modules": [1, 3]},
            ],
            [
                {"machine_id": 1, "config_id": 1, "setup_cost": 10.0,
                 "processing_times": {"1": 0.5, "2": 0.3}, "modules": [1, 2]},
                {"machine_id": 1, "config_id": 2, "setup_cost": 12.0,
                 "processing_times": {"1": 0.55, "2": 0.2}, "modules": [2, 4]},
            ],
        ],
        "N_max": 2,
        "tau": 5.0,
        "K": 8,
        "alpha": 2.0,
        "epsilon": 0.1,
    }


@pytest.fixture
def tiny_instance_path(tmp_path) -> str:
    path = tmp_path / "tiny_instance.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_build_tiny_instance_dict(), f)
    return str(path)


@pytest.fixture
def tiny_instance(tiny_instance_path) -> cs.RMSInstance:
    return cs.load_instance_json(tiny_instance_path)
