"""
Regression tests against a real historical run, kept separate from the
synthetic-instance suite since they depend on data files that aren't part
of the repo.

To run these: set RMS_TEST_DATA_DIR to a folder containing
instance_period1.json and run1.json (or any instance/run pair -- adjust
the filenames below if yours differ), e.g.:

    RMS_TEST_DATA_DIR=/path/to/data pytest tests/test_real_data_regression.py

Skipped automatically (not failed) if the env var isn't set or the files
aren't found there.
"""

import json
import os

import pytest

DATA_DIR = os.environ.get("RMS_TEST_DATA_DIR")
INSTANCE_PATH = os.path.join(DATA_DIR, "instance_period1.json") if DATA_DIR else None
RUN_PATH = os.path.join(DATA_DIR, "run1.json") if DATA_DIR else None

pytestmark = pytest.mark.skipif(
    not DATA_DIR or not os.path.exists(INSTANCE_PATH) or not os.path.exists(RUN_PATH),
    reason="RMS_TEST_DATA_DIR not set or instance_period1.json/run1.json not found there",
)


@pytest.fixture(scope="module")
def real_instance():
    import config_space as cs
    return cs.load_instance_json(INSTANCE_PATH)


@pytest.fixture(scope="module")
def real_run():
    with open(RUN_PATH) as f:
        return json.load(f)


def test_config_setup_cost_matches_every_historical_entry(real_instance, real_run):
    import config_space as cs

    mc_lookup = {}
    for s, stage_configs in enumerate(real_instance.stage_configs):
        for mc in stage_configs:
            mc_lookup[(s, mc.machine_id, mc.config_id)] = mc

    for c in real_run["configs"]:
        mc_list, n_list = [], []
        for stage_entry in c["y"]:
            key = (stage_entry["stage"], stage_entry["machine_id"], stage_entry["config_id"])
            mc_list.append(mc_lookup[key])
            n_list.append(stage_entry["n_machines"])
        y = tuple(zip(mc_list, n_list))
        assert cs.config_setup_cost(y) == pytest.approx(c["C_config"])


def test_evaluation_matches_historical_summary(real_instance, real_run):
    import evaluation as ev

    sched_results = [
        {"config_idx": c["config_idx"], "y": None, "C_config": c["C_config"],
         "Q": c["Q"], "perf": c["perf"]}
        for c in real_run["configs"]
    ]
    total_cost = ev.compute_total_cost(sched_results, real_instance.alpha)
    C_star, threshold, corridor = ev.compute_corridor(sched_results, total_cost, real_instance.epsilon)

    old_summary = real_run["summary"]
    assert C_star == pytest.approx(old_summary["C_star"])
    assert threshold == pytest.approx(old_summary["threshold"])
    assert len(corridor) == old_summary["corridor_size"]

    old_ids = {e["config_idx"] for e in old_summary["corridor_detail"]}
    new_ids = {e["config_idx"] for e in corridor}
    assert old_ids == new_ids
