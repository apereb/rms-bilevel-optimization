import pytest

pytest.importorskip("gurobipy", reason="gurobipy not installed")

import config_space as cs
import scheduling as sch


def test_build_scheduled_jobs_requires_K():
    import dataclasses
    job = cs.Job(job_id=1, operations=[1], demand=1.0)
    instance = cs.RMSInstance(jobs=[job], nb_stages=1, stage_configs=[[]], K=None)
    with pytest.raises(ValueError):
        sch.build_scheduled_jobs(instance)


def test_schedule_configs_returns_one_entry_per_config(tiny_instance):
    configs = cs.feasible_configs(tiny_instance)[:3]
    results, run_info = sch.schedule_configs(
        tiny_instance, configs, time_limit_per_config=5, verbose=False,
    )
    assert len(results) == 3
    assert [r["config_idx"] for r in results] == [0, 1, 2]
    for r in results:
        assert r["Q"] > 0
        assert r["C_config"] == cs.config_setup_cost(configs[r["config_idx"]])
        assert "total_cost" not in r  # scheduling.py must not compute cost


def test_schedule_configs_respects_explicit_config_ids(tiny_instance):
    configs = cs.feasible_configs(tiny_instance)
    subset_ids = [5, 2, 9]
    subset = [configs[i] for i in subset_ids]
    results, _ = sch.schedule_configs(
        tiny_instance, subset, config_ids=subset_ids,
        time_limit_per_config=5, verbose=False,
    )
    assert [r["config_idx"] for r in results] == subset_ids


def test_json_run_writer_creates_missing_parent_dir(tmp_path):
    target = tmp_path / "nested" / "does" / "not" / "exist" / "run"
    writer = sch.JsonRunWriter(str(target), meta={"a": 1})
    writer.add_config_result({"config_idx": 0})
    writer.close()
    assert (tmp_path / "nested" / "does" / "not" / "exist").is_dir()


def test_run_logger_creates_missing_parent_dir(tmp_path):
    target = tmp_path / "nested2" / "run.log"
    log = sch.RunLogger(str(target))
    log.write("hello")
    log.close()
    assert target.exists()
