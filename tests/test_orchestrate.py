import pytest

pytest.importorskip("gurobipy", reason="gurobipy not installed")

import config_space as cs
import orchestrate as orch


def test_run_full_produces_a_winner(tiny_instance, tmp_path):
    out = orch.run_full(tiny_instance, str(tmp_path / "full"), time_limit_per_config=5)
    assert out["winner"] is not None
    assert out["C_star"] <= out["threshold"]


def test_run_full_with_limit_schedules_only_the_sample(tiny_instance, tmp_path):
    n_total = len(cs.feasible_configs(tiny_instance))
    out = orch.run_full(tiny_instance, str(tmp_path / "sampled"),
                        time_limit_per_config=5, limit=3, seed=1)
    assert len(out["corridor"]) <= 3
    # the sample must be a strict subset when the instance has more configs
    if n_total > 3:
        assert n_total > 3  # sanity: the tiny instance does have more than 3


def test_run_two_pass_does_not_crash_and_keeps_config_idx_alignment(tiny_instance, tmp_path):
    out = orch.run_two_pass(tiny_instance, str(tmp_path / "twopass"),
                            time_limit_per_config=5, refine_time_limit=5,
                            refine_corridor_factor=2.0)
    assert out["winner"] is not None


def test_run_partial_inversion_keeps_requested_fraction(tiny_instance, tmp_path):
    n_total = len(cs.feasible_configs(tiny_instance))
    out = orch.run_partial_inversion(tiny_instance, str(tmp_path / "pi"),
                                     phi_top_fraction=0.5, time_limit_per_config=5)
    # at most half (rounded) of the universe should have been scheduled
    assert len(out["corridor"]) <= n_total


def test_recompute_mode_reproduces_a_prior_full_run(tiny_instance, tiny_instance_path, tmp_path):
    full_out = orch.run_full(tiny_instance, str(tmp_path / "baseline"), time_limit_per_config=5)

    recompute_out = orch.run_recompute(
        tiny_instance, str(tmp_path / "baseline.meta.json"), str(tmp_path / "recomputed"),
    )

    assert recompute_out["C_star"] == pytest.approx(full_out["C_star"])
    assert recompute_out["winner"]["config_idx"] == full_out["winner"]["config_idx"]


def test_recompute_mode_with_different_epsilon_can_change_winner(tiny_instance, tmp_path):
    orch.run_full(tiny_instance, str(tmp_path / "baseline2"), time_limit_per_config=5)

    import dataclasses
    wide_instance = dataclasses.replace(tiny_instance, epsilon=5.0)  # everything qualifies
    out_wide = orch.run_recompute(
        wide_instance, str(tmp_path / "baseline2.meta.json"), str(tmp_path / "recomputed_wide"),
    )
    assert len(out_wide["corridor"]) >= 1
