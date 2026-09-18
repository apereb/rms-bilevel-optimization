import pytest

import heuristic_analysis as ha


def test_spearman_perfect_agreement():
    assert ha.spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_disagreement():
    assert ha.spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_kendall_perfect_agreement():
    assert ha.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_kendall_perfect_disagreement():
    assert ha.kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_jaccard_identical_sets():
    assert ha.jaccard({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_disjoint_sets():
    assert ha.jaccard({1, 2}, {3, 4}) == 0.0


def test_jaccard_both_empty():
    assert ha.jaccard(set(), set()) == 1.0


def test_phi_recall_at_100_percent_recovers_full_corridor(tiny_instance, tmp_path):
    pytest.importorskip("gurobipy")
    import orchestrate as orch

    orch.run_full(tiny_instance, str(tmp_path / "baseline"), time_limit_per_config=5)
    results = orch.load_scheduling_results(str(tmp_path / "baseline.meta.json"), tiny_instance)

    rows, context = ha.phi_recall_curve(tiny_instance, results, fractions=[1.0])
    assert rows[0]["corridor_recall"] == pytest.approx(1.0)
    assert rows[0]["winner_included"] is True


def test_compare_identical_run_against_itself_is_perfect(tiny_instance, tmp_path):
    pytest.importorskip("gurobipy")
    import orchestrate as orch

    orch.run_full(tiny_instance, str(tmp_path / "baseline"), time_limit_per_config=5)
    results = orch.load_scheduling_results(str(tmp_path / "baseline.meta.json"), tiny_instance)

    result = ha.compare_baseline_vs_heuristic(tiny_instance, results, results)
    assert result["corridor_jaccard"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["baseline_winner_survives_in_heuristic_corridor"] is True
    assert result["baseline_winner_idx"] == result["heuristic_winner_idx"]
    # NOTE: winner_cost_gap_pct need NOT be 0 here, even against itself --
    # the Phase-2 winner is chosen by argmax Phi within the corridor, not
    # by minimum cost, so its own total_cost is generally > C_star. This
    # gap is a property of the selection rule, not of any discrepancy
    # between the two runs being compared.
    assert result["winner_cost_gap_pct"] >= 0


def test_compare_uses_full_universe_not_local_subset(tiny_instance, tmp_path, monkeypatch):
    """
    Regression test for a real bug: Phi is neighbourhood-dependent, so
    computing it from a run's own (possibly subset) config list gives a
    different value than the full-universe Phi orchestrate.py used
    internally to pick that run's winner. compare_baseline_vs_heuristic
    must always call structural.compute_measures with the FULL feasible
    universe, never with baseline_results' or heuristic_results' own
    (possibly truncated) local config list.
    """
    pytest.importorskip("gurobipy")
    import orchestrate as orch
    import config_space as cs

    orch.run_full(tiny_instance, str(tmp_path / "baseline"), time_limit_per_config=5)
    full_results = orch.load_scheduling_results(str(tmp_path / "baseline.meta.json"), tiny_instance)
    subset_results = full_results[: max(1, len(full_results) // 2)]

    universe_size = len(cs.feasible_configs(tiny_instance))
    assert len(subset_results) < universe_size  # precondition: subset is genuinely smaller

    call_sizes = []
    original = ha.st.compute_measures

    def spy(instance, configs, config_ids=None):
        call_sizes.append(len(configs))
        return original(instance, configs, config_ids) if config_ids is not None else original(instance, configs)

    monkeypatch.setattr(ha.st, "compute_measures", spy)

    ha.compare_baseline_vs_heuristic(tiny_instance, full_results, subset_results)

    assert call_sizes, "compute_measures was never called"
    assert all(n == universe_size for n in call_sizes), (
        f"compute_measures was called with config-list sizes {call_sizes}, "
        f"expected the full universe size ({universe_size}) every time -- "
        f"Phi must never be recomputed from a run's own local subset"
    )


def test_pareto_frontier_is_mutually_non_dominated():
    results = [
        {"config_idx": 0, "C_config": 10.0, "Q": 5.0},
        {"config_idx": 1, "C_config": 8.0, "Q": 6.0},
        {"config_idx": 2, "C_config": 12.0, "Q": 4.0},
        {"config_idx": 3, "C_config": 20.0, "Q": 20.0},  # dominated by everything
        {"config_idx": 4, "C_config": 8.0, "Q": 6.0},    # tie with idx 1, both non-dominated
    ]
    frontier = ha.pareto_frontier(results)
    frontier_ids = {idx for idx, _, _ in frontier}
    assert 3 not in frontier_ids  # strictly dominated, must be excluded
    # no frontier point should be dominated by another frontier point
    for i, (idx_a, c_a, q_a) in enumerate(frontier):
        for idx_b, c_b, q_b in frontier:
            if idx_a == idx_b:
                continue
            dominated = c_b <= c_a and q_b <= q_a and (c_b < c_a or q_b < q_a)
            assert not dominated, f"{idx_a} is dominated by {idx_b} but both are on the frontier"


def test_pareto_frontier_sorted_by_c_config():
    results = [
        {"config_idx": 0, "C_config": 10.0, "Q": 5.0},
        {"config_idx": 1, "C_config": 3.0, "Q": 9.0},
        {"config_idx": 2, "C_config": 6.0, "Q": 7.0},
    ]
    frontier = ha.pareto_frontier(results)
    c_values = [c for _, c, _ in frontier]
    assert c_values == sorted(c_values)


def test_alpha_sweep_topk_is_self_consistent_at_reference_alpha(tiny_instance, tmp_path):
    pytest.importorskip("gurobipy")
    import orchestrate as orch

    orch.run_full(tiny_instance, str(tmp_path / "baseline"), time_limit_per_config=5)
    results = orch.load_scheduling_results(str(tmp_path / "baseline.meta.json"), tiny_instance)

    rows, context = ha.alpha_sweep(tiny_instance, results,
                                   alphas=[tiny_instance.alpha], top_k=5,
                                   reference_alpha=tiny_instance.alpha)
    # at the reference alpha itself, comparing the topK set to itself
    # must be perfect agreement
    assert rows[0]["topk_jaccard_vs_ref"] == pytest.approx(1.0)
    assert rows[0]["topk_spearman_vs_ref"] == pytest.approx(1.0)
    assert rows[0]["topk_kendall_vs_ref"] == pytest.approx(1.0)


def test_find_winner_breakpoints_groups_consecutive_same_winner():
    rows = [
        {"alpha": 1, "winner_idx": 10},
        {"alpha": 2, "winner_idx": 10},
        {"alpha": 3, "winner_idx": 20},
        {"alpha": 4, "winner_idx": 20},
        {"alpha": 5, "winner_idx": 20},
    ]
    intervals = ha.find_winner_breakpoints(rows)
    assert len(intervals) == 2
    assert intervals[0] == {"winner_idx": 10, "alpha_low": 1, "alpha_high": 2}
    assert intervals[1] == {"winner_idx": 20, "alpha_low": 3, "alpha_high": 5}