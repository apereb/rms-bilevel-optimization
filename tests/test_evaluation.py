import evaluation as ev


def _fake_scheduling_results():
    # Hand-built, no Gurobi/config_space needed: just exercising the
    # cost/corridor/selection arithmetic in isolation.
    return [
        {"config_idx": 0, "y": "y0", "C_config": 10.0, "Q": 5.0, "perf": 2.0},
        {"config_idx": 1, "y": "y1", "C_config": 12.0, "Q": 4.0, "perf": 2.5},
        {"config_idx": 2, "y": "y2", "C_config": 20.0, "Q": 1.0, "perf": 10.0},
    ]


def test_compute_total_cost():
    results = _fake_scheduling_results()
    total_cost = ev.compute_total_cost(results, alpha=4.0)
    assert total_cost[0] == 10.0 + 4.0 / 2.0
    assert total_cost[1] == 12.0 + 4.0 / 2.5
    assert total_cost[2] == 20.0 + 4.0 / 10.0


def test_compute_total_cost_zero_perf_is_infinite():
    results = [{"config_idx": 0, "y": "y0", "C_config": 1.0, "Q": 0.0, "perf": 0.0}]
    total_cost = ev.compute_total_cost(results, alpha=1.0)
    assert total_cost[0] == float("inf")


def test_compute_corridor_and_c_star():
    results = _fake_scheduling_results()
    total_cost = ev.compute_total_cost(results, alpha=4.0)
    C_star, threshold, corridor = ev.compute_corridor(results, total_cost, epsilon=0.5)
    assert C_star == min(total_cost.values())
    assert threshold == 1.5 * C_star
    assert all(total_cost[r["config_idx"]] <= threshold for r in corridor)


def test_select_phase2_winner_is_argmax_phi_within_corridor():
    results = _fake_scheduling_results()
    total_cost = ev.compute_total_cost(results, alpha=4.0)
    _, _, corridor = ev.compute_corridor(results, total_cost, epsilon=1.0)  # wide corridor
    phi = {0: 5.0, 1: 9.0, 2: 1.0}
    winner = ev.select_phase2_winner(corridor, phi)
    assert winner["config_idx"] == max(
        (r["config_idx"] for r in corridor), key=lambda cid: phi[cid]
    )


def test_select_phase2_winner_empty_corridor_returns_none():
    assert ev.select_phase2_winner([], {}) is None


def test_evaluate_end_to_end_matches_manual_computation():
    results = _fake_scheduling_results()
    theta = {0: 1.0, 1: 2.0, 2: 3.0}
    phi = {0: 1.0, 1: 2.0, 2: 3.0}

    class FakeInstance:
        alpha = 4.0
        epsilon = 0.5

    out = ev.evaluate(FakeInstance(), results, theta, phi)
    total_cost = ev.compute_total_cost(results, alpha=4.0)
    assert out["C_star"] == min(total_cost.values())
    assert out["winner"]["config_idx"] == max(
        (r["config_idx"] for r in out["corridor"]), key=lambda cid: phi[cid]
    )
    # original dicts must not be mutated
    assert "total_cost" not in results[0]


def test_refinement_subset_uses_wider_corridor_than_final():
    results = _fake_scheduling_results()
    total_cost = ev.compute_total_cost(results, alpha=4.0)
    narrow = ev.refinement_subset(results, total_cost, refine_corridor_factor=0.0, epsilon=0.1)
    wide = ev.refinement_subset(results, total_cost, refine_corridor_factor=10.0, epsilon=0.1)
    assert len(wide) >= len(narrow)
