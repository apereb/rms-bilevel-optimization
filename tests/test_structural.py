import config_space as cs
import structural as st


def test_compute_measures_default_ids_are_positions(tiny_instance):
    configs = cs.feasible_configs(tiny_instance)
    theta, phi, nbrs = st.compute_measures(tiny_instance, configs)
    assert set(theta.keys()) == set(range(len(configs)))
    assert set(phi.keys()) == set(range(len(configs)))


def test_compute_measures_rejects_mismatched_config_ids(tiny_instance):
    configs = cs.feasible_configs(tiny_instance)
    import pytest
    with pytest.raises(ValueError):
        st.compute_measures(tiny_instance, configs, config_ids=list(range(len(configs) - 1)))


def test_phi_is_theta_when_no_neighbours(tiny_instance):
    # With tau=0, a config can only be its own neighbour (or have none),
    # so Phi(y) should reduce to Theta(y) exactly.
    import dataclasses
    zero_tau_instance = dataclasses.replace(tiny_instance, tau=0.0)
    configs = cs.feasible_configs(zero_tau_instance)
    theta, phi, nbrs = st.compute_measures(zero_tau_instance, configs)
    for idx in theta:
        if not nbrs[idx]:
            assert phi[idx] == theta[idx]


def test_global_config_ids_stay_joinable_across_a_filtered_subset(tiny_instance):
    """
    Regression test for the bug caught mid-conversation: structural
    filtering on the full universe, then scheduling only the surviving
    subset, must keep referring to the same configs by config_idx.
    """
    all_configs = cs.feasible_configs(tiny_instance)
    all_ids = list(range(len(all_configs)))
    theta, phi, _ = st.compute_measures(tiny_instance, all_configs, all_ids)

    median_phi = sorted(phi.values())[len(phi) // 2]
    kept_ids = [cid for cid in all_ids if phi[cid] >= median_phi]
    kept_configs = [all_configs[cid] for cid in kept_ids]

    # Recomputing theta/phi on just the kept subset, with the same global
    # ids, must reproduce identical theta values (theta is config-local,
    # doesn't depend on which other configs are in the list).
    theta_subset, _, _ = st.compute_measures(tiny_instance, kept_configs, kept_ids)
    for cid in kept_ids:
        assert theta_subset[cid] == theta[cid]
