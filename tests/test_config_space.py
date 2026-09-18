import config_space as cs


def test_n_operations_is_full_catalogue(tiny_instance):
    # ops 1 and 2 appear across the tiny instance's MachineConfigs; op 0
    # (dummy) is never a processing_times key so it's excluded naturally.
    assert tiny_instance.n_operations == 2


def test_feasible_configs_nonempty_and_deterministic(tiny_instance):
    configs_a = cs.feasible_configs(tiny_instance)
    configs_b = cs.feasible_configs(tiny_instance)
    assert len(configs_a) > 0
    assert configs_a == configs_b, "enumeration order must be stable across calls"


def test_feasible_configs_all_pass_validation(tiny_instance):
    for y in cs.feasible_configs(tiny_instance):
        assert cs.validate_config(y, tiny_instance)


def test_config_setup_cost_matches_manual_sum(tiny_instance):
    y = cs.feasible_configs(tiny_instance)[0]
    expected = sum(n * mc.setup_cost for mc, n in y)
    assert cs.config_setup_cost(y) == expected


def test_config_to_dict_roundtrip_labels(tiny_instance):
    y = cs.feasible_configs(tiny_instance)[0]
    d = cs.config_to_dict(y)
    assert len(d) == tiny_instance.nb_stages
    assert all({"stage", "machine_id", "config_id", "n_machines"} <= entry.keys()
              for entry in d)


def test_count_feasible_matches_materialized_list(tiny_instance):
    assert cs.count_feasible_configs(tiny_instance) == len(cs.feasible_configs(tiny_instance))
