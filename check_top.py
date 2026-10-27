import config_space as cs
for k in (1, 2, 3, 4):
    inst = cs.load_instance_json(f"instances/instance_period{k}.json")   # adjust the name for instance 4
    print(k, len(inst.jobs), len(cs.active_jobs_of(inst)),
          [len(s) for s in inst.stage_configs], inst.n_operations,
          any(0 in mc.processing_times for s in inst.stage_configs for mc in s))