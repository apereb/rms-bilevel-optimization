# Anchoring the degradation corridor on a lower bound: exact selection for the bi-level RMS pipeline

Companion document to `lb_analysis.py`. It covers the problem being solved, the mathematics, the computational side, and every result obtained on the three available instances (period 1, 2, 3). All numbers in Section 5 are copied from the actual outputs of `lb_analysis.py`; nothing is estimated.

Contents

1. Problem: the corridor shift caused by partial inversion
2. Mathematics
3. Computational side
4. Experimental protocol
5. Results (full data)
6. Discussion
7. Limits and what has not been tested
8. Appendix (code, worked example, sanity test)

---

## 1. Problem: the corridor shift caused by partial inversion

### 1.1 Pipeline recap

For a configuration y (one machine configuration and one machine count n_s per stage, 5 stages):

```
total_cost(y) = C_config(y) + alpha / perf(y),      perf(y) = B / Q(y)
             = C_config(y) + alpha * Q(y) / B
```

- `Q(y)` is the makespan returned by the Gurobi hybrid-flowshop MIP (15 s budget per config, so an incumbent, i.e. an upper bound on the true optimum).
- `B` = `batch_size` = number of scheduled jobs (`build_scheduled_jobs`: K copies apportioned by demand, with ceil).
- `C*` = min over the evaluated configurations of `total_cost`; the corridor is `{y : total_cost(y) <= (1+eps) C*}`; Phase 2 selects the corridor config with maximal `Phi` (structural flexibility, no scheduling needed).

`partial-inversion` computes `Phi` on the full universe U (cheap), keeps only the top fraction f by `Phi` (call this set K), schedules only K, and runs Phase 1 and Phase 2 on K.

### 1.2 The shift

Let `C*_K = min_{y in K} total_cost(y)`. If the true optimum y* is not in K (its Phi is low), then `C*_K > C*`. The heuristic's threshold moves from `(1+eps) C*` to `(1+eps) C*_K`. Configurations with cost in the interval `((1+eps) C*, (1+eps) C*_K]` enter the corridor although they are outside the baseline corridor. If one of them has a higher Phi than the baseline winner y_w, it wins, even if y_w itself is in K and in the baseline corridor.

So there are two distinct failure modes:

- **Shift failure:** y_w is in K, but a "new" config admitted by the shifted threshold beats it.
- **Recall failure:** y_w is not in K (its Phi rank is worse than f·|U|).

This document addresses the first mode, and, as a by-product, removes the need for the parameter f (which addresses most of the second).

### 1.3 The idea

Fix the threshold with a quantity that does not depend on which configurations were removed: a lower bound (or a cheaply certified value) of C*, computed **before and independently of** the Phi cut.

---

## 2. Mathematics

### 2.1 Notation

- S = 5 stages, J = scheduled jobs (|J| = B), stage s has n_s identical machines.
- `p[s][j]` = processing time of job j at stage s (0 = no-op; a 0-time operation takes no time and is irrelevant for the bound).
- `n_eff[s] = max(1, min(n_s, #{j : p[s][j] > 0}))`, exactly the machine count the MIP uses (`schedule_configs`).
- Heads and tails: `h_j(s) = sum_{t<s} p[t][j]`, `t_j(s) = sum_{t>s} p[t][j]`.
- Model assumed: hybrid flow shop (identical parallel machines per stage, stages visited in a fixed order), processing times only (no setups, transport, release dates or due dates). This was confirmed for the instances used.

### 2.2 Lower bound on the makespan

**Proposition (job bound).** `Q >= max_j sum_s p[s][j]`. (A job cannot be processed in parallel with itself.)

**Stage bound (Santos, Hunsucker and Deal, SHD).** Fix a stage s and let `J_s = {j : p[s][j] > 0}` (the jobs that really occupy a machine at stage s), `m = min(n_eff[s], |J_s|)`, `L_s = sum_{j in J_s} p[s][j]`, `H_m` = sum of the m smallest heads `h_j(s)` over J_s, `T_m` = sum of the m smallest tails `t_j(s)` over J_s. Then

```
Q  >=  ( H_m + L_s + T_m ) / m                                    (SHD, m machines)
```

This is the center-based bound of Santos et al. (their eq. (1); the published version also takes a ceiling because their data are integer, which is not applied here since processing times are fractional). Two adaptations to our instances: (i) only jobs with `p[s][j] > 0` are considered at stage s, since a 0-time no-op does not start or occupy a machine and cannot serve as the "first" or "last" job of one; (ii) `m` uses the machine count the MIP really has, `n_eff`.

**What is proved here, and what is not.** Define, for any k, `f(k) = (H_k + L_s + T_k) / k`.

*Lemma A (proved).* For any schedule, if k is the number of machines of stage s that process at least one job, then `Q >= f(k)`.

*Proof.* Let `A_i` be the job set of used machine i, `first_i` and `last_i` its first and last jobs. Machine i cannot start `first_i` before `h_{first_i}(s)`, must process `P_i = sum_{j in A_i} p[s][j]` in sequence, and after `last_i` that job still needs `t_{last_i}(s)`, so `Q >= h_{first_i} + P_i + t_{last_i}`. Summing over the k used machines: `k Q >= sum_i h_{first_i} + sum_i P_i + sum_i t_{last_i}`. The `first_i` are k distinct jobs, so the first sum is >= `H_k`; the `last_i` are k distinct jobs, so the third sum is >= `T_k`; the `A_i` partition `J_s`, so the middle sum is `L_s`. Divide by k. ∎

Since the number k of used machines is unknown, Lemma A directly gives the valid bound `min over k in 1..m of f(k)` (call it the *min-version*). It never exceeds SHD's `f(m)`, because k = m is one of the terms of the min.

*Status of SHD's `f(m)` (k = m only).* SHD is a published, peer-reviewed bound, and I rely on it. I did not derive its validity from Lemma A: `f(k)` can be *smaller or larger* at k = m than at some k < m (from `f(k+1) = (k f(k) + h_{k+1} + t_{k+1})/(k+1)`, `f` decreases while `h_{k+1} + t_{k+1} <= f(k)` and then increases, so the k = m term is not always the smallest term of the min). What I checked empirically is that, once combined with the job bound, the two versions coincide:

- on the real instances (your check): the min-version never returned a value different from SHD's;
- on 4,000 random instances plus 200,000 adversarial ones (heavy-tailed heads and tails, small loads): `max(job bound, SHD stage bound)` and `max(job bound, min-version)` were identical every time (so SHD inherits the min-version's validity on those instances); on the 4,000 random ones SHD also never exceeded the makespan of a feasible schedule found by random search;
- a hill-climbing search (300 restarts × 4,000 moves, 3 to 8 jobs, 2 to 8 machines) maximizing `SHD - max(job bound, min-version)` never found a positive value (maximum exactly 0).

So the decision to use SHD as is stands. The equivalence above is evidence, not a proof; the min-version has a complete proof (Lemma A) and would be the safe fallback if a model change makes SHD's assumptions doubtful.

**Definition.**
```
LB_Q(y) = max( job bound, max_s SHD stage bound )
```

Cost: O(S · J log J) per configuration (a sort per stage); negligible against a MIP solve.

### 2.3 Lower bound on the cost

```
LB_cost(y) = C_config(y) + alpha * LB_Q(y) / B   <=   total_cost(y)
```

`C_config` is exact, alpha > 0, and the MIP incumbent Q is >= the true optimum >= LB_Q. So `LB_cost(y) <= total_cost(y)` for every y, whatever the solver's stopping status. This needs no Gurobi call and is independent of Phi.

### 2.4 Three ways to anchor the corridor

Let `r = C* / A_LB` where `A_LB = min_{y in U} LB_cost(y)` (so `A_LB <= C*`, r >= 1).

**(a) LB anchor, inflated epsilon.** Threshold `(1+eps') A_LB`. Since `A_LB <= C*`, `eps' = eps` gives a corridor that is a subset of the baseline one (no config can enter, but the baseline winner may be lost). The threshold reproduces the baseline one exactly for

```
eps'_exact = (1 + eps) * r - 1
```

and keeps y_w inside the corridor for `eps' >= eps'_keep = total_cost(y_w) / A_LB - 1`. For `eps'` in the window `[eps'_keep, eps'_exact]` the corridor is a subset of the baseline one that contains y_w, so y_w is returned as Phase-2 winner (it has the maximal Phi of the whole baseline corridor, hence of any subset containing it). Above `eps'_exact`, extras enter, and an extra with a larger Phi than y_w would displace it (y_w stays in the corridor but is no longer selected). Its position depends on r, which is unknown without the baseline, and corridors are dense in cost (Section 5), so a wrong eps' either loses the winner or admits many extras. This variant is therefore only useful as a diagnostic.

**(b) UB anchor from a few solves.** Solve the m configurations with the smallest `LB_cost`; let `UB_m` be the best `total_cost` among them. Then `UB_m >= C*` and the threshold `(1+eps) UB_m` gives a corridor that is a **superset** of the baseline corridor. The only extras are configs with cost in `((1+eps) C*, (1+eps) UB_m]`, so the gap `UB_m/C* - 1` controls the pollution. The baseline winner y_w always stays in the corridor, but it is returned as Phase-2 winner only if no extra has a larger Phi; when `UB_m = C*` there are no extras and the corridor is exactly the baseline one.

**(c) Certified anchor (best-first).** Solve configs in increasing `LB_cost`; maintain `best` = smallest solved cost; stop as soon as the next `LB_cost >= best`.

*Claim:* at the stop, `best = C*` (over the same 15 s incumbents). *Proof:* every unsolved config y has `total_cost(y) >= LB_cost(y) >= best`, so none can improve on `best`. ∎

The number of solves is `#{y : LB_cost(y) < C*}`, plus one if the optimum's own bound equals C* (ties aside). Indeed every config with `LB_cost < C*` must be solved (the running `best` is always >= C*), and the process cannot stop before reaching the optimum y* since all configs before it in the order have `LB_cost <= LB_cost(y*) <= C* <= best`. This count is computable offline once C* is known and is small when the bound is tight (period 3: no config has `LB_cost < C*`, one solve; period 1: 14 solves). Note that the UB anchor finds C* earlier than it can be certified: in all three instances `UB_5 = C*`, yet certification in period 1 needs the 14 solves.

### 2.5 Exact selection without a Phi fraction

Let `E = {y in U : LB_cost(y) <= (1+eps) C*}`. Any corridor config satisfies `LB_cost <= total_cost <= (1+eps) C*`, so **corridor ⊆ E**. E is computable before any solve beyond the certified anchor.

**Theorem.** Sort E by decreasing Phi (ties broken by index) and solve in that order. The first config whose real `total_cost <= (1+eps) C*` is `argmax Phi` over the corridor, i.e. the baseline Phase-2 winner (up to tie-breaking among equal Phi values).

*Proof.* Everything above it in the order has a higher (or equal, earlier-indexed) Phi and was solved and found outside the corridor. All corridor members are in E, so none was skipped. ∎

Cost in solves: `n_anchor + rank of y_w among E by Phi` (minus overlap between the two sets). No parameter `phi_top_fraction`, and no recall failure: the method returns the baseline winner by construction, given the same solver incumbents.

Comparison with fixed-fraction partial inversion:

| | fixed fraction f | exact selection |
|---|---|---|
| solves | f · #U | n_anchor + rank of y_w in E |
| failure mode | recall (rank(y_w) > f·|U|) and shift | none (by the theorem); only solver noise |
| parameter | f, instance dependent | none |

Phi is purely structural (instance and configuration properties), so computing it over the whole universe involves no scheduling; this is already done by `partial-inversion` (`structural.compute_measures`).

---

## 3. Computational side

### 3.1 Data flow

```
config_space.feasible_configs(instance)      -> universe U, config_idx = position
scheduling.build_scheduled_jobs(instance)     -> J (K copies by demand), B = |J|
config_space.build_pt_matrix(y, instance, J)  -> pt[s][j]   (stage-major)
lb_analysis.effective_machines(y, pt)         -> n_eff (same rule as the MIP)
lb_analysis.makespan_lb(pt, n_eff)            -> LB_Q(y)
LB_cost(y) = config_setup_cost(y) + alpha * LB_Q / B
structural.compute_measures(instance, U, ids) -> theta, phi, neighbours (Phi over full U)
```

### 3.2 Procedure for a real run (no baseline available)

```
1. U    <- feasible_configs; Phi <- compute_measures over U           (no Gurobi)
2. LB_cost(y) for all y in U                                          (no Gurobi)
3. Anchor: for y in U sorted by LB_cost:
       if LB_cost(y) >= best: stop
       solve y (15 s); best <- min(best, total_cost(y))
   -> C* := best  (certified over incumbents)
4. E <- { y : LB_cost(y) <= (1+eps) C* }
   for y in E sorted by decreasing Phi:
       solve y unless already solved
       if total_cost(y) <= (1+eps) C*:  return y   # Phase-2 winner
```

If the whole corridor is needed (plots, `compare`), solve all of E instead of stopping at the first hit; that is |E| solves (7% to 23% of U on the three instances), still far below the full run.

Suggested integration: a new `orchestrate.py` mode (for example `lb-exact`) implementing steps 1 to 4, writing the usual `.meta.json` and `.configs.jsonl` so that `compare_runs.py` and `heuristic_analysis.py compare` work unchanged. Not implemented yet: `lb_analysis.py` currently only simulates the procedure on a finished `full` baseline, reusing its `Q` values.

### 3.3 `lb_analysis.py`: what it does

Usage:

```
python lb_analysis.py instance.json baseline.meta.json \
    --eps-primes 0.151,0.155,0.16 --phi-fractions 1.0,0.5,0.3 --ms 5,10,25,50 [--tol 1e-3]
```

Sections of its output:

| Section | Content |
|---|---|
| `[LB validity]` | worst `LB_Q - Q` over the baseline, number above 0, number above `--tol` |
| `[Tightness]` | `LB_cost / total_cost` (min / median / max) over all configs, over the corridor, and for the winner |
| `[Anchor]` | `LB_min`, `C*`, `LB_min/C*`, `eps'_exact`, `eps'_keep`, number of configs with `LB_cost <= (1+eps)C*` |
| `[Phi]` | Phi rank of the baseline winner and the smallest `phi_top_fraction` that keeps it |
| `[Certified anchor]` | best-first solves needed to certify `C*` (simulated with baseline costs) |
| `[Exact selection]` | size of E, solves until the first corridor hit in decreasing Phi, and total solves |
| `[UB anchor]` | for each m: `UB_m`, `UB_m/C*`, corridor size, extras |
| `[Sweep]` | for each Phi fraction: shifted anchor (current heuristic), LB_min anchor for each eps', UB anchor for each m; corridor size, extras vs baseline corridor, Phase-2 winner, match |

Assumptions in the code:
- `config_idx` equals the position in `feasible_configs(instance)` (as in `compare_baseline_vs_heuristic`).
- The baseline covers the whole universe (no `--limit`).
- `Q` is in source units (unscaled), so `LB_Q` and `Q` are comparable. The ~1e-6 level excesses seen (Section 5.1) are probably solver tolerance or time-scale rounding; this was not investigated.
- `compute_measures` returns `(theta, phi, neighbours)` keyed by config_idx.
- Imports of project modules are lazy; `scheduling` is imported only for `build_scheduled_jobs` (it needs `gurobipy` installed but makes no Gurobi call).

Complexity: LB for all configs is O(|U| · S · J log J), negligible. The dominant cost of the analysis is `compute_measures` (pairwise neighbourhood comparison inside machine-type groups), already paid by `partial-inversion`.

---

## 4. Experimental protocol

- Three instances (period 1, 2, 3), each with a complete `full` baseline (15 s per config, `alpha` and `epsilon` as recorded in the instance; the printed shifted-anchor rows show `eps = 0.15` on all three).
- Everything is simulated offline from the baseline's own `Q` values; no new Gurobi solve. Consequence: the simulation ignores solver-to-solver stochasticity (previously measured at Spearman about 0.999 between real re-solves).
- Baseline reference values:

| | period 1 | period 2 | period 3 |
|---|---|---|---|
| universe size #U | 3024 | 3990 | 4500 |
| C* | 23.9781 | 32.1881 | 31.2548 |
| threshold (1+eps)C*, eps = 0.15 | 27.5748 | 37.0163 | 35.9430 |
| baseline corridor size | 526 | 315 | 260 |
| baseline Phase-2 winner (config_idx) | 1419 | 199 | 90 |

(The thresholds are computed as 1.15 × C* from the C* values in the logs.)

---

## 5. Results (full data)

### 5.1 Bound validity

Final runs (`--tol 1e-3`):

| | period 1 | period 2 | period 3 |
|---|---|---|---|
| configs checked | 3024 | 3990 | 4500 |
| worst `LB_Q - Q` | -3.60e-01 | 2.50e-06 | 7.50e-06 |
| number with `LB_Q - Q > 0` (1e-9) | 0 | 44 | 96 |
| number above tol = 1e-3 | 0 | 0 | 0 |

The first runs used a strict tolerance of 1e-6 and reported, at that time, "3 violation(s), worst +0.0000" for period 2 (config_idx 166, 2408, 2826) and "8 violation(s), worst +0.0000" for period 3 (config_idx 570, 615, 660, 1020, 1065, 1110, 1635, 2085); period 1 had 0. These are the tiny gaps (about 1e-6 to 1e-5) later quantified above, probably due to solver tolerance or time-scale rounding (not investigated); no violation above 1e-3 was found. Period 1's worst value of -0.36 means the bound is strictly below Q everywhere on that instance (minimum slack 0.36 in makespan units).

### 5.2 Tightness

`LB_cost / total_cost`:

| | period 1 | period 2 | period 3 |
|---|---|---|---|
| all configs: min / median / max | 0.955 / 0.982 / 0.997 | 0.981 / 0.990 / 1.000 | 0.980 / 0.991 / 1.000 |
| corridor: min / median / max | 0.955 / 0.978 / 0.997 | 0.982 / 0.991 / 1.000 | 0.980 / 0.991 / 1.000 |
| baseline winner | 0.978 (idx 1419) | 0.994 (idx 199) | 1.000 (idx 90) |

### 5.3 Anchor quantities

| | period 1 | period 2 | period 3 |
|---|---|---|---|
| `LB_min` (config_idx) | 23.4162 (324) | 31.8976 (0) | 31.2548 (225) |
| `C*` | 23.9781 | 32.1881 | 31.2548 |
| `LB_min / C*` | 0.977 | 0.991 | 1.000 |
| `eps'_exact` (reproduces threshold) | 0.178 | 0.160 | 0.150 |
| `eps'_keep` (keeps the winner) | 0.172 | 0.151 | 0.147 |
| configs with `LB_cost <= (1+eps)C*` (set E) | 703 (23.2%) | 375 (9.4%) | 311 (6.9%) |

Note: in period 3 the config with the smallest LB is the optimum itself and its bound is exact (`LB_min = C*`).

### 5.4 Phi rank of the baseline winner

| | period 1 | period 2 | period 3 |
|---|---|---|---|
| Phi rank of winner | 427 / 3024 | 472 / 3990 | 1534 / 4500 |
| smallest `phi_top_fraction` keeping it | 14.1% | 11.8% | 34.1% |

### 5.5 Certified anchor and exact selection

| | period 1 | period 2 | period 3 |
|---|---|---|---|
| certified-anchor solves (best-first in LB order) | 14 (0.5%) | 1 (0.0%) | 1 (0.0%) |
| resulting anchor equals baseline C* | yes | yes | yes |
| exact selection: solves until first corridor hit (decreasing Phi over E) | 5 | 2 | 1 |
| returned config equals baseline winner | 1419: yes | 199: yes | 90: yes |
| total solves including anchor | 19 (0.6%) | 3 (0.1%) | 2 (0.0%) |

For reference, fixed-fraction partial inversion at f = 30% solves 907, 1197 and 1350 configurations respectively, and full baseline solves 3024, 3990 and 4500.

### 5.6 UB anchor (top-m by LB)

Identical for m = 5, 10, 25, 50 on all three instances:

| | period 1 | period 2 | period 3 |
|---|---|---|---|
| `UB_m` | 23.9781 | 32.1881 | 31.2548 |
| `UB_m / C*` | 1.0000 | 1.0000 | 1.0000 |
| corridor size | 526 | 315 | 260 |
| extras vs baseline corridor | 0 | 0 | 0 |

(m = 5 already contains the optimum on every instance.)

### 5.7 Sweep: partial inversion with different anchors

Columns: corridor size / extras vs baseline corridor / Phase-2 winner / equals baseline winner. "shifted" is the current heuristic (anchor = best cost among the kept configs, eps = 0.15). "LB_min eps'" is the corridor `(1+eps') LB_min`. "UB (m = 5..50)" is the UB anchor at eps' = eps = 0.15; the four values of m give identical rows. Each table lists every eps' value that was run on that instance (the lists differ between instances, see 8.4).

#### Period 1 (baseline winner 1419, corridor 526)

| anchor | 100% (3024 kept) | 50% (1512 kept) | 30% (907 kept) |
|---|---|---|---|
| shifted, 0.15 | 526 / 0 / 1419 / yes | 500 / 449 / 1743 / no | 276 / 252 / 285 / no |
| LB_min 0.15 | 323 / 0 / 135 / no | 11 / 0 / 135 / no | 6 / 0 / 135 / no |
| LB_min 0.151 | 328 / 0 / 135 / no | 11 / 0 / 135 / no | 6 / 0 / 135 / no |
| LB_min 0.155 | 354 / 0 / 135 / no | 16 / 0 / 135 / no | 7 / 0 / 135 / no |
| LB_min 0.16 | 393 / 0 / 135 / no | 28 / 0 / 135 / no | 10 / 0 / 135 / no |
| LB_min 0.172 | 475 / 0 / 1419 / yes | 43 / 0 / 1419 / yes | 20 / 0 / 1419 / yes |
| LB_min 0.175 | 505 / 0 / 1419 / yes | 46 / 0 / 1419 / yes | 20 / 0 / 1419 / yes |
| LB_min 0.178 | 528 / 2 / 1419 / yes | 53 / 2 / 1419 / yes | 24 / 0 / 1419 / yes |
| LB_min 0.2 | 703 / 177 / 162 / no | 105 / 54 / 162 / no | 48 / 24 / 162 / no |
| LB_min 0.3 | 1723 / 1197 / 285 / no | 586 / 535 / 285 / no | 291 / 267 / 285 / no |
| LB_min 0.4 | 2564 / 2038 / 1833 / no | 1130 / 1079 / 1833 / no | 642 / 618 / 1833 / no |
| LB_min 0.5 | 2934 / 2408 / 1833 / no | 1426 / 1375 / 1833 / no | 842 / 818 / 1833 / no |
| UB (m = 5..50), 0.15 | 526 / 0 / 1419 / yes | 51 / 0 / 1419 / yes | 24 / 0 / 1419 / yes |

#### Period 2 (baseline winner 199, corridor 315)

| anchor | 100% (3990 kept) | 50% (1995 kept) | 30% (1197 kept) |
|---|---|---|---|
| shifted, 0.15 | 315 / 0 / 199 / yes | 95 / 33 / 1757 / no | 139 / 106 / 143 / no |
| LB_min 0.15 | 253 / 0 / 1625 / no | 48 / 0 / 1625 / no | 22 / 0 / 1625 / no |
| LB_min 0.151 | 258 / 0 / 1625 / no | 48 / 0 / 1625 / no | 22 / 0 / 1625 / no |
| LB_min 0.155 | 281 / 0 / 199 / yes | 55 / 0 / 199 / yes | 27 / 0 / 199 / yes |
| LB_min 0.16 | 312 / 0 / 199 / yes | 62 / 0 / 199 / yes | 33 / 0 / 199 / yes |
| LB_min 0.2 | 557 / 242 / 1757 / no | 119 / 57 / 1757 / no | 72 / 39 / 1757 / no |
| LB_min 0.3 | 1488 / 1173 / 238 / no | 440 / 378 / 238 / no | 258 / 225 / 238 / no |
| LB_min 0.4 | 2547 / 2232 / 2045 / no | 955 / 893 / 2045 / no | 523 / 490 / 2045 / no |
| LB_min 0.5 | 3330 / 3015 / 2045 / no | 1461 / 1399 / 2045 / no | 818 / 785 / 2045 / no |
| UB (m = 5..50), 0.15 | 315 / 0 / 199 / yes | 62 / 0 / 199 / yes | 33 / 0 / 199 / yes |

The script prints `eps'_keep` = 0.151 (rounded to three decimals), yet eps' = 0.151 still loses the winner while 0.155 keeps it. So the true `eps'_keep` lies in (0.151, 0.155].

#### Period 3 (baseline winner 90, corridor 260)

| anchor | 100% (4500 kept) | 50% (2250 kept) | 30% (1350 kept) |
|---|---|---|---|
| shifted, 0.15 | 260 / 0 / 90 / yes | 290 / 286 / 196 / no | 210 / 210 / 202 / no |
| LB_min 0.15 | 260 / 0 / 90 / yes | 4 / 0 / 90 / yes | 0 / 0 / none / no |
| LB_min 0.151 | 265 / 5 / 90 / yes | 4 / 0 / 90 / yes | 0 / 0 / none / no |
| LB_min 0.155 | 283 / 23 / 90 / yes | 6 / 2 / 90 / yes | 0 / 0 / none / no |
| LB_min 0.16 | 312 / 52 / 90 / yes | 6 / 2 / 90 / yes | 0 / 0 / none / no |
| LB_min 0.2 | 550 / 290 / 1080 / no | 24 / 20 / 1080 / no | 3 / 3 / 1080 / no |
| LB_min 0.3 | 1432 / 1172 / 195 / no | 211 / 207 / 195 / no | 63 / 63 / 195 / no |
| LB_min 0.4 | 2590 / 2330 / 199 / no | 734 / 730 / 199 / no | 305 / 305 / 199 / no |
| LB_min 0.5 | 3589 / 3329 / 200 / no | 1416 / 1412 / 200 / no | 694 / 694 / 200 / no |
| UB (m = 5..50), 0.15 | 260 / 0 / 90 / yes | 4 / 0 / 90 / yes | 0 / 0 / none / no |

### 5.8 Summary of the anchors on the winner-recovery criterion

Number of the 9 (instance, fraction) cells in which the baseline winner is returned:

| anchor | cells recovered |
|---|---|
| shifted (current heuristic) | 3 / 9 (the three 100% cells, where nothing is cut) |
| LB_min with eps' = eps = 0.15 | 2 / 9 (period 3 at 100% and 50%; periods 1 and 2 fail at all three fractions) |
| UB anchor (m >= 5), eps' = eps | 8 / 9 (only period 3 at 30% fails: winner not in the top 30% by Phi) |
| exact selection (no fraction) | 3 / 3 instances |

---

## 6. Discussion

1. **The shift is real and large.** With the current anchor, cutting to 50% or 30% by Phi admits 33 to 449 configurations outside the baseline corridor and the winner is lost in 6 of 6 instance-fraction cells where cutting occurs (periods 1 and 2 at 50% and 30%; period 3 at 50% and 30%). Only at 100% (no cut) does the current method reproduce the baseline.
2. **The bound is tight enough.** `LB_cost / total_cost` has median 0.98 to 0.99 and never falls below 0.955. `LB_min / C*` is 0.977, 0.991, 1.000. This is much tighter than the width eps = 0.15 of the corridor, which is why an anchor based on it works at all.
3. **A fixed eps' cannot be the mechanism.** The admissible window `[eps'_keep, eps'_exact]` is 0.172 to 0.178, 0.151 to 0.160, 0.147 to 0.150 across periods. Corridors are dense in cost: in period 2, eps' = 0.2 admits 242 extras (77% of the baseline corridor size); in period 3, eps' = 0.16 admits 52. The right eps' needs r = C*/LB_min, unknown a priori.
4. **A handful of solves fixes the anchor.** Best-first in LB order certifies C* with 14, 1 and 1 solves (0.5% or less of the universe). Any m >= 5 is enough for the UB anchor on these instances, because the optimum is among the 5 configs with smallest LB.
5. **Exact selection needs 19, 3 and 2 solves in total** (0.6%, 0.1%, 0.04% of the universe), because the LB-eligible set E (7% to 23% of U) can be walked in decreasing Phi and the first corridor member is the winner. Fixed-fraction partial inversion at f = 30% solves 907, 1197, 1350 configs and still fails in period 3 (the winner's Phi rank corresponds to 34.1%).
6. **What remains of partial inversion is recall.** The winner's Phi rank fraction varies by a factor of three (11.8%, 14.1%, 34.1%), so no single f is safe. Exact selection removes f entirely.
7. **Sensitivity to alpha is untested.** `LB_Q` does not depend on alpha, but the gap of `LB_cost` is `alpha * (Q - LB_Q) / B`, so when alpha grows relative to `C_config` the makespan term dominates and the tightness of `LB_Q` matters more (Section 7).

---

## 7. Limits and what has not been tested

- **Simulation, not a real run.** All results reuse the baseline's `Q`. In a real run, each solve is a 15 s incumbent and can differ slightly from the baseline's (Spearman about 0.999 measured earlier). A config sitting right at the corridor threshold could flip; the anchor (min cost over few configs) is robust to this, the selection near the threshold less so.
- **Three instances, one family.** Periods 1 to 3 presumably share the same structure. The certified-anchor count (14, 1, 1) depends on the tightness of the bound; period 1 (min ratio 0.955) needed 14. An instance with a looser bound would need more solves; this is the number to look at first on any new instance.
- **Model-dependent validity.** `makespan_lb` is valid for processing times only, stages in fixed order. Setups between jobs, transport, release dates or non-permutation restrictions in the MIP would require re-deriving the bound (additive terms may need their own lower bounds).
- **Ties in Phi.** The exact-selection theorem returns the winner up to tie-breaking. The order used here is decreasing Phi then increasing index; `evaluation.select_phase2_winner` may break ties differently. Not checked.
- **alpha and epsilon sensitivity.** Only the recorded alpha and eps = 0.15 were used. `alpha-sweep` in `heuristic_analysis.py` shows the winner can change with alpha; the LB is independent of alpha (`LB_Q` does not use it) but `LB_cost` is not. Suggested check: add an `--alpha` override to `lb_analysis.py` and repeat.
- **Complete corridor requires solving E.** Exact selection returns the winner, not the corridor. If the corridor is needed (plots, Jaccard comparisons), |E| solves are required.
- **Simple bound.** No two-machine Johnson bound or LP relaxation was tried; not needed on these instances, possible if a future instance is looser.

Possible next steps, in order of value: (1) implement the real `orchestrate.py` mode and compare its output to the baselines with `heuristic_analysis.py compare`; (2) test on an instance with a different structure (looser LB) and with other alpha values; (3) check the Phi tie-breaking convention.

---

## 8. Appendix

### 8.1 The bound in code

```python
def makespan_lb(pt_matrix, n_machines):
    """pt_matrix[s][j]: processing time of job j at stage s (0 = no-op).
    n_machines[s]: machines effectively available at stage s.
    Job bound + SHD (Santos, Hunsucker, Deal) stage bound with m = min(n_s, #active jobs)."""
    S = len(pt_matrix)
    J = len(pt_matrix[0]) if S else 0
    lb = 0.0
    for j in range(J):                                   # job bound
        lb = max(lb, sum(pt_matrix[s][j] for s in range(S)))
    for s in range(S):                                   # SHD stage bound
        idx = [j for j in range(J) if pt_matrix[s][j] > 0]
        if not idx:
            continue
        heads = sorted(sum(pt_matrix[t][j] for t in range(s)) for j in idx)
        tails = sorted(sum(pt_matrix[t][j] for t in range(s + 1, S)) for j in idx)
        load = sum(pt_matrix[s][j] for j in idx)
        m = min(n_machines[s], len(idx))
        stage_lb = (sum(heads[:m]) + load + sum(tails[:m])) / m
        lb = max(lb, stage_lb)
    return lb


def effective_machines(y, pt_matrix):
    """Same rule as scheduling.schedule_configs."""
    return [max(1, min(n, sum(1 for v in pt_matrix[s] if v > 0)))
            for s, (_, n) in enumerate(y)]
```

Note: the runs reported in Section 5 were produced with the earlier min-over-k version of `makespan_lb`; your check that it never differs from SHD on these instances means the results carry over unchanged.

### 8.2 Worked example of the stage bound

Three jobs, three stages, stage 2 has two machines. Processing times (stage 1, 2, 3):

```
A: 1, 4, 1      B: 2, 3, 1      C: 1, 5, 2
```

At stage 2: heads (stage 1) are A = 1, B = 2, C = 1; tails (stage 3) are A = 1, B = 1, C = 2; load L = 4 + 3 + 5 = 12.

- k = 1: (smallest head 1 + 12 + smallest tail 1) / 1 = 14
- k = 2: (1 + 1 + 12 + 1 + 1) / 2 = 8

SHD stage bound (m = 2): 8 (the min-version also takes the min with k = 1, i.e. min(14, 8) = 8: same value). Job bound = max(6, 6, 8) = 8. `LB_Q = 8`. In this example the two versions agree; Lemma A requires the min over k because k, the number of machines actually used, is not known in advance.

### 8.3 Sanity test of the bound (abstract instances)

3,000 random instances (5 stages, 2 to 9 jobs, random 0-time operations with probability about 2/3 per operation, 1 to 3 machines per stage capped by the number of non-zero jobs). For each, the bound was compared with the best makespan among 200 random permutation schedules with greedy earliest-machine assignment (a feasible schedule, hence an upper bound on the optimum).

- violations (`LB_Q > best found`): 0 out of 3,000
- mean `LB_Q / best found`: 0.965

This test validates the bound in the abstract; the check on real data is the `[LB validity]` line of Section 5.1.

That test used the min-version. The SHD version (k = m only) was tested afterwards against the same kind of feasible schedules (4,000 random instances, no violation) and adversarially (200,000 instances with heavy-tailed heads and tails, then a hill-climbing search); it coincided with the min-version every time (see Section 2.2).

### 8.4 Inventory of the runs behind this document

| run | instance | eps' list | Phi fractions | m list | notes |
|---|---|---|---|---|---|
| 1 | period 1 | 0.15, 0.2, 0.3, 0.4, 0.5 | 1.0, 0.5, 0.3 | (not yet implemented) | first version of the script; tolerance 1e-6; 0 violations |
| 1 | period 2 | same | same | - | 3 violations at tol 1e-6 (worst +0.0000) |
| 1 | period 3 | same | same | - | 8 violations at tol 1e-6 (worst +0.0000) |
| 2 | period 1 | 0.172, 0.175, 0.178 | same | - | |
| 3 | period 1 | 0.151, 0.155, 0.16 | same | 5, 10, 25, 50 | tolerance-aware, adds Phi rank, certified anchor, UB anchor |
| 3 | period 2 | 0.151, 0.155, 0.16 | same | 5, 10, 25, 50 | |
| 3 | period 3 | 0.151, 0.155, 0.16 | same | 5, 10, 25, 50 | (a first attempt failed on a mistyped file name, then reran) |
| 4 | period 1 | 0.151, 0.155, 0.16, 0.172, 0.175, 0.178 | same | 5, 10, 25, 50 | adds exact selection; source file `lb_period1.txt` |
| 4 | period 2 | 0.151, 0.155, 0.16 | same | 5, 10, 25, 50 | `lb_period2.txt` |
| 4 | period 3 | 0.15, 0.151, 0.155, 0.16 | same | 5, 10, 25, 50 | `lb_period3.txt` |