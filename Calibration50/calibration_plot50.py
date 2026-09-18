import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# Chargement
# ============================================================

with open("calibration_K50.json", "r", encoding="utf8") as f:
    data = json.load(f)

df = pd.DataFrame(data["probes"])

print(df.head())

# Automatically detect tested time limits
time_limits = sorted(df["time_limit"].unique())
ref_t = max(time_limits)

# ============================================================
# 1) Coût moyen en fonction du time limit
# ============================================================

cost_mean = (
    df.groupby("time_limit")["total_cost"]
      .mean()
      .sort_index()
)

plt.figure(figsize=(8,5))
plt.plot(cost_mean.index, cost_mean.values, marker="o")
plt.xlabel("Time limit (s)")
plt.ylabel("Average total cost")
plt.title("Average solution cost vs time limit")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 2) Gap moyen
# ============================================================

gap_mean = (
    df.groupby("time_limit")["mip_gap_pct"]
      .mean()
      .sort_index()
)

plt.figure(figsize=(8,5))
plt.plot(gap_mean.index, gap_mean.values, marker="o")
plt.xlabel("Time limit (s)")
plt.ylabel("Average MIP gap (%)")
plt.title("Average MIP gap vs time limit")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 3) Distribution des gaps
# ============================================================

plt.figure(figsize=(8,5))

groups = [
    df[df["time_limit"] == t]["mip_gap_pct"]
    for t in time_limits
]

plt.boxplot(groups, labels=time_limits)

plt.xlabel("Time limit (s)")
plt.ylabel("MIP gap (%)")
plt.title("Gap distribution")
plt.tight_layout()
plt.show()

# ============================================================
# 4) Gain moyen entre time limits
# ============================================================

pivot_cost = df.pivot(
    index="config_idx",
    columns="time_limit",
    values="total_cost"
)

gains = {}

for t1, t2 in zip(time_limits[:-1], time_limits[1:]):
    gains[f"{t1}→{t2}"] = (pivot_cost[t1] - pivot_cost[t2]).mean()

plt.figure(figsize=(7,5))
plt.bar(gains.keys(), gains.values())
plt.ylabel("Average cost improvement")
plt.title("Marginal gain from additional solve time")
plt.tight_layout()
plt.show()

# ============================================================
# 5) Corrélation des classements avec la plus longue limite
# ============================================================

print("\nSpearman rank correlations:")

for t in time_limits[:-1]:
    rho, _ = spearmanr(
        pivot_cost[t],
        pivot_cost[ref_t]
    )
    print(f"Spearman({t}s,{ref_t}s) = {rho:.4f}")

# ============================================================
# 6) Scatter coût(t) vs coût(ref)
# ============================================================

for t in time_limits[:-1]:

    plt.figure(figsize=(6,6))

    plt.scatter(
        pivot_cost[t],
        pivot_cost[ref_t],
        alpha=0.7
    )

    m = min(
        pivot_cost[t].min(),
        pivot_cost[ref_t].min()
    )

    M = max(
        pivot_cost[t].max(),
        pivot_cost[ref_t].max()
    )

    plt.plot([m, M], [m, M], "--")

    plt.xlabel(f"Cost @ {t}s")
    plt.ylabel(f"Cost @ {ref_t}s")
    plt.title(f"{t}s vs {ref_t}s")
    plt.tight_layout()
    plt.show()

# ============================================================
# 7) Amélioration relative entre la plus petite et la plus grande limite
# ============================================================

improvement_pct = (
    (pivot_cost[time_limits[0]] - pivot_cost[ref_t])
    / pivot_cost[ref_t]
    * 100
)

plt.figure(figsize=(8,5))
plt.hist(improvement_pct, bins=20)
plt.xlabel(f"Improvement from {time_limits[0]}s to {ref_t}s (%)")
plt.ylabel("Number of configurations")
plt.title("Distribution of improvements")
plt.tight_layout()
plt.show()

# ============================================================
# 8) Nombre de solutions quasi résolues
# ============================================================

threshold = 5.0

resolved = (
    df.assign(resolved=df["mip_gap_pct"] <= threshold)
      .groupby("time_limit")["resolved"]
      .mean()
      * 100
)

plt.figure(figsize=(8,5))
plt.plot(
    resolved.index,
    resolved.values,
    marker="o"
)

plt.xlabel("Time limit (s)")
plt.ylabel(f"% gap <= {threshold}%")
plt.title("Fraction of nearly solved instances")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 9) Rank stability
# ============================================================

print("\nRanking stability:")

rank = pivot_cost.rank()

for t in time_limits[:-1]:
    changed = (rank[t] != rank[ref_t]).sum()
    print(f"{t}s : {changed}/{len(rank)} rankings differ from {ref_t}s")