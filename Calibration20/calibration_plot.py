import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# Chargement
# ============================================================

with open("calibration.json", "r", encoding="utf8") as f:
    data = json.load(f)

df = pd.DataFrame(data["probes"])

print(df.head())

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
    for t in sorted(df["time_limit"].unique())
]

plt.boxplot(
    groups,
    labels=sorted(df["time_limit"].unique())
)

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

gains = {
    "5→15": (pivot_cost[5] - pivot_cost[15]).mean(),
    "15→30": (pivot_cost[15] - pivot_cost[30]).mean(),
    "30→60": (pivot_cost[30] - pivot_cost[60]).mean(),
}

plt.figure(figsize=(7,5))
plt.bar(gains.keys(), gains.values())
plt.ylabel("Average cost improvement")
plt.title("Marginal gain from additional solve time")
plt.tight_layout()
plt.show()

# ============================================================
# 5) Corrélation des classements avec 60s
# ============================================================

ref_rank = pivot_cost[60].rank()

for t in [5, 15, 30]:
    rho, p = spearmanr(
        pivot_cost[t],
        pivot_cost[60]
    )
    print(f"Spearman({t}s,60s) = {rho:.4f}")

# ============================================================
# 6) Scatter coût(t) vs coût(60)
# ============================================================

for t in [5, 15, 30]:

    plt.figure(figsize=(6,6))

    plt.scatter(
        pivot_cost[t],
        pivot_cost[60],
        alpha=0.7
    )

    m = min(
        pivot_cost[t].min(),
        pivot_cost[60].min()
    )

    M = max(
        pivot_cost[t].max(),
        pivot_cost[60].max()
    )

    plt.plot([m, M], [m, M], "--")

    plt.xlabel(f"Cost @ {t}s")
    plt.ylabel("Cost @ 60s")
    plt.title(f"{t}s vs 60s")
    plt.tight_layout()
    plt.show()

# ============================================================
# 7) Amélioration relative entre 5s et 60s
# ============================================================

improvement_pct = (
    (pivot_cost[5] - pivot_cost[60])
    / pivot_cost[60]
    * 100
)

plt.figure(figsize=(8,5))
plt.hist(improvement_pct, bins=20)
plt.xlabel("Improvement from 5s to 60s (%)")
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