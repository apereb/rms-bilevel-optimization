import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# PARSER
# ------------------------------------------------------------
def parse_log(path):
    import re
    import pandas as pd

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    configs = []
    current = None

    print("Scanning log...")

    for i, line in enumerate(lines):

        # DEBUG: show suspicious lines
        if "Cconfig" in line or "Cmax" in line or "TP" in line or "Total" in line:
            print(f"[DEBUG line {i}] {line.strip()}")

        # try to detect ANY config start
        # (very loose pattern on purpose)
        if "[" in line and "/" in line and "]" in line:
            try:
                parts = re.findall(r"\d+", line)
                if len(parts) >= 2:
                    if current is not None:
                        configs.append(current)

                    current = {
                        "id": int(parts[0]),
                        "total": int(parts[1]),
                        "status": "UNKNOWN"
                    }
            except:
                pass

        if current is None:
            continue

        # ultra-loose parsing (handles most formats)

        # Prefer extracting the number directly after the label to avoid
        # picking the last numeric token on the same line (which can be
        # e.g. the Total value when multiple numbers are present).
        if "Cconfig" in line:
            m = re.search(r"Cconfig\s*=\s*([-+]?\d*\.?\d+)", line)
            if m:
                current["cost"] = float(m.group(1))

        if "Cmax" in line:
            m = re.search(r"Cmax\s*=\s*([-+]?\d*\.?\d+)", line)
            if m:
                current["cmax"] = float(m.group(1))

        if "TP" in line:
            m = re.search(r"TP\s*=\s*([-+]?\d*\.?\d+)", line)
            if m:
                current["tp"] = float(m.group(1))

        if "Total" in line:
            m = re.search(r"Total\s*=\s*([-+]?\d*\.?\d+)", line)
            if m:
                current["objective"] = float(m.group(1))

        if "TIME_LIMIT" in line:
            current["status"] = "TIME_LIMIT"

        if "OPTIMAL" in line:
            current["status"] = "OPTIMAL"

    if current is not None:
        configs.append(current)

    df = pd.DataFrame(configs)

    print(f"Parsed configurations: {len(df)}")

    if len(df) > 0:
        print(df.head())

    return df


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------

def plot_cost_vs_tp(df, outdir):
    plt.figure()
    plt.scatter(df["cost"], df["tp"], s=10, alpha=0.5)

    plt.xlabel("Configuration cost")
    plt.ylabel("Operational performance (TP)")
    plt.title("Cost vs Operational Performance")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cost_vs_tp.png"), dpi=300)
    plt.close()


def plot_cost_vs_cmax(df, outdir):
    plt.figure()
    plt.scatter(df["cost"], df["cmax"], s=10, alpha=0.5)

    plt.xlabel("Configuration cost")
    plt.ylabel("Makespan (Cmax)")
    plt.title("Cost vs Makespan")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cost_vs_cmax.png"), dpi=300)
    plt.close()


def plot_cost_vs_objective(df, outdir):
    plt.figure()
    plt.scatter(df["cost"], df["objective"], s=10, alpha=0.5)

    plt.xlabel("Configuration cost")
    plt.ylabel("Objective")
    plt.title("Cost vs Objective")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cost_vs_objective.png"), dpi=300)
    plt.close()


def plot_objective_hist(df, outdir):
    plt.figure()
    plt.hist(df["objective"], bins=40)

    plt.xlabel("Objective")
    plt.ylabel("Count")
    plt.title("Objective distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "objective_hist.png"), dpi=300)
    plt.close()


def plot_status(df, outdir):
    plt.figure()

    counts = df["status"].value_counts()
    plt.bar(counts.index, counts.values)

    plt.title("Solver status")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "status.png"), dpi=300)
    plt.close()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main(logfile):
    outdir = "figures"
    os.makedirs(outdir, exist_ok=True)

    print("Parsing log...")
    df = parse_log(logfile)

    print(f"Parsed {len(df)} configurations")

    print("Generating plots...")

    plot_cost_vs_tp(df, outdir)
    plot_cost_vs_cmax(df, outdir)
    plot_cost_vs_objective(df, outdir)
    plot_objective_hist(df, outdir)
    plot_status(df, outdir)

    df.to_csv(os.path.join(outdir, "parsed_run.csv"), index=False)

    print("Done. Figures saved in /figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    args = parser.parse_args()

    main(args.logfile)