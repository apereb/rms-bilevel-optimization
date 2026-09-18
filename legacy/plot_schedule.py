import json
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def load_best_solution(json_file, config_idx=1563):
    with open(json_file, "r") as f:
        data = json.load(f)

    corridor = data["summary"]["corridor_detail"]

    best = next(
        (c for c in corridor if c.get("config_idx") == config_idx),
        None
    )

    if best is None:
        raise ValueError(
            f"Config {config_idx} not found in corridor_detail — "
            "its schedule may not have been saved. It would need "
            "to be re-solved with Gurobi to recover the Gantt data."
        )

    df = pd.DataFrame(best["activity"])
    df = df[df["end"] > df["start"]].copy()
    return df


def build_machine_gantt(df, output_html="machines.html"):

    df = df.copy()

    df["resource"] = (
        "S" + df["stage"].astype(str)
        + "-M" + df["machine"].astype(str)
    )

    resources = sorted(
        df["resource"].unique(),
        key=lambda x: (
            int(x.split("-")[0][1:]),
            int(x.split("-")[1][1:])
        )
    )

    colors = px.colors.qualitative.Set1

    type_colors = {}
    for i, t in enumerate(sorted(df["job_id"].unique())):
        type_colors[t] = colors[i % len(colors)]

    fig = go.Figure()

    for _, row in df.iterrows():

        duration = row["end"] - row["start"]

        fig.add_trace(
            go.Bar(
                x=[duration],
                y=[row["resource"]],
                base=row["start"],
                orientation="h",

                marker_color=type_colors[row["job_id"]],

                text=f"J{row['job_index']}",
                textposition="inside",

                customdata=[[
                    row["job_index"],
                    row["job_id"],
                    row["stage"],
                    row["machine"],
                    row["start"],
                    row["end"]
                ]],

                hovertemplate=
                "Job %{customdata[0]}<br>"
                "Type %{customdata[1]}<br>"
                "Stage %{customdata[2]}<br>"
                "Machine %{customdata[3]}<br>"
                "Start %{customdata[4]:.3f}<br>"
                "End %{customdata[5]:.3f}"
                "<extra></extra>",

                showlegend=False
            )
        )

    fig.update_layout(
        title="Schedule per machine",
        barmode="overlay",
        xaxis_title="Time",
        yaxis_title="Machine",
        height=max(600, 50 * len(resources))
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(resources))
    )

    fig.write_html(output_html)

    print(f"Saved {output_html}")


def build_job_gantt(df, output_html="jobs.html"):

    df = df.copy()

    df["job_label"] = (
        "J" + df["job_index"].astype(str)
        + " (T" + df["job_id"].astype(str) + ")"
    )

    jobs = sorted(df["job_index"].unique())

    job_labels = [
        f"J{j} (T{df[df['job_index']==j]['job_id'].iloc[0]})"
        for j in jobs
    ]

    colors = px.colors.qualitative.Set2

    stage_colors = {}
    for s in sorted(df["stage"].unique()):
        stage_colors[s] = colors[s % len(colors)]

    fig = go.Figure()

    for _, row in df.iterrows():

        duration = row["end"] - row["start"]

        fig.add_trace(
            go.Bar(
                x=[duration],
                y=[row["job_label"]],
                base=row["start"],
                orientation="h",

                marker_color=stage_colors[row["stage"]],

                text=f"S{row['stage']}",
                textposition="inside",

                customdata=[[
                    row["job_index"],
                    row["job_id"],
                    row["stage"],
                    row["machine"],
                    row["start"],
                    row["end"]
                ]],

                hovertemplate=
                "Job %{customdata[0]}<br>"
                "Type %{customdata[1]}<br>"
                "Stage %{customdata[2]}<br>"
                "Machine %{customdata[3]}<br>"
                "Start %{customdata[4]:.3f}<br>"
                "End %{customdata[5]:.3f}"
                "<extra></extra>",

                showlegend=False
            )
        )

    fig.update_layout(
        title="Schedule per job",
        barmode="overlay",
        xaxis_title="Time",
        yaxis_title="Job",
        height=max(700, 35 * len(job_labels))
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(job_labels))
    )

    fig.write_html(output_html)

    print(f"Saved {output_html}")


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("python plot_schedule.py result.json")
        sys.exit(1)

    json_file = sys.argv[1]

    df = load_best_solution(json_file)

    print(df.head())

    build_machine_gantt(df)
    build_job_gantt(df)


if __name__ == "__main__":
    main()