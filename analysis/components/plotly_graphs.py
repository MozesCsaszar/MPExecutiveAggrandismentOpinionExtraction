import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

OPINION_DISPLAY_RANGE = [-1.05, 1.05]
POLARIZATION_DISPLAY_RANGE = [-0.05, 2]


def get_community_label(community_col: str):
    return "Louvain Community" if community_col == "Community" else community_col


def plotly_group_distributions(df, group_col="group", score_col="score"):
    group_label = get_community_label(group_col)

    fig = px.violin(
        df,
        x=group_col,
        y=score_col,
        box=True,
        points="all",
        hover_data=df.columns,
        title=f"Opinion Distribution by {group_label}",
    )

    fig.add_hline(y=0, line_dash="dash")

    fig.update_yaxes(
        range=OPINION_DISPLAY_RANGE,
        title="Opinion score (-1 negative, 0 neutral, +1 positive)",
    )

    fig.update_xaxes(title=group_label)

    return fig


def plotly_group_mean_and_mad(metrics, group_col="group"):
    plot_df = metrics.sort_values("mean_opinion")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df[group_col],
            y=plot_df["mean_opinion"],
            mode="markers",
            error_y=dict(type="data", array=plot_df["mad"], visible=True),
            text=plot_df[group_col],
            customdata=plot_df[["n", "median_opinion", "Polarization"]],
            hovertemplate=(
                "Group: %{text}<br>"
                "Mean: %{y:.3f}<br>"
                "MAD: %{error_y.array:.3f}<br>"
                "n: %{customdata[0]}<br>"
                "Median: %{customdata[1]:.3f}<br>"
                "Polarization: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        title="Mean Opinion with Mean Absolute Deviation",
        xaxis_title=group_col,
        yaxis_title="Mean opinion ± MAD",
    )

    fig.update_yaxes(range=OPINION_DISPLAY_RANGE)

    return fig


def plotly_group_polarization(metrics, group_col="group"):
    plot_df = metrics.sort_values("Polarization")

    fig = px.bar(
        plot_df,
        x=group_col,
        y="Polarization",
        hover_data=["n", "mean_opinion", "mad", "polarization_normalized"],
        title="Group Polarization",
    )

    fig.update_yaxes(
        range=POLARIZATION_DISPLAY_RANGE, title="Average pairwise opinion distance"
    )

    group_label = get_community_label(group_col)

    fig.update_xaxes(title=group_label)

    return fig


def plotly_opinion_polarization_map(metrics, group_col="group"):
    fig = px.scatter(
        metrics,
        x="mean_opinion",
        y="Polarization",
        size="n",
        text=group_col,
        hover_data=[group_col, "n", "median_opinion", "mad", "delta_vs_population"],
        title=f"{group_col} Average Opinion vs Polarization",
    )

    fig.add_vline(x=0, line_dash="dash")

    fig.update_traces(textposition="top center")

    fig.update_xaxes(range=OPINION_DISPLAY_RANGE, title="Mean opinion")

    fig.update_yaxes(range=POLARIZATION_DISPLAY_RANGE, title="Polarization")

    return fig


def plotly_mean_opinion_over_time(ts_metrics, time_col="time", group_col="group"):
    fig = px.scatter(
        ts_metrics,
        x=time_col,
        y="mean_opinion",
        color=group_col,
        size="n",
        hover_data=[
            "n",
            "median_opinion",
            "mad",
            "Polarization",
            "delta_vs_population_at_T",
        ],
        title=f"Mean Opinion Over Time by {group_col}",
    )

    fig.update_traces(mode="lines+markers")

    fig.add_hline(y=0, line_dash="dash")

    fig.update_yaxes(range=OPINION_DISPLAY_RANGE, title="Mean opinion")

    fig.update_xaxes(title="Time")

    return fig


def plotly_polarization_over_time(ts_metrics, time_col="time", group_col="group"):

    fig = px.scatter(
        ts_metrics,
        x=time_col,
        y="Polarization",
        color=group_col,
        size="n",
        title=f"Group Polarization Over Time by {group_col}",
    )

    fig.update_traces(mode="lines+markers")

    fig.update_yaxes(
        range=POLARIZATION_DISPLAY_RANGE, title="Average pairwise opinion distance"
    )

    fig.update_xaxes(title="Time")

    return fig


def plot_divergence_from_population(
    ts_metrics: pd.DataFrame,
    time_col: str = "time",
    group_col: str = "group",
):
    fig = px.scatter(
        ts_metrics,
        x=time_col,
        y="delta_vs_population_at_T",
        size="n",
        color=group_col,
        hover_data=[
            "mean_opinion",
            "population_mean_at_T",
            "Polarization",
            "mad",
            "n",
        ],
        title=f"Group Divergence From Population Over Time by {group_col}",
    )

    fig.update_traces(mode="lines+markers")

    fig.add_hline(y=0, line_dash="dash")

    fig.update_yaxes(range=[-2, 2], title="Group mean - population mean")

    fig.update_xaxes(title="Time")

    return fig


def plot_community_metric_bar(
    metrics: pd.DataFrame, metric: str, title: str | None = None, group_col: str = ""
):
    plot_df = metrics.sort_values(metric)

    group_label = get_community_label(group_col)

    fig = px.bar(
        plot_df,
        x=group_col,
        y=metric,
        hover_data=[
            "size",
            "mean_opinion",
            "Polarization",
            "density",
            "avg_degree",
        ],
        title=title or f"Community Ranking by {metric}",
    )

    fig.update_layout(
        xaxis_title=group_label,
        yaxis_title=metric,
    )

    return fig


def compute_community_opinion_distance_matrix(metrics: pd.DataFrame, group_col: str):
    means = metrics.set_index(group_col)["mean_opinion"]

    communities = means.index.tolist()

    matrix = pd.DataFrame(
        index=communities,
        columns=communities,
        dtype=float,
    )

    for c1 in communities:
        for c2 in communities:
            matrix.loc[c1, c2] = abs(means.loc[c1] - means.loc[c2])

    return matrix


def plot_community_opinion_distance_heatmap(ts_metrics: pd.DataFrame, group_col: str):
    matrix = compute_community_opinion_distance_matrix(ts_metrics, group_col)

    group_label = get_community_label(group_col)

    fig = px.imshow(
        matrix,
        color_continuous_scale="Viridis",
        zmin=0,
        zmax=2,
        aspect="auto",
        title=f"{group_label} Opinion Distance Matrix",
    )

    fig.update_layout(
        xaxis_title=group_label,
        yaxis_title=group_label,
        coloraxis_colorbar_title="Opinion distance",
    )

    return fig


def plot_community_attribute_heatmap(
    df: pd.DataFrame,
    community_col: str,
    attribute_col: str,
    normalize: bool = True,
):
    table = pd.crosstab(
        df[community_col],
        df[attribute_col],
        normalize="index" if normalize else False,
    )

    community_label = get_community_label(community_col)
    attribute_label = get_community_label(attribute_col)

    fig = px.imshow(
        table,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"{community_label} Composition by {attribute_label}",
    )

    fig.update_layout(
        xaxis_title=attribute_label,
        yaxis_title=community_label,
        coloraxis_colorbar_title="Share" if normalize else "Count",
    )

    return fig
