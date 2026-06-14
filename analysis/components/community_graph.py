import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def build_community_graph(G: nx.Graph) -> nx.Graph:
    CG = nx.Graph()

    for node, data in G.nodes(data=True):
        c = data.get("Community")

        if c not in CG:
            CG.add_node(c, size=0)

        CG.nodes[c]["size"] += 1

    for u, v, data in G.edges(data=True):
        cu = G.nodes[u].get("Community")
        cv = G.nodes[v].get("Community")

        weight = data.get("Weight", 1)

        if cu == cv:
            continue

        if CG.has_edge(cu, cv):
            CG[cu][cv]["Weight"] += weight
            CG[cu][cv]["count"] += 1
        else:
            CG.add_edge(cu, cv, weight=weight, count=1)

    return CG


def plot_community_network(
    G: nx.Graph,
    metrics: pd.DataFrame,
    seed: int = 42,
):
    CG = build_community_graph(G)

    metric_map = metrics.set_index("Community").to_dict("index")

    for node in CG.nodes():
        if node in metric_map:
            CG.nodes[node].update(metric_map[node])

    pos = nx.spring_layout(CG, weight="Weight", seed=seed)

    edge_x = []
    edge_y = []
    edge_text = []

    for u, v, data in CG.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        edge_text.append(
            f"{u} ↔ {v}<br>"
            f"Cross edges: {data.get('count', 0)}<br>"
            f"Weight: {data.get('weight', 0)}"
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
        showlegend=False,
    )

    node_x = []
    node_y = []
    node_size = []
    node_color = []
    node_text = []

    for node, data in CG.nodes(data=True):
        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)
        node_size.append(20 + 4 * data.get("size", 1))
        node_color.append(data.get("mean_opinion", 0))

        node_text.append(
            f"Community: {node}<br>"
            f"Size: {data.get('size')}<br>"
            f"Mean opinion: {data.get('mean_opinion', np.nan):.3f}<br>"
            f"Polarization: {data.get('polarization', np.nan):.3f}<br>"
            f"Density: {data.get('density', np.nan):.3f}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n) for n in CG.nodes()],
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="RdBu",
            cmin=-1,
            cmax=1,
            showscale=True,
            colorbar=dict(title="Mean opinion"),
            line=dict(width=1),
        ),
        textposition="middle center",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Community-Level Network",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=650,
    )

    return fig


def compute_community_transitions(
    df: pd.DataFrame,
    time_col: str = "Month",
    member_col: str = "member_id",
    community_col: str = "Community",
):
    data = df[[time_col, member_col, community_col]].drop_duplicates()
    data = data.sort_values([member_col, time_col])

    rows = []

    for member, mdf in data.groupby(member_col):
        mdf = mdf.sort_values(time_col)

        prev = None

        for _, row in mdf.iterrows():
            current = row

            if prev is not None:
                rows.append(
                    {
                        "source_time": prev[time_col],
                        "target_time": current[time_col],
                        "source_community": prev[community_col],
                        "target_community": current[community_col],
                        "member_id": member,
                    }
                )

            prev = current

    transitions = (
        pd.DataFrame(rows)
        .groupby(
            [
                "source_time",
                "target_time",
                "source_community",
                "target_community",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    return transitions


def plot_community_sankey(
    transitions: pd.DataFrame, *, community_name: str = "Community"
):
    labels = []

    for _, row in transitions.iterrows():
        labels.append(f"{row['source_time']} | C-{row['source_community']}")
        labels.append(f"{row['target_time']} | C-{row['target_community']}")

    labels = list(dict.fromkeys(labels))
    label_to_id = {label: i for i, label in enumerate(labels)}

    sources = []
    targets = []
    values = []

    for _, row in transitions.iterrows():
        source_label = f"{row['source_time']} | C-{row['source_community']}"
        target_label = f"{row['target_time']} | C-{row['target_community']}"

        sources.append(label_to_id[source_label])
        targets.append(label_to_id[target_label])
        values.append(row["count"])

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=labels,
                    pad=15,
                    thickness=15,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"{community_name} Evolution Sankey Diagram",
        height=750,
    )

    return fig


def compute_community_overlap_table(
    df: pd.DataFrame, community_col: str, organic_col: str, normalize=False
):
    return pd.crosstab(df[community_col], df[organic_col], normalize=normalize)


# How much each detected Louvain community is composed of each organic category.
def plot_community_overlap_heatmap(
    df,
    community_col="Community",
    organic_col="department",
    normalize="index",
):

    table = compute_community_overlap_table(
        df,
        community_col=community_col,
        organic_col=organic_col,
        normalize=normalize,  # type: ignore
    )

    fig = px.imshow(
        table,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"{community_col} Overlap with {organic_col}",
    )

    fig.update_layout(
        xaxis_title=organic_col,
        yaxis_title="Louvain community",
        coloraxis_colorbar_title="Share" if normalize else "Count",
    )

    return fig


def compute_overall_purity(
    df,
    community_col="Community",
    organic_col="department",
):
    table = pd.crosstab(df[community_col], df[organic_col])
    return table.max(axis=1).sum() / table.values.sum()


# Normalized Mutual Information
# 0 = no relationship
# 1 = perfect alignment
# Adjusted Rand Index
# 0 = roughly random alignment
# 1 = perfect alignment
# negative = worse than random
def compare_community_assignments(
    df,
    community_col="Community",
    organic_col="department",
):
    data = df[[community_col, organic_col]].dropna()

    return {
        "normalized_mutual_information": normalized_mutual_info_score(
            data[community_col],
            data[organic_col],
        ),
        "adjusted_rand_index": adjusted_rand_score(
            data[community_col],
            data[organic_col],
        ),
    }


def compute_purity_for_many_organic_cols(
    df,
    community_col="Community",
    organic_cols: list[str] = [],
):
    rows = []

    for col in organic_cols:
        rows.append(
            {
                "ground_truth_community": col,
                "overall_purity": compute_overall_purity(
                    df,
                    community_col=community_col,
                    organic_col=col,
                ),
                **compare_community_assignments(
                    df,
                    community_col=community_col,
                    organic_col=col,
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_organic_alignment_summary(summary):
    fig = px.bar(
        summary,
        x="ground_truth_community",
        y=[
            "overall_purity",
            "normalized_mutual_information",
            "adjusted_rand_index",
        ],
        barmode="group",
        title="Alignment Between Louvain and Ground Truth Communities",
    )

    fig.update_layout(
        xaxis_title="Ground Truth Communities",
        yaxis_title="Alignment Score",
    )

    return fig
