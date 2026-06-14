import streamlit as st
import networkx as nx
import seaborn as sns
import pandas as pd
import plotly.express as px
import numpy as np
from helpers import centralized_html_text
from components.cytoscape_graph.cytoscape_elements import (
    show_cytoscape_graph,
)
from analysis_helpers import (
    analyze_centrality,
    compute_community_metrics,
    opinion_group_metrics,
)
from services.data import load_data, build_graphs
from components.plotly_graphs import (
    plot_community_attribute_heatmap,
    plot_community_metric_bar,
    plot_community_opinion_distance_heatmap,
    plotly_group_distributions,
    plotly_group_mean_and_mad,
    plotly_opinion_polarization_map,
)
from matplotlib.transforms import blended_transform_factory

from components.community_graph import (
    compute_purity_for_many_organic_cols,
    plot_organic_alignment_summary,
)

CATEGORICAL = [
    "Speaker Party Simple",
    "Party Orientation",
    "Party Status",
    "Speaker Minister",
    "Speaker Mp",
    "Orientation Simple",
    "Louvain Community",
]
NUMERICAL = [
    "Opinion",
    "Nr Sentences",
    "Degree",
    "Closeness",
    "Degree Centrality",
    "Betweenness",
    "Pagerank",
    "Eigenvector",
]


def build_graph_sidebar():
    layout_name = st.sidebar.selectbox(
        "Layout",
        options=["fcose", "cose", "circle", "grid", "concentric", "breadthfirst"],
        index=0,
    )

    group_attribute = st.sidebar.selectbox(
        "Physically group nodes by",
        options=[None, *CATEGORICAL],
        index=0,
    )

    fill_color_attribute = st.sidebar.selectbox(
        "Node fill color from",
        options=[*NUMERICAL, *CATEGORICAL],
        index=0,
    )

    size_attribute = st.sidebar.selectbox(
        "Node size from",
        options=[*NUMERICAL],
        index=2,
    )

    border_color_attribute = st.sidebar.selectbox(
        "Node border color from",
        options=[None, *CATEGORICAL, *NUMERICAL],
        index=0,
    )

    border_width_attribute = st.sidebar.selectbox(
        "Node border width from",
        options=[None, *NUMERICAL],
        index=0,
    )

    show_node_labels = st.sidebar.checkbox("Show node labels", value=False)
    use_physical_groups = st.sidebar.checkbox(
        "Use physical compound groups", value=False
    )

    min_node_size = st.sidebar.slider("Minimum node size", 1, 80, 30)
    max_node_size = st.sidebar.slider("Maximum node size", 10, 180, 95)
    max_border_width = st.sidebar.slider("Maximum border width", 1, 14, 7)

    MAX_SPACING = 5.0
    MIN_SPACING = 0.01
    # group_spacing = (
    #     MAX_SPACING
    #     + MIN_SPACING
    #     - st.sidebar.slider("Group Spacing", MIN_SPACING, MAX_SPACING, 1.0)
    # )

    group_spacing = st.sidebar.slider("Group Spacing", MIN_SPACING, MAX_SPACING, 1.0)

    return {
        "layout_name": layout_name,
        "group_attribute": group_attribute,
        "fill_color_attribute": fill_color_attribute,
        "size_attribute": size_attribute,
        "border_color_attribute": border_color_attribute,
        "border_width_attribute": border_width_attribute,
        "show_node_labels": show_node_labels,
        "use_physical_groups": use_physical_groups,
        "min_node_size": min_node_size,
        "max_node_size": max_node_size,
        "max_border_width": max_border_width,
        "group_spacing": group_spacing,
    }


def build_cytoscape_graph(graph: nx.Graph, sidbar_settings: dict):
    selected = show_cytoscape_graph(graph, **sidbar_settings)
    return selected


def display_general(
    sim_matrix: np.ndarray,
    df: pd.DataFrame,
    graph: nx.Graph,
    edge_threshold: float = 0.99,
):
    kpis = st.columns(5)
    kpis[0].metric("Number of Nodes", graph.number_of_nodes())
    kpis[1].metric("Number of Edges", graph.number_of_edges())
    kpis[2].metric("Average Opinion Score", f"{df['Opinion'].mean():.2f}")
    kpis[3].metric(
        "Number of Connected Components", nx.number_connected_components(graph)
    )
    kpis[4].metric("Number of Isolated Nodes", nx.number_of_isolates(graph))

    cols = st.columns(2)
    with cols[0]:
        ax = sns.histplot(data=df, x="Degree", kde=True)
        ax.set(
            xlabel="Degree",
            title="Degree Distribution",
        )
        st.pyplot(ax.figure.figure)

        ax.figure.figure.clear()

    with cols[1]:
        # extract similarity scores
        sim_flattened = []
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix[i])):
                sim_flattened.append(sim_matrix[i, j])

        # plot the distribution
        ax = sns.histplot(data=sim_flattened, kde=True)

        ax.axvline(edge_threshold, linestyle="-", c="red")
        ax.set_title("Similarity Score Distrubition")
        ax.text(
            x=edge_threshold - 0.001,
            y=0.985,
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            s=f"Edge Creation Threshold p={edge_threshold:.2f}",
            va="top",
            ha="right",
        )

        st.pyplot(ax.figure.figure)
        ax.clear()


def display_centrality(graph: nx.Graph, df: pd.DataFrame):
    # get a few distributions of centralities
    cols = st.columns(2)
    metric_names = ["Degree Centrality", "Closeness", "Betweenness", "Eigenvector"]
    for i, metric_name in enumerate(metric_names):
        with cols[i % 2]:
            ax = sns.histplot(data=df, x=metric_name, kde=True)
            ax.set(
                xlabel=metric_name,
                title=f"{metric_name} Centrality Distribution",
            )
            st.pyplot(ax.figure.figure)

            ax.figure.figure.clear()

    corr = df[metric_names].corr()
    st.plotly_chart(
        px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Centrality Measure Correlation Heatmap",
        )
    )

    st.html(
        centralized_html_text(
            "Top 5 Members of Parliament by Centrality Measures", heading="h3"
        )
    )
    st.dataframe(analyze_centrality(graph))


def display_community_analysis(
    df: pd.DataFrame, group_col: str, score_col: str, g: nx.Graph
):
    metrics = opinion_group_metrics(df, group_col, score_col)

    st.plotly_chart(
        plotly_group_distributions(df, group_col, score_col), use_container_width=True
    )

    st.plotly_chart(
        plotly_opinion_polarization_map(metrics, group_col), use_container_width=True
    )

    st.plotly_chart(
        plotly_group_mean_and_mad(metrics, group_col), use_container_width=True
    )

    community_metrics = compute_community_metrics(df, g, group_col, score_col)
    # st.plotly_chart(plot_community_network(g, community_metrics))

    # st.plotly_chart(plot_community_opinion_map(community_metrics))
    st.plotly_chart(
        plot_community_opinion_distance_heatmap(community_metrics, group_col)
    )

    st.plotly_chart(
        plot_community_metric_bar(
            community_metrics, "Polarization", group_col=group_col
        )
    )


def display_community_alignment(df: pd.DataFrame, group_attribute: str):
    organic_cols = ["Speaker Party Simple", "Orientation Simple", "Party Status"]

    # make sure to not use community twice
    if group_attribute == "Louvain Community":
        group_attribute = "Speaker Party Simple"

    summary = compute_purity_for_many_organic_cols(
        df,
        community_col="Louvain Community",
        organic_cols=organic_cols,
    )

    st.plotly_chart(plot_organic_alignment_summary(summary))

    st.plotly_chart(
        plot_community_attribute_heatmap(
            df, community_col="Louvain Community", attribute_col=group_attribute
        )
    )

    st.plotly_chart(
        plot_community_attribute_heatmap(
            df, community_col=group_attribute, attribute_col="Louvain Community"
        )
    )


df_speech = load_data()
graph_datas, combined_data = build_graphs(df_speech)

graph_months = [
    i + 1
    for i in range(len(graph_datas))
    if graph_datas[i]["graph"].number_of_nodes() != 0
]
graph_month = st.selectbox("Graph Index", graph_months, index=1)

graph_data, graph_df = (
    graph_datas[graph_month - 1],
    combined_data[combined_data["Month"] == graph_month],
)

graph, sim_matrix, metrics, start_date, end_date = (
    graph_data["graph"],
    graph_data["sim_matrix"],
    graph_data["metrics"],
    graph_data["start_date"],
    graph_data["end_date"],
)

st.html(
    centralized_html_text(
        f"Analyzing Data From Period {start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}"
    )
)


sidebar_settings = build_graph_sidebar()
build_cytoscape_graph(graph, sidebar_settings)


tabs = st.tabs(
    [
        "General Information",
        "Centrality Measures",
        "Community Analysis",
        "Community Alignment",
    ]
)

with tabs[0]:
    display_general(sim_matrix, graph_df, graph)

with tabs[1]:
    display_centrality(graph, graph_df)


group_col = sidebar_settings["group_attribute"] or "Speaker Party Simple"

with tabs[2]:
    display_community_analysis(graph_df, group_col, "Opinion", graph)

with tabs[3]:
    display_community_alignment(graph_df, group_col)
