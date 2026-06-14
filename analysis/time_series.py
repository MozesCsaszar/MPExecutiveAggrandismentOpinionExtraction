from components.community_graph import (
    compute_community_transitions,
    plot_community_sankey,
)
from analysis_helpers import opinion_time_metrics
from components.plotly_graphs import (
    plotly_mean_opinion_over_time,
    plotly_polarization_over_time,
    plot_divergence_from_population,
)
from services.data import load_data, build_graphs, load_vdem_data
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

df_speech = load_data()
graph_datas, combined_data = build_graphs(df_speech)
df_hd = load_vdem_data()

CATEGORICAL = [
    "Speaker Party Simple",
    "Party Orientation",
    "Party Status",
    "Speaker Minister",
    "Speaker Mp",
    "Orientation Simple",
    "Louvain Community",
]


def build_sidebar():
    opinion_grouping = st.sidebar.selectbox(
        "Opinion Grouping",
        options=[*CATEGORICAL],
        index=0,
    )

    second_opinion_grouping = st.sidebar.selectbox(
        "Secondary Opinion Grouping",
        options=[None, *CATEGORICAL],
        index=0,
    )

    return {
        "opinion_grouping": opinion_grouping,
        "second_opinion_grouping": second_opinion_grouping,
    }


def display_graph_metrics(graph_datas):
    cols = st.columns(2)
    data = []

    for i, graph_data in enumerate(graph_datas):
        g = graph_data["graph"]

        data.append(
            {
                "step": i,
                "nodes": g.number_of_nodes(),
                "edges": g.number_of_edges(),
                "isolated": len(list(nx.isolates(g))),
                "subgraphs": nx.number_connected_components(g),
            }
        )

    df_data = pd.DataFrame(data)

    df_melted = df_data.melt(
        id_vars="step",
        value_vars=["nodes", "isolated", "subgraphs"],
        var_name="metric",
        value_name="value",
    )

    with cols[0]:
        ax = sns.lineplot(data=df_melted, x="step", y="value", hue="metric")

        ax.set_title("Graph Node Metrics Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Nodes")

        st.pyplot(ax.figure.figure)
        plt.figure().clear()

    with cols[1]:
        # Display the number of edges
        ax = sns.lineplot(data=df_data, x="step", y="edges")

        ax.set_title("Number of Graph Edges Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Edges")

        st.pyplot(ax.figure.figure)
        plt.figure().clear()


def plot_vdem(df_hd: pd.DataFrame):
    ax = df_hd[
        [
            "v2x_polyarchy",
            "v2x_libdem",
            "v2x_partipdem",
            "v2x_delibdem",
            "v2x_egaldem",
        ]
    ][(df_hd.index > "2004-01-01")].plot.line()

    ax.set_xlabel("Historical Date")
    ax.set_ylabel("Indicator Score")
    ax.set_title("High-Level Democracy Indices from V-Dem")
    ax.set_ylim(0, 1)

    return ax


def display_general_information(
    graph_datas, sim_matrixes: list[list[list[float]]], df_hd
):
    # V-Dem graph
    ax = plot_vdem(df_hd)
    st.pyplot(ax.figure.figure)
    ax.figure.figure.clear()
    plt.figure().clear()

    cols = st.columns(2)

    with cols[0]:
        sim_flattened = []
        for k in range(len(sim_matrixes)):
            for i in range(len(sim_matrixes[k])):
                for j in range(i + 1, len(sim_matrixes[k][i])):
                    sim_flattened.append(sim_matrixes[k][i][j])

        # plot the distribution
        ax = sns.histplot(data=sim_flattened, kde=True)

        ax.axvline(0.99, linestyle="-", c="red")
        ax.set_title("Similarity Score Distrubition")
        ymax = ax.get_ylim()[1]
        ax.text(x=0.989, y=ymax * 0.95, s="Edge Creation Threshold p=0.99", ha="right")
        ax.set_xlabel("Similarity Score")

        st.pyplot(ax.figure.figure)
        ax.figure.figure.clear()

    with cols[1]:
        # Political Opinion Distribution
        ax = sns.histplot(data=combined_data, x="Opinion", kde=True)
        mean = combined_data["Opinion"].mean()

        ax.axvline(mean, linestyle="-", c="red")
        ax.set_title("EA Opinion Distrubition")
        ax.text(x=mean * 1.1, y=120, s=f"Average: {mean:.2f}", va="bottom")
        st.pyplot(ax.figure.figure)
        plt.figure().clear()

    display_graph_metrics(graph_datas)


#   political opinion evolution by different groupings
def display_opinion_evolution(
    combined_data, *, opinion_grouping: str = "Speaker Party Simple", **_
):
    ts_metrics = opinion_time_metrics(
        combined_data,
        time_col="Month",
        group_col=opinion_grouping,
        score_col="Opinion",
    )

    cols = st.columns(3)

    with cols[0]:
        ax = sns.lineplot(combined_data, x="Month", y="Opinion")
        ax.set_title("Mean Opinion Over Time")
        st.pyplot(ax.figure.figure)

        ax.figure.figure.clear()

    with cols[1]:
        ax = sns.lineplot(combined_data, x="Month", y="Opinion", hue="Party Status")
        ax.set_title("Mean Opinion by Party Status")
        st.pyplot(ax.figure.figure)

        ax.figure.figure.clear()

    with cols[2]:
        ax = sns.lineplot(
            combined_data, x="Month", y="Opinion", hue="Orientation Simple"
        )
        ax.set_title("Mean Opinion by Orientation Simple")
        st.pyplot(ax.figure.figure)

        ax.figure.figure.clear()

    st.plotly_chart(
        plotly_mean_opinion_over_time(
            ts_metrics, time_col="Month", group_col=opinion_grouping
        ),
        use_container_width=True,
    )

    cols = st.columns(2)

    with cols[0]:
        st.plotly_chart(
            plotly_polarization_over_time(
                ts_metrics, time_col="Month", group_col=opinion_grouping
            ),
            use_container_width=True,
        )

    with cols[1]:
        st.plotly_chart(
            plot_divergence_from_population(
                ts_metrics,
                time_col="Month",
                group_col=opinion_grouping,
            ),
            use_container_width=True,
        )


def plot_community_changes(
    combined_data: pd.DataFrame,
    *,
    time_col="Month",
    opinion_grouping="Speaker Party Simple",
    second_opinion_grouping: str | None = None,
    **_,
):
    transitions = compute_community_transitions(
        combined_data,
        time_col=time_col,
        member_col="Speaker Name",
        community_col=opinion_grouping,
    )

    st.plotly_chart(plot_community_sankey(transitions, community_name=opinion_grouping))

    if second_opinion_grouping:
        transitions = compute_community_transitions(
            combined_data,
            time_col=time_col,
            member_col="Speaker Name",
            community_col=second_opinion_grouping,
        )

        st.plotly_chart(
            plot_community_sankey(transitions, community_name=second_opinion_grouping)
        )


tabs = st.tabs(
    [
        "General Information",
        "Opinion Evolution",
        "Community Change Analysis",
    ]
)

sidebar_vars = build_sidebar()

with tabs[0]:
    display_general_information(
        graph_datas, [g["sim_matrix"] for g in graph_datas], df_hd
    )

with tabs[1]:
    display_opinion_evolution(combined_data, **sidebar_vars)

with tabs[2]:
    plot_community_changes(combined_data, **sidebar_vars)
