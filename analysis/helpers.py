import networkx as nx
from networkx.algorithms import community
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer
import os
import re

tqdm.pandas()


def humanize_text(text: str):
    return re.sub(r"_+", " ", text).title()


def centralized_html_text(text: str, heading: str = "h2"):
    return f"<{heading} style='text-align:center;'>{text}</{heading}>"


def embed_texts(texts: list[str], model):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # important for cosine similarity
    )


def embed_text(text: str, model):
    return embed_texts([text], model)


def aggregate_mean(embeddings):
    return np.mean(embeddings, axis=0)


def get_average_label_value(labels: list[Literal["PRO", "NEUTRAL", "CONTRA"]]):
    values_dict = {"PRO": 1, "NEUTRAL": 0, "CONTRA": -1}
    values = [values_dict[label] for label in labels]
    nr_non_neutral = sum(map(abs, values))
    divisor = 1 if nr_non_neutral == 0 else nr_non_neutral
    return sum(values) / divisor


def get_opinion_color(labels: list[Literal["PRO", "NEUTRAL", "CONTRA"]]):
    # custom node coloring
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_grey_red", ["#00AA00", "#BBBBBB", "#CC0000"]
    )
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # get the color
    average_label_value = get_average_label_value(labels)
    rgba = cmap(norm(average_label_value))
    hex_color = mcolors.to_hex(rgba)
    return hex_color


def get_simple_orientation(orientations: pd.Series):
    mapper = {
        "Right to far-right": "Right",
        "Centre-right": "Center",
        "Far-right": "Right",
        "Right": "Right",
        "Centre-left": "Center",
        "-": "-",
        "Centre-left to left": "Left",
        "Centre to centre-left": "Center",
    }
    return mapper[orientations.iloc[0]]


def create_embeddings(
    df: pd.DataFrame,
    # alternatives: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # alternatives: "sentence-transformers/LaBSE"
    model_name: str = "intfloat/multilingual-e5-base",
    *,
    save_on_complete: bool = True,
    save_suffix: str = "",
):
    model = SentenceTransformer(model_name)

    # fill empty text fields
    df["Text"] = df["Text"].fillna("")
    # do the embedding
    df["embedding"] = df["Text"].progress_apply(lambda x: embed_text(x, model))

    if save_on_complete:
        df.to_csv(f"../outputs/embedded_{save_suffix}.csv")

    return df


def load_data(suffix: str = "2020"):
    # read the data in
    df = (
        pd.read_csv(
            f"../outputs/embedded_{suffix}.csv",
            dtype={
                "Speaker_minister": "bool",
                "Speaker_MP": "bool",
            },
            parse_dates=["Date"],
        )
        .set_index(["Speaker_ID", "Date", "ID"])
        .sort_index()
    )

    # get the embeddings back into numeric format
    df["embedding"] = df["embedding"].apply(
        lambda s: np.fromstring(s.strip("[]"), sep=" ", dtype=float).reshape(1, -1)  # type: ignore
    )  # type: ignore

    # Get a simplified party label
    df["speaker_party_simple"] = df["Speaker_party"].apply(
        lambda x: x.strip("-frakció")
    )

    return df


def compute_all_metrics(g: nx.Graph):
    metrics = {}

    metrics["degree"] = dict(g.degree())

    # centrality measures
    metrics["degree_centrality"] = nx.degree_centrality(g)
    metrics["closeness"] = nx.closeness_centrality(g)
    metrics["betweenness"] = nx.betweenness_centrality(g)
    metrics["pagerank"] = nx.pagerank(g)
    metrics["eigenvector"] = nx.eigenvector_centrality(g, max_iter=1000)

    # community measures
    if g.number_of_edges() > 0:
        comms = community.louvain_communities(g, seed=42)
        metrics["communities"] = {n: i for i, c in enumerate(comms) for n in c}
        metrics["modularity"] = community.modularity(g, comms)
    else:
        metrics["communities"] = {}
        metrics["modularity"] = {}

    return metrics


def add_time_derivatives(
    ts_metrics: pd.DataFrame,
    time_col: str = "time",
    group_col: str = "group",
    value_col: str = "mean_opinion",
) -> pd.DataFrame:
    df = ts_metrics.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([group_col, time_col])

    df[f"{value_col}_velocity"] = df.groupby(group_col)[value_col].diff()
    df[f"{value_col}_acceleration"] = df.groupby(group_col)[
        f"{value_col}_velocity"
    ].diff()

    return df


def add_network_metrics(g: nx.Graph):
    metrics = compute_all_metrics(g)

    # attach info to individual nodes
    for node in g.nodes:
        g.nodes[node]["degree"] = metrics["degree"].get(node, 0)
        g.nodes[node]["closeness"] = metrics["degree"].get(node, 0)
        g.nodes[node]["degree_centrality"] = metrics["degree_centrality"].get(node, 0)
        g.nodes[node]["betweenness"] = metrics["betweenness"].get(node, 0)
        g.nodes[node]["pagerank"] = metrics["pagerank"].get(node, 0)
        g.nodes[node]["eigenvector"] = metrics["eigenvector"].get(node, 0)
        g.nodes[node]["louvain_community"] = metrics["communities"].get(node, 0)

    return g, metrics


# build graph
def build_graph(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    similarity_threshold: float = 0.99,
):
    # extract data from the specified timeframe
    data = df.loc[(slice(None), slice(start_date, end_date)), :]

    # if there is no relevant data, return None
    if len(data) == 0:
        return {
            "graph": nx.Graph(),
            "sim_matrix": [],
            "actors": [],
            "metrics": [],
        }

    # get speech groups
    df_group = (
        data.groupby(
            [
                "Speaker_name",
                "Speaker_ID",
                "Party_orientation",
                "Party_status",
                "speaker_party_simple",
                "Speaker_minister",
                "Speaker_MP",
            ],
        )
        .agg(
            aggregated_embedding=("embedding", aggregate_mean),
            opinion=("label", get_average_label_value),
            opinion_color=("label", get_opinion_color),
            nr_sentences=("label", len),
            orientation_simple=("Party_orientation", get_simple_orientation),
        )
        .reset_index()
        .set_index("Speaker_ID")
    )

    # calculate similarities
    actors = list(df_group.index)
    matrix = np.vstack(list(df_group["aggregated_embedding"]))

    sim_matrix = cosine_similarity(matrix)

    # create the network
    G = nx.Graph()

    # create the nodes
    nodes = zip(
        df_group.index,
        df_group.drop(columns=["aggregated_embedding"]).to_dict("records"),
    )
    G.add_nodes_from(nodes)

    # create the edges
    edges = []
    for i, a1 in enumerate(actors):
        for j, a2 in enumerate(actors):
            if i > j and sim_matrix[i, j] >= similarity_threshold:
                edges.append((a1, a2, {"weight": sim_matrix[i, j]}))
    G.add_edges_from(edges)

    # add in the network metrics
    G, metrics = add_network_metrics(G)

    # return the graph, the similarities and the list of actors
    return {
        "graph": G,
        "sim_matrix": sim_matrix,
        "actors": actors,
        "metrics": metrics,
    }


# build multiple graphs
def build_graphs(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    start_offset: pd.offsets.DateOffset,
    end_offset: pd.offsets.DateOffset,
) -> tuple[list[dict], pd.DataFrame]:
    results = []
    graph_start_date = start_date
    graph_end_date = start_date + end_offset
    while graph_end_date < end_date:
        results.append(
            {
                "start_date": graph_start_date,
                "end_date": graph_end_date,
                **build_graph(df, graph_start_date, graph_end_date),
            }
        )
        graph_start_date = graph_end_date + start_offset
        graph_end_date = graph_start_date + end_offset

    # get the combined data from the nodes
    combined_data = (
        pd.concat(
            [
                pd.DataFrame.from_dict(
                    dict(r["graph"].nodes(data=True)), orient="index"
                )
                for r in results
            ],
            keys=[r["start_date"].month for r in results],
            names=["month"],
        )
        .reset_index(level="month")
        .reset_index(drop=True)
    )

    return (results, combined_data)
