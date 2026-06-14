import networkx as nx
from networkx.algorithms import community
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def top_n(centrality_dict, n):
    return [
        f"{x[0]} - {x[1]:.2f}"
        for x in sorted(
            [(k, v) for k, v in centrality_dict.items()],
            reverse=True,
            key=lambda x: x[1],
        )[:n]
    ]


# analyze the most important nodes in a graph and display the information
def analyze_centrality(g: nx.Graph) -> pd.DataFrame:
    # get the largest connected component from the graph
    largest_cc = max(nx.connected_components(g), key=len)
    S = g.subgraph(largest_cc).copy()
    # calculate the four centralities
    degree = nx.degree_centrality(S)
    betweenness = nx.betweenness_centrality(S)
    closeness = nx.closeness_centrality(S)
    eigenvector = nx.eigenvector_centrality(S)
    # get the top n values
    N = 6
    degree = top_n(degree, N)
    betweenness = top_n(betweenness, N)
    closeness = top_n(closeness, N)
    eigenvector = top_n(eigenvector, N)

    return pd.DataFrame(
        {
            "Degree": degree,
            "Betweenness": betweenness,
            "Closeness": closeness,
            "Eigenvector": eigenvector,
        }
    )


# ChatGPT
def jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)


def match_communities(prev_comms, curr_comms, threshold=0.3):
    matches = {}

    for i, c1 in prev_comms.items():
        best_match = None
        best_score = 0

        for j, c2 in curr_comms.items():
            score = jaccard(set(c1), set(c2))
            if score > best_score:
                best_score = score
                best_match = j

        if best_score > threshold:
            matches[i] = best_match

    return matches


def mean_absolute_deviation(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return np.nan

    return float(np.mean(np.abs(x - x.mean())))


def average_pairwise_distance(x: pd.Series) -> float:
    """
    Average absolute distance between all pairs of members in range [0, 2] for values
    between [-1, 1], where 0 is complete agreement and 2 the oppositve.
    Tells you how "diverse" each group is.
    """

    x_np = np.sort(x.dropna().to_numpy())
    n = len(x_np)
    if n < 2:
        return 0

    weights = np.arange(n)
    pairwise_sum = np.sum((2 * weights - n + 1) * x_np)
    num_pairs = n * (n - 1) / 2
    return pairwise_sum / num_pairs


def opinion_group_metrics(
    df: pd.DataFrame,
    group_col: str | list[str] = "Speaker Party Simple",
    score_col: str = "Opinion",
):
    global_mean = df[score_col].mean()

    result = (
        df.groupby(group_col)[score_col]
        .agg(
            n="count",
            mean_opinion="mean",
            median_opinion="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mad=mean_absolute_deviation,
            Polarization=average_pairwise_distance,
        )
        .reset_index()
    )

    result["iqr"] = result["q75"] - result["q25"]
    result["delta_vs_population"] = result["mean_opinion"] - global_mean

    result["polarization_normalized"] = result["Polarization"] / 2

    return result.sort_values("mean_opinion")


def pairwise_group_tests(
    df: pd.DataFrame,
    group_col: str | list[str] = "Speaker Party Simple",
    score_col: str = "aggregated_opinion",
) -> pd.DataFrame:

    rows = []
    groups = df[group_col]

    for g1, g2 in combinations(groups, 2):
        x1 = df.loc[df[group_col] == g1, score_col].dropna()
        x2 = df.loc[df[group_col] == g2, score_col].dropna()

        if len(x1) == 0 or len(x2) == 0:
            continue

        # TODO: Before I use this, understand it
        stat, p_value = mannwhitneyu(x1, x2, alternative="two-sided")

        rows.append(
            {
                "group_1": g1,
                "group_2": g2,
                "mean_1": x1.mean(),
                "mean_2": x2.mean(),
                "mean_difference": x1.mean() - x2.mean(),
                "mannwitney_u": stat,
                "p_value": p_value,
            }
        )

    results = pd.DataFrame(rows)
    results["p_value_fdr"] = multipletests(results["p_value"], method="fdr_bh")[1]

    return results


def opinion_time_metrics(
    df: pd.DataFrame,
    time_col: str = "Month",
    group_col: str | list[str] = "Speaker Party Simple",
    score_col: str = "aggregated_opinion",
):
    global_by_time = (
        df.groupby(time_col)[score_col]
        .mean()
        .rename("population_mean_at_T")
        .reset_index()
    )

    group_by = (
        [time_col, group_col] if isinstance(group_col, str) else [time_col, *group_col]
    )

    result = (
        df.groupby(group_by)[score_col]
        .agg(
            n="count",
            mean_opinion="mean",
            median_opinion="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mad=mean_absolute_deviation,
            Polarization=average_pairwise_distance,
        )
        .reset_index()
    )

    result = result.merge(global_by_time, on=time_col, how="left")

    result["iqr"] = result["q75"] - result["q25"]
    result["delta_vs_population_at_T"] = (
        result["mean_opinion"] - result["population_mean_at_T"]
    )

    result["polarization_normalized"] = result["Polarization"] / 2

    group_by = (
        [group_col, time_col] if isinstance(group_col, str) else [*group_col, time_col]
    )

    return result.sort_values(group_by)


def compute_community_metrics(
    df: pd.DataFrame,
    G: nx.Graph,
    community_col: str = "Louvain Community",
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Computes opinion and network metrics per detected community.
    """

    rows = []

    for community_id, cdf in df.groupby(community_col):
        community_nodes = [
            node
            for node, data in G.nodes(data=True)
            if data.get(community_col) == community_id
        ]

        subgraph = G.subgraph(community_nodes)

        scores = cdf[score_col]

        rows.append(
            {
                community_col: community_id,
                "size": len(cdf),
                "mean_opinion": scores.mean(),
                "median_opinion": scores.median(),
                "mad": mean_absolute_deviation(scores),
                "Polarization": average_pairwise_distance(scores),
                "density": nx.density(subgraph) if len(subgraph) > 1 else 0,
                "avg_degree": (
                    np.mean([d for _, d in subgraph.degree()])
                    if len(subgraph) > 0
                    else 0
                ),
                "avg_weighted_degree": (
                    np.mean([d for _, d in subgraph.degree(weight="Weight")])
                    if len(subgraph) > 0
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_community_purity(
    df, community_col="Louvain Community", organic_col="Speaker Party Simple"
):
    table = pd.crosstab(df[community_col], df[organic_col])

    purity = table.max(axis=1) / table.sum(axis=1)

    return purity.reset_index(name=f"purity_vs_{organic_col}")
