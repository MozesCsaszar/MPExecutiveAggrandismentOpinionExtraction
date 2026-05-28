import networkx as nx


def top_n(centrality_dict, n):
    return sorted(
        [(k, v) for k, v in centrality_dict.items()], reverse=True, key=lambda t: t[1]
    )[:n]


# analyze the most important nodes in a graph and display the information
def analyze_centrality(g: nx.Graph, graph_name: str):
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
    # put the measures into a single tuple
    centralities = (degree, betweenness, closeness, eigenvector)
    # dipsplay them in a table
    # display the header
    middle = (
        "of the largest connected component of"
        if len(list(nx.connected_components(g))) > 1
        else "of"
    )
    title = f"Centrality measures {middle} {graph_name}"
    print("{:^119s}".format(title))
    print(
        ("|".join(["{:^29s}"] * 4)).format(
            "Degree", "Betweenness", "Closeness", "Eigenvector"
        )
    )
    for i in range(N):
        # extract the values from the centralities tuple
        values = []
        for c in centralities:
            values.append(f"({c[i][0]})")
            values.append(c[i][1])
        print(" " + (" | ".join(["{:^16s} - {:.6f}"] * 4).format(*values)))
    print()
