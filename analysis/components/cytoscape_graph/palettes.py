DISTINCT_20_COLORS: list[str] = [
    "#1F77B4",  # blue
    "#D62728",  # red
    "#2CA02C",  # green
    "#FF7F0E",  # orange
    "#9467BD",  # purple
    "#8C564B",  # brown
    "#E377C2",  # pink
    "#17BECF",  # cyan
    "#BCBD22",  # olive
    "#7F7F7F",  # gray
    "#005F73",  # deep teal
    "#9B2226",  # dark red
    "#0A9396",  # teal
    "#CA6702",  # burnt orange
    "#6A4C93",  # violet
    "#588157",  # muted green
    "#B56576",  # rose
    "#3A86FF",  # bright blue
    "#FB5607",  # vivid orange
    "#8338EC",  # vivid purple
]

DEFAULT_MISSING_COLOR = "#BDC3C7"
DEFAULT_NODE_FILL_COLOR = DEFAULT_MISSING_COLOR
DEFAULT_NODE_BORDER_COLOR = "#222222"
DEFAULT_NODE_SIZE = 35
DEFAULT_NODE_BORDER_WIDTH = 0


CUSTOM_CATEGORY_COLORS: dict[str, dict[str, str]] = {
    # "party": {
    #     "Independent": "#7F7F7F",
    #     # "PSD": "#D62728",
    #     # "PNL": "#1F77B4",
    #     # "USR": "#F39C12",
    # },
    # "bloc": {
    #     "Government": "#1B4F72",
    #     "Opposition": "#7B241C",
    #     "Minority": "#145A32",
    # },
    # "role": {
    #     "Leader": "#111111",
    #     "Minister": "#6C3483",
    #     "MP": "#566573",
    #     "Journalist": "#117A65",
    #     "Lobbyist": "#BA4A00",
    # },
    # "Community": {
    #     # Example if you run Louvain/Leiden/community detection.
    #     # "0": "#1F77B4",
    #     # "1": "#D62728",
    # },
}

# ---------------------------------------------------------------------
# Visual encoding configuration
# ---------------------------------------------------------------------

CATEGORICAL_NODE_ATTRIBUTES = [
    "Speaker Name",
    "Party Orientation",
    "Party Status",
    "Speaker Party Simple",
    "Speaker Minister",
    "Speaker Mp",
    "Orientation Simple",
    "Community",
]
NUMERIC_NODE_ATTRIBUTES = [
    "Opinion",
    "Nr Sentences",
    "Degree",
    "Closeness",
    "Degree Centrality",
    "Betweenness",
    "Pagerank",
    "Eigenvector",
]

# Sequential scale for normal numeric quantities.
# Low value -> light color.
# High value -> dark color.
SEQUENTIAL_COLOR_SCALE = {
    "low": "#D6EAF8",
    "high": "#154360",
}

# Diverging scale for signed values such as ideology.
# Negative -> blue.
# Neutral -> near white.
# Positive -> red.
DIVERGING_COLOR_SCALE = {
    "low": "#00AA00",
    "mid": "#BBBBBB",
    "high": "#B2182B",
}
