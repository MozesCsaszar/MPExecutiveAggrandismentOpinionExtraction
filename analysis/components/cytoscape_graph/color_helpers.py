from __future__ import annotations

from typing import Any, Iterable
import hashlib
import networkx as nx
from .palettes import DISTINCT_20_COLORS, DEFAULT_MISSING_COLOR


def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Safely convert a value to float.

    Returns default if the value is missing, malformed, or non-numeric.
    """
    if value is None:
        return default

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return default

        # Common missing-value placeholders.
        if value.lower() in {"unknown", "unknow", "none", "nan", "null", "missing"}:
            return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value into [low, high]."""
    return max(low, min(high, value))


def normalize(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a numeric value into [0, 1].

    If all values are equal, return 0.5 so every item gets a midpoint color.
    """
    if max_value == min_value:
        return 0.5

    return (value - min_value) / (max_value - min_value)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' into an RGB tuple."""
    hex_color = hex_color.strip().lstrip("#")

    if len(hex_color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {hex_color}")

    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert an RGB tuple into '#RRGGBB'."""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def interpolate_color(
    value: float,
    low_color: str,
    high_color: str,
) -> str:
    """
    Interpolate between two hex colors.

    Parameters
    ----------
    value:
        A normalized value in [0, 1].
    low_color:
        Color used when value is 0.
    high_color:
        Color used when value is 1.
    """
    value = clamp(value, 0.0, 1.0)

    low_rgb = hex_to_rgb(low_color)
    high_rgb = hex_to_rgb(high_color)

    r = round(low_rgb[0] + value * (high_rgb[0] - low_rgb[0]))
    g = round(low_rgb[1] + value * (high_rgb[1] - low_rgb[1]))
    b = round(low_rgb[2] + value * (high_rgb[2] - low_rgb[2]))

    return rgb_to_hex((r, g, b))


def interpolate_diverging_color(
    value: float,
    min_value: float,
    midpoint: float,
    max_value: float,
    low_color: str = "#2166AC",
    mid_color: str = "#F7F7F7",
    high_color: str = "#B2182B",
) -> str:
    """
    Map a numeric value to a diverging color scale.

    Useful for variables such as:
    - ideology: left/right
    - sentiment: negative/positive
    - polarization: anti/pro
    - signed centrality scores

    Example:
        ideology = -1 -> blue
        ideology =  0 -> near white
        ideology = +1 -> red
    """
    value = clamp(value, min_value, max_value)

    if value <= midpoint:
        normalized = normalize(value, min_value, midpoint)
        return interpolate_color(normalized, low_color, mid_color)

    normalized = normalize(value, midpoint, max_value)
    return interpolate_color(normalized, mid_color, high_color)


def stable_hash_index(value: Any, modulo: int) -> int:
    """
    Deterministically map an arbitrary value to an index.

    This is useful if you do not want color assignments to depend on sorting order.
    """
    encoded = str(value).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    integer_value = int(digest[:12], 16)

    return integer_value % modulo


def get_unique_node_values(
    graph: nx.Graph,
    attribute: str,
    *,
    missing_value: str = "Unknown",
) -> list[str]:
    """Return sorted unique categorical values for a node attribute."""
    values = {
        str(attrs.get(attribute, missing_value)) for _, attrs in graph.nodes(data=True)
    }

    return sorted(values)


def build_categorical_color_map(
    values: Iterable[Any],
    *,
    palette: list[str],
    custom_colors: dict[Any, str] | None = None,
    missing_color: str = DEFAULT_MISSING_COLOR,
    deterministic_by_hash: bool = False,
) -> dict[str, str]:
    """
    Build a color map for categorical values.

    Parameters
    ----------
    values:
        Category values to map.
    palette:
        List of fallback colors.
    custom_colors:
        Optional manual overrides.
        Example:
            {"Government": "#1B4F72", "Opposition": "#7B241C"}
    missing_color:
        Color used for None or empty values.
    deterministic_by_hash:
        If True, each category gets a color by hash.
        If False, colors are assigned by sorted category order.

    Returns
    -------
    dict[str, str]
        Mapping from category value to hex color.
    """
    custom_colors = custom_colors or {}

    normalized_values = [
        "Unknown" if value is None or str(value).strip() == "" else str(value)
        for value in values
    ]

    unique_values = sorted(set(normalized_values))
    color_map: dict[str, str] = {}

    for index, value in enumerate(unique_values):
        # Custom color overrides always win.
        if value in custom_colors:
            color_map[value] = custom_colors[value]
            continue

        # Missing or unknown values get a neutral fallback.
        if value == "Unknown":
            color_map[value] = missing_color
            continue

        if deterministic_by_hash:
            palette_index = stable_hash_index(value, len(palette))
        else:
            palette_index = index % len(palette)

        color_map[value] = palette[palette_index]

    return color_map


def build_node_attribute_color_map(
    graph: nx.Graph,
    attribute: str,
    *,
    palette: list[str] = DISTINCT_20_COLORS,
    custom_colors: dict[Any, str] | None = None,
    missing_color: str = DEFAULT_MISSING_COLOR,
    deterministic_by_hash: bool = False,
) -> dict[str, str]:
    """
    Convenience wrapper for categorical node attributes.

    Example:
        party_color_map = build_node_attribute_color_map(
            G,
            "party",
            custom_colors={"Independent": "#7F7F7F"},
        )
    """
    values = get_unique_node_values(graph, attribute)

    return build_categorical_color_map(
        values,
        palette=palette,
        custom_colors=custom_colors,
        missing_color=missing_color,
        deterministic_by_hash=deterministic_by_hash,
    )


def get_numeric_node_range(
    graph: nx.Graph,
    attribute: str,
    *,
    default_min: float = 0.0,
    default_max: float = 1.0,
) -> tuple[float, float]:
    """
    Return min/max for a numeric node attribute.

    Non-numeric values like 'Unknown', 'Unknow', '', None are ignored.
    """
    values: list[float] = []

    for _, attrs in graph.nodes(data=True):
        numeric_value = safe_float(attrs.get(attribute), default=None)

        if numeric_value is not None:
            values.append(numeric_value)

    if not values:
        return default_min, default_max

    return min(values), max(values)


def map_numeric_to_sequential_color(
    value: float,
    min_value: float,
    max_value: float,
    *,
    low_color: str = "#D6EAF8",
    high_color: str = "#154360",
) -> str:
    """
    Map a numeric value to a sequential color scale.

    Useful for:
    - influence
    - degree centrality
    - betweenness
    - risk score
    - frequency/count values
    """
    normalized = normalize(value, min_value, max_value)
    return interpolate_color(normalized, low_color, high_color)


# ---------------------------------------------------------------------
# Build reusable visual mappings for a graph
# ---------------------------------------------------------------------


def build_visual_mappings(
    graph: nx.Graph,
    *,
    categorical_attributes: set[str],
    custom_category_colors: dict[str, dict[str, str]] | None = None,
    palette: list[str] = DISTINCT_20_COLORS,
) -> dict[str, dict[str, str]]:
    """
    Build categorical color maps for multiple node attributes.

    Parameters
    ----------
    graph:
        Your NetworkX graph.
    categorical_attributes:
        Node attributes that should be treated as categorical.
    custom_category_colors:
        Optional per-attribute custom overrides.
    palette:
        Automatic fallback palette.

    Returns
    -------
    dict[str, dict[str, str]]
        Example:
        {
            "party": {
                "Progressive": "#1F77B4",
                "Conservative": "#D62728",
            },
            "role": {
                "Leader": "#111111",
                "MP": "#566573",
            }
        }
    """
    custom_category_colors = custom_category_colors or {}

    mappings: dict[str, dict[str, str]] = {}

    for attribute in categorical_attributes:
        if not any(attribute in attrs for _, attrs in graph.nodes(data=True)):
            continue

        mappings[attribute] = build_node_attribute_color_map(
            graph,
            attribute,
            palette=palette,
            custom_colors=custom_category_colors.get(attribute, {}),
            deterministic_by_hash=False,
        )

    return mappings


# ---------------------------------------------------------------------
# Edge weight helper functions
# ---------------------------------------------------------------------

DEFAULT_EDGE_COLOR = "#7F8C8D"

EDGE_SEQUENTIAL_COLOR_SCALE = {
    "low": "#D6DBDF",
    "high": "#2C3E50",
}


def get_numeric_edge_range(
    graph: nx.Graph,
    attribute: str = "Weight",
    *,
    default_min: float = 0.0,
    default_max: float = 1.0,
) -> tuple[float, float]:
    """Return min/max for a numeric edge attribute."""
    values: list[float] = []

    for _, _, attrs in graph.edges(data=True):
        value = attrs.get(attribute)

        if isinstance(value, int | float):
            values.append(float(value))

    if not values:
        return default_min, default_max

    return min(values), max(values)


def map_edge_width(
    graph: nx.Graph,
    edge_attrs: dict[str, Any],
    *,
    weight_attribute: str = "Weight",
    min_width: float = 1.0,
    max_width: float = 8.0,
) -> float:
    """Map edge weight to Cytoscape edge width."""
    min_value, max_value = get_numeric_edge_range(graph, weight_attribute)

    raw_weight = float(edge_attrs.get(weight_attribute, 1.0) or 1.0)
    normalized = normalize(raw_weight, min_value, max_value)

    return round(min_width + normalized * (max_width - min_width), 2)


def map_edge_color(
    graph: nx.Graph,
    edge_attrs: dict[str, Any],
    *,
    weight_attribute: str = "Weight",
    low_color: str = EDGE_SEQUENTIAL_COLOR_SCALE["low"],
    high_color: str = EDGE_SEQUENTIAL_COLOR_SCALE["high"],
) -> str:
    """Map edge weight to a sequential edge color."""
    min_value, max_value = get_numeric_edge_range(graph, weight_attribute)

    raw_weight = float(edge_attrs.get(weight_attribute, 1.0) or 1.0)

    return map_numeric_to_sequential_color(
        raw_weight,
        min_value,
        max_value,
        low_color=low_color,
        high_color=high_color,
    )
