from .palettes import (
    DEFAULT_MISSING_COLOR,
    DEFAULT_NODE_BORDER_COLOR,
    DEFAULT_NODE_BORDER_WIDTH,
    DEFAULT_NODE_FILL_COLOR,
    DEFAULT_NODE_SIZE,
    SEQUENTIAL_COLOR_SCALE,
    DIVERGING_COLOR_SCALE,
)
from .color_helpers import (
    get_numeric_node_range,
    map_edge_color,
    map_edge_width,
    map_numeric_to_sequential_color,
    interpolate_diverging_color,
    normalize,
    get_unique_node_values,
    safe_float,
)
from components.cytoscape_graph.cytoscape_styles import (
    build_edge_styles,
    build_node_styles,
)

from components.cytoscape_graph.color_helpers import build_visual_mappings
from components.cytoscape_graph.palettes import (
    DISTINCT_20_COLORS,
    CUSTOM_CATEGORY_COLORS,
    CATEGORICAL_NODE_ATTRIBUTES,
)
from streamlit_cytoscape import streamlit_cytoscape

import networkx as nx
from typing import Any


def get_node_value(
    attrs: dict[str, Any],
    attribute: str,
    default: Any = "Unknown",
) -> Any:
    """Safely retrieve a node attribute."""
    value = attrs.get(attribute, default)

    if value is None:
        return default

    return value


def map_node_fill_color(
    graph: nx.Graph,
    attrs: dict[str, Any],
    attribute: str | None,
    visual_color_maps: dict[str, dict[str, str]],
    *,
    default_color: str = DEFAULT_NODE_FILL_COLOR,
    missing_color: str = DEFAULT_MISSING_COLOR,
) -> str:
    """
    Map node fill color from categorical or numeric data.

    Behavior:
    - If attribute is None: use default color.
    - If attribute has a categorical color map: use that map.
    - If attribute is numeric: use sequential color scale.
    - If value is missing or malformed: use missing/default color.
    """
    if attribute is None:
        return default_color

    value = get_node_value(attrs, attribute, default=None)

    # Categorical color.
    # This branch must happen before numeric conversion.
    if attribute in visual_color_maps:
        if value is None:
            return missing_color

        value_str = str(value).strip()

        if value_str == "":
            return missing_color

        return visual_color_maps[attribute].get(value_str, missing_color)

    # Numeric color.
    numeric_value = safe_float(value, default=None)

    if numeric_value is None:
        return missing_color

    min_value, max_value = get_numeric_node_range(graph, attribute)

    if attribute == "Opinion":
        return interpolate_diverging_color(
            numeric_value,
            min_value=min_value,
            midpoint=0.0,
            max_value=max_value,
            low_color=DIVERGING_COLOR_SCALE["low"],
            mid_color=DIVERGING_COLOR_SCALE["mid"],
            high_color=DIVERGING_COLOR_SCALE["high"],
        )

    return map_numeric_to_sequential_color(
        numeric_value,
        min_value,
        max_value,
        low_color=SEQUENTIAL_COLOR_SCALE["low"],
        high_color=SEQUENTIAL_COLOR_SCALE["high"],
    )


def map_node_border_color(
    graph: nx.Graph,
    attrs: dict[str, Any],
    attribute: str | None,
    visual_color_maps: dict[str, dict[str, str]],
    *,
    default_color: str = DEFAULT_NODE_BORDER_COLOR,
    missing_color: str = DEFAULT_MISSING_COLOR,
) -> str:
    """
    Map node border color from categorical or numeric data.

    Behavior:
    - If attribute is None: use default border color.
    - If attribute has a categorical color map: use that map.
    - If attribute is numeric: use numeric color scale.
    - If value is missing or malformed: use missing/default color.
    """
    if attribute is None:
        return default_color

    value = get_node_value(attrs, attribute, default=None)

    # Categorical color.
    if attribute in visual_color_maps:
        if value is None:
            return missing_color

        value_str = str(value).strip()

        if value_str == "":
            return missing_color

        return visual_color_maps[attribute].get(value_str, missing_color)

    # Numeric color.
    numeric_value = safe_float(value, default=None)

    if numeric_value is None:
        return missing_color

    min_value, max_value = get_numeric_node_range(graph, attribute)

    if attribute == "ideology":
        return interpolate_diverging_color(
            numeric_value,
            min_value=min_value,
            midpoint=0.0,
            max_value=max_value,
            low_color=DIVERGING_COLOR_SCALE["low"],
            mid_color=DIVERGING_COLOR_SCALE["mid"],
            high_color=DIVERGING_COLOR_SCALE["high"],
        )

    return map_numeric_to_sequential_color(
        numeric_value,
        min_value,
        max_value,
        low_color="#FAD7A0",
        high_color="#78281F",
    )


def map_node_size(
    graph: nx.Graph,
    attrs: dict[str, Any],
    attribute: str | None,
    *,
    min_size: float = 30.0,
    max_size: float = 95.0,
    default_size: float = DEFAULT_NODE_SIZE,
) -> float:
    """
    Map a numeric node attribute to node size.

    If the value is missing or malformed, use default_size.
    """
    if attribute is None:
        return default_size

    raw_value = safe_float(attrs.get(attribute), default=None)

    if raw_value is None:
        return default_size

    min_value, max_value = get_numeric_node_range(graph, attribute)
    normalized = normalize(raw_value, min_value, max_value)

    return round(min_size + normalized * (max_size - min_size), 2)


def map_node_border_width(
    graph: nx.Graph,
    attrs: dict[str, Any],
    attribute: str | None,
    *,
    min_width: float = 1.0,
    max_width: float = 7.0,
    default_width: float = DEFAULT_NODE_BORDER_WIDTH,
) -> float:
    """
    Map a numeric node attribute to border width.

    If the value is missing or malformed, use default_width.
    """
    if attribute is None:
        return default_width

    raw_value = safe_float(attrs.get(attribute), default=None)

    if raw_value is None:
        return default_width

    min_value, max_value = get_numeric_node_range(graph, attribute)
    normalized = normalize(raw_value, min_value, max_value)

    return round(min_width + normalized * (max_width - min_width), 2)


def build_cytoscape_elements_from_networkx(
    graph: nx.Graph,
    *,
    visual_color_maps: dict[str, dict[str, str]],
    label_attribute: str = "label",
    group_attribute: str | None = None,
    use_physical_groups: bool = True,
    fill_color_attribute: str = "party",
    size_attribute: str | None = None,
    border_color_attribute: str | None = None,
    border_width_attribute: str | None = None,
    edge_weight_attribute: str = "Weight",
    show_node_labels: bool = False,
    show_edge_weights: bool = False,
    min_node_size: float = 30.0,
    max_node_size: float = 95.0,
    min_border_width: float = 1.0,
    max_border_width: float = 7.0,
    min_edge_width: float = 1.0,
    max_edge_width: float = 8.0,
) -> dict[str, list[dict[str, Any]]]:
    """
    Convert an existing NetworkX graph into the element format expected by
    streamlit_cytoscape(...).

    This version assumes edges have only a numeric weight attribute.

    Returns
    -------
    dict
        {
            "nodes": [...],
            "edges": [...]
        }
    """

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Add compound parent/group nodes
    # -------------------------------------------------------------
    # The current streamlit-cytoscape package uses node data["label"]
    # as a selector for NodeStyle. Therefore:
    #
    # - label="group" means the node uses the group NodeStyle.
    # - caption=<actual text> is what gets displayed.

    if use_physical_groups and group_attribute is not None:
        group_values = get_unique_node_values(graph, group_attribute)
        group_color_map = visual_color_maps.get(group_attribute, {})

        for group_value in group_values:
            group_id = f"group-{group_attribute}-{group_value}"
            group_color = group_color_map.get(group_value, "#EAECEE")

            nodes.append(
                {
                    "data": {
                        "id": group_id,
                        # Style selector.
                        "label": "group",
                        # Visible text.
                        "caption": group_value,
                        "is_group": True,
                        "group_value": group_value,
                        "group_color": group_color,
                    }
                }
            )

    # -------------------------------------------------------------
    # 2. Add ordinary graph nodes
    # -------------------------------------------------------------

    for node_id, attrs in graph.nodes(data=True):
        display_label = str(attrs.get(label_attribute, node_id))

        fill_color = map_node_fill_color(
            graph,
            attrs,
            fill_color_attribute,
            visual_color_maps,
        )

        node_size = map_node_size(
            graph,
            attrs,
            size_attribute,
            min_size=min_node_size,
            max_size=max_node_size,
        )

        border_color = map_node_border_color(
            graph,
            attrs,
            border_color_attribute,
            visual_color_maps,
        )

        border_width = map_node_border_width(
            graph,
            attrs,
            border_width_attribute,
            min_width=min_border_width,
            max_width=max_border_width,
        )

        node_data: dict[str, Any] = {
            "id": str(node_id),
            # Style selector.
            "label": "actor",
            # Visible text.
            "caption": display_label if show_node_labels else "",
            # Keep original node attributes available for tooltips/inspection.
            **attrs,
            # Visual attributes consumed by NodeStyle custom_styles.
            "fill_color": fill_color,
            "node_size": node_size,
            "border_color": border_color,
            "border_width": border_width,
        }

        # Physical grouping via Cytoscape compound nodes.
        if use_physical_groups and group_attribute is not None:
            group_value = str(attrs.get(group_attribute, "Unknown"))
            node_data["parent"] = f"group-{group_attribute}-{group_value}"

        nodes.append({"data": node_data})

    # -------------------------------------------------------------
    # 3. Add weighted edges
    # -------------------------------------------------------------
    # Your edges only need:
    #
    #   G.add_edge(source, target, weight=0.73)
    #
    # If an edge has no weight, it falls back to 1.0.

    for source, target, attrs in graph.edges(data=True):
        weight = float(attrs.get(edge_weight_attribute, 1.0) or 1.0)

        edge_width = map_edge_width(
            graph,
            attrs,
            weight_attribute=edge_weight_attribute,
            min_width=min_edge_width,
            max_width=max_edge_width,
        )

        edge_color = map_edge_color(
            graph,
            attrs,
            weight_attribute=edge_weight_attribute,
        )

        edge_id = f"{source}-{target}"

        edges.append(
            {
                "data": {
                    "id": edge_id,
                    "source": str(source),
                    "target": str(target),
                    # Style selector for EdgeStyle.
                    "label": "weighted_edge",
                    # Optional visible edge caption.
                    "caption": f"{weight:.2f}" if show_edge_weights else "",
                    # Preserve the numeric edge weight.
                    "Weight": weight,
                    # Visual attributes consumed by EdgeStyle custom_styles.
                    "edge_width": edge_width,
                    "edge_color": edge_color,
                }
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
    }


def build_layout(
    layout_name: str = "fcose",
    *,
    group_spacing: float = 1.8,
) -> dict:
    """
    Build a Cytoscape layout configuration.

    group_spacing controls how aggressively the layout tries to spread groups.
    Higher values generally produce more separation.
    """

    if layout_name == "fcose":
        return {
            "name": "fcose",
            # Fit graph into viewport.
            "fit": True,
            "padding": 80,
            # Animation.
            "animate": True,
            "animationDuration": 800,
            # Stronger node repulsion spreads nodes and groups.
            "nodeRepulsion": 9000 * group_spacing,
            # Longer ideal edge length prevents everything from collapsing.
            "idealEdgeLength": 140 * group_spacing,
            # Lower elasticity makes edges less aggressively pull nodes together.
            "edgeElasticity": 0.25,
            # Important for compound nodes.
            # Higher values increase separation between nested structures.
            "nestingFactor": 1.4 * group_spacing,
            # Reduces global collapsing toward the center.
            "gravity": 0.15,
            # Reduces compound-group collapsing.
            "gravityCompound": 0.08,
            # Larger range means gravity acts less locally.
            "gravityRange": 3.8,
            "gravityRangeCompound": 3.8,
            # Try to avoid overlap.
            "randomize": True,
            "packComponents": True,
            "nodeDimensionsIncludeLabels": True,
        }

    if layout_name == "cose":
        return {
            "name": "cose",
            "fit": True,
            "padding": 80,
            "animate": True,
            "animationDuration": 800,
            "nodeRepulsion": 10000 * group_spacing,
            "idealEdgeLength": 150 * group_spacing,
            "edgeElasticity": 0.20,
            "gravity": 0.12,
            "componentSpacing": 160 * group_spacing,
            "nodeOverlap": 30,
        }

    if layout_name == "circle":
        return {
            "name": "circle",
            "fit": True,
            "padding": 100,
            "nodeDimensionsIncludeLabels": True,
        }

    if layout_name == "grid":
        return {
            "name": "grid",
            "fit": True,
            "padding": 100,
            "avoidOverlap": True,
            "avoidOverlapPadding": 80,
            "nodeDimensionsIncludeLabels": True,
        }

    if layout_name == "concentric":
        return {"name": "concentric"}

    if layout_name == "breadthfirst":
        return {"name": "breadthfirst"}

    return {
        "name": "fcose",
        "fit": True,
        "padding": 80,
    }


def show_cytoscape_graph(
    graph: nx.Graph,
    *,
    label_attribute: str = "label",
    group_attribute: str | None = None,
    use_physical_groups: bool = True,
    fill_color_attribute: str = "party",
    size_attribute: str | None = None,
    border_color_attribute: str | None = None,
    border_width_attribute: str | None = None,
    edge_weight_attribute: str = "Weight",
    show_node_labels: bool = False,
    show_edge_weights: bool = False,
    min_node_size: float = 30.0,
    max_node_size: float = 95.0,
    min_border_width: float = 1.0,
    max_border_width: float = 7.0,
    min_edge_width: float = 1.0,
    max_edge_width: float = 8.0,
    layout_name: str = "fcose",
    group_spacing: float = 1,
):
    visual_color_maps = build_visual_mappings(
        graph,
        categorical_attributes=CATEGORICAL_NODE_ATTRIBUTES,
        custom_category_colors=CUSTOM_CATEGORY_COLORS,
        palette=DISTINCT_20_COLORS,
    )

    node_styles = build_node_styles()
    edge_styles = build_edge_styles(directed=False)

    elements = build_cytoscape_elements_from_networkx(
        graph,
        visual_color_maps=visual_color_maps,
        label_attribute=label_attribute,
        group_attribute=group_attribute,
        use_physical_groups=use_physical_groups,
        fill_color_attribute=fill_color_attribute,
        size_attribute=size_attribute,
        border_color_attribute=border_color_attribute,
        border_width_attribute=border_width_attribute,
        edge_weight_attribute=edge_weight_attribute,
        show_node_labels=show_node_labels,
        show_edge_weights=show_edge_weights,
        min_node_size=float(min_node_size),
        max_node_size=float(max_node_size),
        min_border_width=min_border_width,
        max_border_width=float(max_border_width),
        min_edge_width=min_edge_width,
        max_edge_width=max_edge_width,
    )

    selected = streamlit_cytoscape(
        elements=elements,
        layout=build_layout(layout_name, group_spacing=group_spacing),
        node_styles=node_styles,
        edge_styles=edge_styles,
        height=760,
        key="political-network-cytoscape",
    )

    return selected
