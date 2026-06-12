from streamlit_cytoscape import NodeStyle, EdgeStyle


def build_node_styles() -> list[NodeStyle]:
    """
    Build node styles for the streamlit_cytoscape API.

    The style selector is matched against node data["label"].
    """

    return [
        NodeStyle(
            label="actor",
            color="#1F77B4",
            caption="caption",
            icon=None,
            custom_styles={
                "background-color": "data(fill_color)",
                "width": "data(node_size)",
                "height": "data(node_size)",
                "border-color": "data(border_color)",
                "border-width": "data(border_width)",
                "border-opacity": 1.0,
                "shape": "ellipse",
                "color": "#17202A",
                "font-size": 11,
                "text-valign": "center",
                "text-halign": "center",
                "text-wrap": "wrap",
                "text-max-width": 85,
                "overlay-padding": 6,
                "z-index": 20,
            },
        ),
        NodeStyle(
            label="group",
            color="#EAECEE",
            caption="caption",
            icon=None,
            custom_styles={
                "background-color": "data(group_color)",
                "background-opacity": 0.10,
                "border-color": "data(group_color)",
                "border-width": 4,
                "border-style": "dashed",
                "shape": "roundrectangle",
                "color": "#17202A",
                "font-size": 60,
                "font-weight": "bold",
                "text-valign": "top",
                "text-halign": "center",
                "text-margin-y": -12,
                "padding": 60,
                "min-width": 220,
                "min-height": 160,
                "z-index": 1,
            },
        ),
    ]


def build_edge_styles(
    *,
    directed: bool = False,
) -> list[EdgeStyle]:
    """
    Build edge styles.

    Since your graph only has weighted edges, one generic EdgeStyle is enough.
    """

    custom_styles = {
        "line-color": "data(edge_color)",
        "width": "data(edge_width)",
        "opacity": 0.75,
        "curve-style": "bezier",
        "font-size": 9,
        "color": "#566573",
        "text-rotation": "autorotate",
        "text-margin-y": -8,
    }

    if directed:
        custom_styles.update(
            {
                "target-arrow-color": "data(edge_color)",
                "target-arrow-shape": "triangle",
            }
        )

    return [
        EdgeStyle(
            label="weighted_edge",
            caption="caption",
            directed=directed,
            custom_styles=custom_styles,
        )
    ]
