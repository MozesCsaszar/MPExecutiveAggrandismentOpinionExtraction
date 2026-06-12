# ---------------------------------------------------------------------
# Automatic Streamlit legend helpers
# ---------------------------------------------------------------------

import streamlit as st

from .color_helpers import (
    EDGE_SEQUENTIAL_COLOR_SCALE,
    get_numeric_edge_range,
    get_numeric_node_range,
)
from .palettes import DIVERGING_COLOR_SCALE, SEQUENTIAL_COLOR_SCALE
import networkx as nx


def render_categorical_legend(
    title: str,
    color_map: dict[str, str],
) -> None:
    """Render a categorical color legend."""
    st.markdown(f"**{title}**")

    for value, color in color_map.items():
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;margin-bottom:0.25rem;">
                <div style="
                    width:14px;
                    height:14px;
                    border-radius:50%;
                    background:{color};
                    border:1px solid #333;
                    margin-right:0.5rem;">
                </div>
                <span>{value}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_numeric_gradient_legend(
    title: str,
    *,
    low_color: str,
    high_color: str,
    low_label: str,
    high_label: str,
) -> None:
    """Render a sequential numeric gradient legend."""
    st.markdown(f"**{title}**")

    st.markdown(
        f"""
        <div style="
            width:100%;
            height:16px;
            background:linear-gradient(90deg, {low_color}, {high_color});
            border:1px solid #666;
            border-radius:3px;">
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
            <span>{low_label}</span>
            <span>{high_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diverging_gradient_legend(
    title: str,
    *,
    low_color: str,
    mid_color: str,
    high_color: str,
    low_label: str,
    mid_label: str,
    high_label: str,
) -> None:
    """Render a diverging numeric gradient legend."""
    st.markdown(f"**{title}**")

    st.markdown(
        f"""
        <div style="
            width:100%;
            height:16px;
            background:linear-gradient(90deg, {low_color}, {mid_color}, {high_color});
            border:1px solid #666;
            border-radius:3px;">
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
            <span>{low_label}</span>
            <span>{mid_label}</span>
            <span>{high_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_attribute_color_legend(
    graph: nx.Graph,
    attribute: str,
    visual_color_maps: dict[str, dict[str, str]],
    *,
    title_prefix: str,
) -> None:
    """
    Render a legend for either categorical or numeric color mappings.
    """
    if attribute in visual_color_maps:
        render_categorical_legend(
            title=f"{title_prefix}: {attribute}",
            color_map=visual_color_maps[attribute],
        )
        return

    min_value, max_value = get_numeric_node_range(graph, attribute)

    if attribute == "ideology":
        render_diverging_gradient_legend(
            title=f"{title_prefix}: {attribute}",
            low_color=DIVERGING_COLOR_SCALE["low"],
            mid_color=DIVERGING_COLOR_SCALE["mid"],
            high_color=DIVERGING_COLOR_SCALE["high"],
            low_label=f"{min_value:.2f}",
            mid_label="0.00",
            high_label=f"{max_value:.2f}",
        )
        return

    render_numeric_gradient_legend(
        title=f"{title_prefix}: {attribute}",
        low_color=SEQUENTIAL_COLOR_SCALE["low"],
        high_color=SEQUENTIAL_COLOR_SCALE["high"],
        low_label=f"{min_value:.2f}",
        high_label=f"{max_value:.2f}",
    )


def render_edge_weight_legend(
    graph: nx.Graph,
    *,
    weight_attribute: str = "Weight",
) -> None:
    """Render an edge-weight legend."""
    min_weight, max_weight = get_numeric_edge_range(graph, weight_attribute)

    render_numeric_gradient_legend(
        title=f"Edge weight: {weight_attribute}",
        low_color=EDGE_SEQUENTIAL_COLOR_SCALE["low"],
        high_color=EDGE_SEQUENTIAL_COLOR_SCALE["high"],
        low_label=f"{min_weight:.2f}",
        high_label=f"{max_weight:.2f}",
    )

    st.caption("Thin/light edge = low weight; thick/dark edge = high weight.")
