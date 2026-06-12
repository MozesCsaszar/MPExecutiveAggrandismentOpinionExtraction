import streamlit as st
import pandas as pd
from helpers import (
    humanize_text,
    load_data as _load_data,
    build_graphs as _build_graphs,
)


@st.cache_resource
def load_data():
    return _load_data()


@st.cache_data
def build_graphs(
    df,
):
    start = pd.Timestamp(year=2020, month=1, day=1)
    end = pd.Timestamp(year=2020, month=12, day=31)

    graphs, combined_data = _build_graphs(
        df, start, end, pd.offsets.MonthBegin(1), pd.offsets.MonthEnd(0)  # type: ignore
    )

    # change the capitalization of the properties
    combined_data.columns = [humanize_text(c) for c in combined_data.columns]
    for graph in graphs:
        for _, attrs in graph["graph"].nodes(data=True):
            new_attrs = {humanize_text(k): v for k, v in attrs.items()}
            attrs.clear()
            attrs.update(new_attrs)

        for _, _, attrs in graph["graph"].edges(data=True):
            new_attrs = {humanize_text(k): v for k, v in attrs.items()}
            attrs.clear()
            attrs.update(new_attrs)

    return graphs, combined_data
