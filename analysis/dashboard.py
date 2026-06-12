import streamlit as st

st.set_page_config(page_title="Political Data Dashboard", layout="wide")

# main = st.container(width=1400)
left, center, right = st.columns([1, 6, 1])

with center:
    single_graph_page = st.Page("single_graph.py", title="Single Graph", default=True)
    time_series_page = st.Page("time_series.py", title="Time Series")

    page = st.navigation(
        [single_graph_page, time_series_page],
        position="top",
    )

    page.run()
