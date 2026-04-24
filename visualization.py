import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

YEAR = 2017


@st.cache_data
def load_data():
    return (
        pd.read_csv(
            f"./outputs/hu_{YEAR}.csv",
            dtype={
                "Speaker_minister": "bool",
                "Speaker_MP": "bool",
            },
            parse_dates=["Date"],
        )
        .set_index(["Speaker_ID", "Date", "ID"])
        .sort_index()
    )


df_speech = load_data()


timestep = st.selectbox("Timestep (Length of interval)", [1, 3, 12])
start_options = list(np.arange(1, 13, timestep))
start_time = st.selectbox("Start Time (Month)", start_options, index=1)

start = pd.Timestamp(year=YEAR, month=int(start_time), day=1)
end = start + pd.offsets.MonthEnd(timestep - 1)

data = df_speech.loc[(slice(None), slice(start, end)), :]
parties = {
    value: i for i, value in enumerate(list(df_speech["Speaker_party"].unique()))
}

print(len(data))


x, y, colors, area = [], [], [], []

i = 0
for key, group in data.groupby(level=0):
    x.append(i)
    i += 1
    y.append(group.iloc[0]["Speaker_party"])
    area.append(len(group) * 2)
    ea_score = (
        len(group[group["label"] == "PRO"]) - len(group[group["label"] == "CONTRA"])
    ) / len(group)
    colors.append(ea_score)

fig, ax = plt.subplots()
cmap = LinearSegmentedColormap.from_list(
    "red_gray_green", ["red", "lightgray", "green"]
)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
sc = ax.scatter(x, y, s=area, c=colors, alpha=0.7, cmap=cmap, norm=norm)
ax.set_title("MP Opinion Score About EA")

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Score (CONTRA = red, NEUTRAL = gray, PRO = green)")

st.pyplot(fig)
