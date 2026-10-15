from __future__ import annotations

import time
from collections import deque

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CV Tracking Dashboard", layout="wide")
st.title("Real-Time Tracking Analytics")

placeholder = st.empty()
col1, col2, col3 = st.columns(3)
metric_fps = col1.empty()
metric_tracks = col2.empty()
metric_total = col3.empty()

chart_placeholder = st.empty()
table_placeholder = st.empty()

history = deque(maxlen=200)

while True:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=1.0)
        data = r.json()

        metric_fps.metric("FPS", f"{data['fps']:.1f}")
        metric_tracks.metric("Active Tracks", data["active_tracks"])

        total = sum(ls.get("total", 0) for ls in data.get("line_stats", []))
        metric_total.metric("Total Crossings", total)

        history.append(
            {
                "time": pd.Timestamp.now(),
                "fps": data["fps"],
                "tracks": data["active_tracks"],
            }
        )
        df = pd.DataFrame(history)
        fig = px.line(df, x="time", y=["fps", "tracks"], title="Live metrics")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        if data.get("tracks"):
            tdf = pd.DataFrame(data["tracks"])
            table_placeholder.dataframe(tdf, use_container_width=True)

    except Exception as e:
        st.warning(f"API not reachable: {e}")

    time.sleep(1.0)