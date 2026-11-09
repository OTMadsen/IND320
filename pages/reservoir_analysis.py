import matplotlib.pyplot as plt
import streamlit as st
from ui.filters import get_selected_electricity_area
from services.data import get_reservoir_series
from analysis.reservoir import plot_reservoir_stl, plot_reservoir_spectrogram


def page_reservoir_analysis():
    region = get_selected_electricity_area()
    st.title("Reservoir Analysis")
    st.caption(f"Electricity area: {region} · Weekly Magasinstatistikk filling degree")
    try:
        series = get_reservoir_series(region)
    except Exception as exc:
        st.error(f"Could not prepare reservoir data: {exc}")
        return

    st.write(f"Observations: **{len(series)}**; "
             f"range: **{series.index.min():%Y-%m-%d}** to "
             f"**{series.index.max():%Y-%m-%d}**")

    tab_stl, tab_spectrogram = st.tabs(["STL decomposition", "Spectrogram"])
    with tab_stl:
        st.write("Separate the series into trend, recurring seasonality and residuals.")
        seasonal = st.slider("Seasonal smoother (odd)", 3, 51, 13, 2)
        trend = st.slider("Trend smoother (odd)", 53, 261, 105, 2)
        robust = st.checkbox("Robust STL", value=True)
        try:
            fig = plot_reservoir_stl(series, seasonal, trend, robust)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"STL calculation failed: {exc}")

    with tab_spectrogram:
        st.write("Inspect how reservoir filling frequencies vary over time.")
        max_window = min(len(series), 260)
        default_window = min(104, max_window)
        window = st.slider(
            "Window length (weeks)", min_value=4,
            max_value=int(max_window), value=int(default_window),
        )
        overlap = st.slider(
            "Window overlap (weeks)", min_value=0,
            max_value=window - 1, value=window // 2,
        )
        try:
            fig = plot_reservoir_spectrogram(series, window, overlap)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Spectrogram calculation failed: {exc}")
