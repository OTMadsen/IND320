import streamlit as st

from pages.home import page_home
from pages.reservoir_table import page_data_table
from pages.reservoir_plot import page_data_plot
from pages.reservoir_analysis import page_reservoir_analysis
from pages.weather_anomalies import page_weather_anomalies
from pages.area_map import page_area_map
from pages.sliding_window_correlation import page_sliding_window_correlation
from pages.river_forecasting import page_river_forecasting
from pages.electricity import page_electricity


st.set_page_config(page_title="Hydro Reservoir Dashboard", layout="wide")

pg = st.navigation([
    st.Page(page_home, title="Home"),
    st.Page(page_data_table, title="Reservoir Data Table"),
    st.Page(page_data_plot, title="Reservoir Data Plot"),
    st.Page(page_reservoir_analysis, title="Reservoir Analysis"),
    st.Page(page_weather_anomalies, title="Weather Anomalies"),
    st.Page(page_area_map, title="Interactive Area Map"),
    st.Page(page_sliding_window_correlation, title="Sliding Window Correlation"),
    st.Page(page_river_forecasting, title="River Forecasting"),
    st.Page(page_electricity, title="Electricity Flow"),
])



AREA_OPTIONS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

if "selected_electricity_area" not in st.session_state:
    st.session_state["selected_electricity_area"] = "NO3"


# Update the shared area when the sidebar selection changes
def on_sidebar_area_change():
    chosen = st.session_state["_sidebar_electricity_area"]

    st.session_state["selected_electricity_area"] = chosen
    st.session_state["selected_area_id"] = chosen.replace("NO", "NO ")
    st.session_state["selected_area_type"] = "EL"



shared_area = st.session_state["selected_electricity_area"]

if shared_area not in AREA_OPTIONS:
    shared_area = "NO3"
    st.session_state["selected_electricity_area"] = shared_area

if st.session_state.get("_sidebar_electricity_area") != shared_area:
    st.session_state["_sidebar_electricity_area"] = shared_area


st.sidebar.divider()

st.sidebar.selectbox(
    "Electricity area for analysis",
    options=AREA_OPTIONS,
    key="_sidebar_electricity_area",
    on_change=on_sidebar_area_change,
)


pg.run()
