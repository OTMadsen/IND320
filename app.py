import streamlit as st

from pages.home import page_home
from pages.reservoir_table import page_data_table
from pages.reservoir_plot import page_data_plot
from pages.reservoir_analysis import page_reservoir_analysis
from pages.weather_anomalies import page_weather_anomalies
from pages.electricity import page_electricity

st.set_page_config(page_title="Hydro Reservoir Dashboard", layout="wide")

pg = st.navigation([
    st.Page(page_home, title="Home"),
    st.Page(page_data_table, title="Reservoir Data Table"),
    st.Page(page_data_plot, title="Reservoir Data Plot"),
    st.Page(page_reservoir_analysis, title="Reservoir Analysis"),
    st.Page(page_weather_anomalies, title="Weather Anomalies"),
    st.Page(page_electricity, title="Electricity Flow"),
])

st.sidebar.divider()
st.sidebar.selectbox(
    "Electricity area for analysis",
    options=["NO1", "NO2", "NO3", "NO4", "NO5"],
    index=2,
    key="selected_electricity_area",
)

pg.run()
