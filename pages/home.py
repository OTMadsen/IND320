import streamlit as st


def page_home():
    st.title("Norwegian Hydro Reservoir Dashboard")
    st.write("Welcome to the Norwegian Hydro Reservoir Analytics Application")
    st.header("Project Overview")
    st.markdown("""

    Welcome to the Norwegian Hydro Reservoir Dashboard!

    This web-application explores Norwegian reservoir levels, electricity flows and weather observations
    through interactive visualizations and statistical analysis.    

    Use the different pages in the menu to explore the datasets, investigate patterns and anomalies, or generate
    forecasts of future reservoir filling levels.


     **Data Sources**: NVE Magasinstatistikk API and ENTSO-E electricity flows (MongoDB Atlas)  
     **Time Range**: Based on currently available records  
     **Features**: Reservoir tables, interactive plots, and electricity flow charts.

     The NVE API provides the full historical dataset. The selected month interval
     is applied locally to the cached response, not to the API download.
     """)
