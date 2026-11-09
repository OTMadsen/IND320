import streamlit as st


def page_home():
    st.title("Norwegian Hydro Reservoir Dashboard")
    st.write("Welcome to the Norwegian Hydro Reservoir Analytics Application")
    st.header("Project Overview")
    st.markdown("""
     **Data Sources**: NVE Magasinstatistikk API and ENTSO-E electricity flows (MongoDB Atlas)  
     **Time Range**: Based on currently available records  
     **Features**: Reservoir tables, interactive plots, and electricity flow charts.

     The NVE API provides the full historical dataset. The selected month interval
     is applied locally to the cached response, not to the API download.
     """)
