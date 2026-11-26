
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.data import load_data, filter_nve_data
from ui.filters import select_nve_month_range


def page_data_plot():
    st.title("Page 3: Interactive Data Plotter")
    st.markdown(
        """
        
        About this analysis:

        This page visualizes how Norwegian reservoir measurements change over time. 

        Select a geographical area classification, time interval,
        and measurement to compare historical developments.

        It is also possible to display multiple variables together using normalization,
        allowing measurements with different scales to be compared more easily. 

        """
    )
    df = load_data()
    area_type = st.radio(
        "Select area type:",
        ["EL", "NO", "VASS"],
        horizontal=True,
        key="plot_area_type"
    )
    
    df_no = (
        df[df["Area_Type"] == area_type]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    
    num_cols = [
        "Fillings_Degree",
        "Filling_TWh",
        "Capacity_TWh",
        "Change_Filling_Degree",
    ]
    options = ["All columns together"] + num_cols

    selected_col = st.selectbox(
        "Select Column to Plot:", options=options, index=1
    )

    selected_range = select_nve_month_range(df, "plot")
    if selected_range is None:
        return
    start_m, end_m = selected_range
    df_filtered = filter_nve_data(df_no, start_m, end_m)
    if df_filtered.empty:
        st.info("No reservoir data for this selection.")
        return

    st.caption(
        f"Selected data interval: {start_m} to {end_m}. "
        "NVE returns the complete historical dataset through this endpoint; "
        "the selected interval is filtered locally after download."
    )

    st.subheader(f"Plotting: {selected_col} ({start_m} – {end_m})")
    
    fig = go.Figure()

    if selected_col == "All columns together":

        area_number = st.selectbox(
            "Select area number:",
            sorted(df_filtered["Area_Number"].unique()),
        )

        area_df = df_filtered[
            df_filtered["Area_Number"] == area_number
        ].sort_values("Date")

        # Normalize each variable to a common scale [0, 1]
        for col in num_cols:

            values = pd.to_numeric(
                area_df[col], errors="coerce"
            )

            mn = values.min()
            mx = values.max()

            if pd.notna(mn) and pd.notna(mx) and mx > mn:
                normalized = (values - mn) / (mx - mn)
            else:
                normalized = pd.Series(
                    0.0, index=area_df.index
                )

            fig.add_trace(
                go.Scatter(
                    x=area_df["Date"],
                    y=normalized,
                    mode="lines",
                    name=col,
                )
            )

        y_title = "Normalized value (0–1)"

    else:

        # Plot the selected property for every area
        for area_number, area_df in df_filtered.groupby(
            "Area_Number"
        ):

            area_df = area_df.sort_values("Date")

            fig.add_trace(
                go.Scatter(
                    x=area_df["Date"],
                    y=area_df[selected_col],
                    mode="lines",
                    name=f"Area {area_number}",
                )
            )

        y_title = selected_col

    # Shared layout and interactive features
    fig.update_layout(
        title=f"{selected_col} — {start_m} to {end_m}",
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",
        height=500,
        legend=dict(
            orientation="h",
            y=-0.25,
        ),
        margin=dict(l=20, r=20, t=60, b=80),
    )

    fig.update_xaxes(
        rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
    )

