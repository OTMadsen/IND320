import pandas as pd
import streamlit as st
import plotly.express as px


from services.data import load_electricity_data


def page_electricity():
    st.title("Electricity Flow Dashboard")
    st.markdown(
        """
        Investigate recorded international electricity flows between Norway and neighbouring electricity markets. 

        Select a connection and year to compare electricity flow directions and observe how recorded values
        develop over time

        The available dataset currently covers three international connections from Norways NO2 region.
        """
    )

    try:
        df = load_electricity_data()
    except Exception as e:
        st.error(f"Could not load MongoDB data: {e}")
        return

    if df.empty:
        st.info(
    "Dataset coverage: This dashboard currently includes "
    "three international electricity connections from NO2 "
    "(Germany, Denmark and the Netherlands). "
    "Other connections, including NO4–SE1, are not included "
    "in the current MongoDB collection."
)
        st.warning("No usable electricity records found.")
        return
    
    # Diagnostic: inspect available electricity connections
    with st.expander("Available electricity connections"):
        st.write("Total records:", len(df))
        st.write("Available columns:", df.columns.tolist())

        connections = sorted(
            df["connection"].dropna().unique().tolist()
        )

        st.write("Number of connections:", len(connections))
        st.write("Connection names:", connections)


        connection_summary = (
            df.groupby("connection")
            .agg(
                Observations=("value", "size"),
                First_Date=("timestamp", "min"),
                Last_Date=("timestamp", "max")
            )

            .reset_index()
            .sort_values("connection")
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Electricity flow by direction")

        connection = st.radio(
            "Select connection:",
            sorted(df["connection"].unique()),
            key="electricity_connection",
        )

        connection_df = df[df["connection"] == connection]
        year = st.selectbox(
            "Select year:",
            sorted(connection_df["year"].unique(), reverse=True),
            key="electricity_year",
        )

        year_df = connection_df[connection_df["year"] == year]
        totals = year_df.groupby("direction")["value"].sum()
        totals = totals[totals > 0]

        if totals.empty:
            st.info("No positive flow values available.")
        else:

            totals_df = totals.rename("total").reset_index()

            fig = px.pie(
                totals_df,
                names="direction",
                values="total",
                color="direction",
                hole=0.38,
                color_discrete_sequence=[
                    "#2B9978",
                    "#F0A32F",
                    "#3978BE"
                ],
                title=f"Electricity flow direction - {connection} ({year})",
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Share:%{percent}<br>"
                    "Summed recorded value: %{value:,.2f}"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=65, b=10),
                legend_title_text="Direction",
                legend=dict(
                    orientation="h",
                    y=-0.1,
                ),
            )

            st.caption(
                "Shared of summed recorded values by direction."
                "Energy totals required confirming the sampling"
                "interval and units"
            )
             
    with right:
        st.subheader("Electricity flow over time")

        available_months = sorted(year_df["month"].unique().tolist())
        selected_months = st.pills(
            "Select months:",
            options=available_months,
            default=available_months,
            selection_mode="multi",
            format_func=lambda month: pd.Timestamp(
                2000, int(month), 1
            ).strftime("%b"),
            key=f"electricity_months_{connection}_{year}",
        )

        plot_df = year_df[year_df["month"].isin(selected_months)]
        if plot_df.empty:
            st.info("Select at least one month with data.")
        else:
            chart_df = (
                plot_df.groupby(["timestamp", "direction"], as_index=False)["value"]
                .sum()
                .pivot(index="timestamp", columns="direction", values="value")
                .sort_index()
            )
            st.line_chart(
                chart_df,
                x_label="Date",
                y_label="Recorded flow value",
            )

    with st.expander("Data source and interpretation"):
        st.write(
            "Electricity flow data comes from ENTSO-E and is stored in "
            "MongoDB Atlas (database: ind320_project; collection: electricity_flows). "
            "Only available years and months can be selected. "
            "The pie chart compares summed recorded values; interpreting "
            "these totals as energy requires confirming units and sampling intervals."
        )
