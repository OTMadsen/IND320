import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from services.data import load_electricity_data


def page_electricity():
    st.title("Electricity Flow Dashboard")

    try:
        df = load_electricity_data()
    except Exception as e:
        st.error(f"Could not load MongoDB data: {e}")
        return

    if df.empty:
        st.warning("No usable electricity records found.")
        return

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
            fig, ax = plt.subplots()
            ax.pie(
                totals,
                labels=totals.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title(f"{connection} - {year}")
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Share of summed recorded values by direction. "
                "Energy totals require confirming the sampling interval and units."
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
