import pandas as pd
import streamlit as st
from services.data import load_data, filter_nve_data
from ui.filters import select_nve_month_range


def page_data_table():
    
    st.title("Page 2: Data Overview & First Month Sparklines")
    df = load_data()

    area_type = st.radio(
        "Select Area Type:",
        ["EL", "NO", "VASS"]
    )
    
    df_no = (
        df[df["Area_Type"] == area_type]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    selected_range = select_nve_month_range(df, "table")
    if selected_range is None:
        return
    start_m, end_m = selected_range
    df_no = filter_nve_data(df_no, start_m, end_m)
    if df_no.empty:
        st.info("No reservoir data for this selection.")
        return

    first_month = df_no["YearMonth"].min()
    st.subheader(f"Data Summary ({start_m} to {end_m})")
    st.caption(f"Sparklines show the first selected month: {first_month}.")

    num_cols = df_no.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    table_data = []

    for area_number, area_df in df_no.groupby("Area_Number"):
        area_df = area_df.sort_values("Date")
        first_month_df = area_df[area_df["YearMonth"] == first_month]

        for col in num_cols:
            if col == "Area_Number":
                continue

            table_data.append({
                "Area Number": area_number,
                "Column Name": col,
                "Min": area_df[col].min(),
                "Max": area_df[col].max(),
                "Mean": area_df[col].mean(),
                "First Month Trend": first_month_df[col].tolist(),
            })

    summary_df = pd.DataFrame(table_data)
    st.dataframe(
        summary_df,
        column_config={
            "Area Number": st.column_config.NumberColumn("Area Number"),
            "Column Name": st.column_config.TextColumn("Column"),
            "Min": st.column_config.NumberColumn("Dataset Min", format="%.2f"),
            "Max": st.column_config.NumberColumn("Dataset Max", format="%.2f"),
            "Mean": st.column_config.NumberColumn(
                "Dataset Mean", format="%.2f"
            ),
            "First Month Trend": st.column_config.LineChartColumn(
                f"Trend ({first_month})"
            ),
        },
        use_container_width=True,
        hide_index=True,
    )
