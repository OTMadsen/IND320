import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from services.data import load_data, filter_nve_data
from ui.filters import select_nve_month_range


def page_data_plot():
    st.title("Page 3: Interactive Data Plotter")
    
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
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="whitegrid")

    if selected_col == "All columns together":
        area_number = st.selectbox(
            "Select area number:",
            sorted(df_filtered["Area_Number"].unique()),
        )
        area_df = df_filtered[df_filtered["Area_Number"] == area_number]

        for col in num_cols:
            mn = area_df[col].min()
            mx = area_df[col].max()
            if mx > mn:
                values = (area_df[col] - mn) / (mx - mn)
            else:
                values = pd.Series(0.0, index=area_df.index)
            ax.plot(area_df["Date"], values, label=f"{col} (Normalized)")

        ax.set_ylabel("Normalized Scale [0, 1]")
    else:
        for area_number, area_df in df_filtered.groupby("Area_Number"):
            ax.plot(
                area_df["Date"],
                area_df[selected_col],
                label=f"Area {area_number}",
            )
        ax.set_ylabel(selected_col)

    ax.set_xlabel("Date", fontweight="bold")

    ax.legend(loc="upper right")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
