import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Hydro Reservoir Dashboard", layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("reservoirs_improved.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    return df


# Page 1: Home
def page_home():
    st.title("Norwegian Hydro Reservoir Dashboard")
    st.write("Welcome to the Norwegian Hydro Reservoir Analytics Application")
    st.header("Project Overview")
    st.markdown("""
     **Data Sources**: Norwegian Hydro Reservoirs Dataset (`reservoirs_improved.csv`)  
     **Time Range**: 1995–2026  
     **Features**: Interactive tables and custom plots.
     """)


# Page 2: Data Table
def page_data_table():
    st.title("Page 2: Data Overview & First Month Sparklines")
    df = load_data()
    df_no = (
        df[df["Area_Type"] == "NO"].sort_values("Date").reset_index(drop=True)
    )

    first_month = df_no["YearMonth"].min()
    df_first_month = df_no[df_no["YearMonth"] == first_month]

    st.subheader(f"Data Summary (First Month: {first_month})")

    num_cols = df_no.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    table_data = []
    for col in num_cols:
        table_data.append({
            "Column Name": col,
            "Min": df_no[col].min(),
            "Max": df_no[col].max(),
            "Mean": df_no[col].mean(),
            "First Month Trend": df_first_month[col].tolist(),
        })

    summary_df = pd.DataFrame(table_data)
    st.dataframe(
        summary_df,
        column_config={
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


# Page 3: Interactive Data Plotter
def page_data_plot():
    st.title("Page 3: Interactive Data Plotter")
    df = load_data()
    df_no = (
        df[df["Area_Type"] == "NO"].sort_values("Date").reset_index(drop=True)
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

    unique_months = sorted(df_no["YearMonth"].unique().tolist())
    first_month = unique_months[0]

    selected_months = st.select_slider(
        "Select Month Range:",
        options=unique_months,
        value=(first_month, unique_months[min(11, len(unique_months) - 1)]),
    )

    if isinstance(selected_months, tuple):
        start_m, end_m = selected_months
    else:
        start_m = end_m = selected_months

    df_filtered = df_no[
        (df_no["YearMonth"] >= start_m) & (df_no["YearMonth"] <= end_m)
    ]
    st.subheader(f"Plotting: {selected_col} ({start_m} – {end_m})")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="whitegrid")

    if selected_col == "All columns together":
        for col in num_cols:
            mn = df_filtered[col].min()
            mx = df_filtered[col].max()

            if mx > mn:
                values = (df_filtered[col] - mn) / (mx - mn)
            else:
                values = pd.Series(0.0, index=df_filtered.index)

            ax.plot(df_filtered["Date"], values, label=f"{col} (Normalized)")
        ax.set_ylabel("Normalized Scale [0, 1]")
    else:
        ax.plot(
            df_filtered["Date"],
            df_filtered[selected_col],
            label=selected_col,
        )
        ax.set_xlabel("Date", fontweight="bold")
        ax.set_ylabel(selected_col)

    ax.legend(loc="upper right")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)



# Navigation Setup
pg = st.navigation([
    st.Page(page_home, title="Home"),
    st.Page(page_data_table, title="Data Table"),
    st.Page(page_data_plot, title="Data Plot"),
    
])

pg.run()