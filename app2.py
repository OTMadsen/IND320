import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import requests
from pymongo import MongoClient

st.set_page_config(
    page_title="Hydro Reservoir Dashboard", layout="wide"
)


@st.cache_data(ttl=3600)
def load_data():
    url = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    df = pd.DataFrame(response.json())

    df = df.rename(columns={
        "dato_Id": "Date",
        "omrType": "Area_Type",
        "omrnr": "Area_Number",
        "fyllingsgrad": "Fillings_Degree",
        "fylling_TWh": "Filling_TWh",
        "kapasitet_TWh": "Capacity_TWh",
        "endring_fyllingsgrad": "Change_Filling_Degree"
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    return df

def filter_nve_data(df, start_month, end_month):
    return df[
        df["YearMonth"].between(start_month, end_month)
    ].copy()


def select_nve_month_range(df, page_key):
    """Keep the selected NVE month interval when switching between pages."""
    unique_months = sorted(df["YearMonth"].dropna().unique().tolist())
    if not unique_months:
        st.warning("No NVE months are available.")
        return None

    default_range = (
        unique_months[0],
        unique_months[min(11, len(unique_months) - 1)],
    )
    saved_range = st.session_state.get("nve_selected_range", default_range)
    if (saved_range[0] not in unique_months or
            saved_range[1] not in unique_months):
        saved_range = default_range

    widget_key = f"nve_month_slider_{page_key}"
    # Restore the shared selection when this page's widget is created.
    if widget_key not in st.session_state:
        st.session_state[widget_key] = saved_range

    selected = st.select_slider(
        "Select Month Range:",
        options=unique_months,
        key=widget_key,
    )
    if not isinstance(selected, tuple):
        selected = (selected, selected)
    st.session_state["nve_selected_range"] = selected
    return selected


@st.cache_resource
def get_mongo_client():
    return MongoClient(
        st.secrets["MONGODB_URI"],
        serverSelectionTimeoutMS=10000
    )

@st.cache_data(ttl=600)
def load_electricity_data():
    client = get_mongo_client()
    collection = client["ind320_project"]["electricity_flows"]

    records = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(records)

    required = {"timestamp", "connection", "direction", "value"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing MongoDB fields: {sorted(required - set(df.columns))}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["connection"] = (
        df["connection"]
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
    )

    df["direction"] = df["direction"].astype(str).str.strip()

    df = df.dropna(subset=["timestamp", "value"])

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    return df


# Page 1: Home
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


# Page 2: Data Table
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


# Page 3: Interactive Data Plotter
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


# Navigation Setup
pg = st.navigation([
    st.Page(page_home, title="Home"),
    st.Page(page_data_table, title="Data Table"),
    st.Page(page_data_plot, title="Data Plot"),
    st.Page(page_electricity, title="Electricity")
    
])

pg.run()