import pandas as pd
import requests
import streamlit as st
from pymongo import MongoClient


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

    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .dt.tz_convert("Europe/Oslo")
    )

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    df["connection"] = (
        df["connection"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "", regex=False)
        .str.replace("N0_2", "NO_2", regex=False)
    )
    
    return df


def get_reservoir_series(region):
    """Weekly filling degree for an electricity area, NO1..NO5."""
    if region not in {"NO1", "NO2", "NO3", "NO4", "NO5"}:
        raise ValueError(f"Unsupported electricity region: {region}")

    df = load_data()
    selected = df[
        (df["Area_Type"] == "EL") &
        (pd.to_numeric(df["Area_Number"], errors="coerce") == int(region[-1]))
    ].copy()

    if selected.empty:
        raise ValueError(f"No reservoir measurements found for {region}")

    selected["Fillings_Degree"] = pd.to_numeric(
        selected["Fillings_Degree"], errors="coerce"
    )
    selected["Date"] = pd.to_datetime(selected["Date"], errors="coerce")

    weekly = (
        selected.dropna(subset=["Date", "Fillings_Degree"])
        .groupby("Date")["Fillings_Degree"].mean()
        .sort_index()
        .resample("W-SUN").mean()
    )
    # Fill only short, internal gaps. Do not silently fabricate long periods.
    weekly = weekly.interpolate("time", limit=2, limit_area="inside")
    if weekly.isna().any():
        raise ValueError(
            f"{weekly.isna().sum()} missing weeks remain for {region}. "
            "Review the reservoir data before decomposition."
        )
    if len(weekly) < 104:
        raise ValueError("At least 104 weekly observations are required.")
    return weekly
