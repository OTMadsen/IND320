from pathlib import Path
import pandas as pd
import requests
import streamlit as st

@st.cache_data(show_spinner=False, max_entries=10)
def weather_daily_from_hourly(hourly):
    data = hourly.copy()
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    for col in ("precipitation", "temperature_2m"):
        if col not in data.columns:
            raise ValueError(f"Weather data is missing '{col}'.")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["time"]).set_index("time").sort_index()
    daily = data.resample("D").agg(
        precipitation_mm=("precipitation", lambda x: x.sum(min_count=1)),
        temperature_c=("temperature_2m", "mean"),
    )
    if daily.empty:
        raise ValueError("No usable daily observations in the weather data.")
    return daily


@st.cache_data(show_spinner=False)
def download_era5_by_coordinate(region, latitude, longitude,
                                start_date, end_date):
    """Request ERA5 hourly data only after an explicit user action."""
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": latitude, "longitude": longitude,
            "start_date": start_date, "end_date": end_date,
            "hourly": ",".join([
                "temperature_2m", "precipitation", "rain", "snowfall",
                "relative_humidity_2m", "wind_speed_10m"
            ]),
            "models": "era5", "timezone": "UTC",
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise ValueError(f"No hourly values returned: {payload}")
    frame = pd.DataFrame(payload["hourly"])
    frame["region"] = region
    frame["latitude"] = latitude
    frame["longitude"] = longitude
    return frame

@st.cache_data(show_spinner=False)
def read_locations():

    project_root = Path(__file__).resolve().parent.parent


    csv_path = (
        project_root
        / "assignments"
        / "assignment3"
        / "electricity_sampled_locations.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find sampled locations {csv_path}. Please try again."
        )

    points = pd.read_csv(csv_path)

    required = {"region", "latitude", "longitude"}

    if not required.issubset(points.columns):
        raise ValueError(
            f"Missing columns: {required - set(points.columns)}"
        )

    points["region"] = (
        points["region"]
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.upper()
        
    )

    points["latitude"] = pd.to_numeric(
        points["latitude"], errors="coerce"
    )
    points["longitude"] = pd.to_numeric(
        points["longitude"], errors="coerce"
    )

    return points.dropna(
        subset=["longitude", "latitude"]
    )