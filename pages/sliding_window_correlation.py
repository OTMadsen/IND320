"""
Interactive sliding-window weather/reservoir correlation for IND320.

"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pathlib import Path

from services.data import get_reservoir_series
from services.weather import read_locations, download_era5_by_coordinate
from ui.filters import get_selected_electricity_area


# Hourly columns from the existing Open-Meteo ERA5 download function.
# Rain/precipitation = millimetres per week; snowfall = centimetres per week.
WEATHER_PROPERTIES = {
    "Temperature": ("temperature_2m", "°C", "mean"),
    "Precipitation": ("precipitation", "mm/week", "sum"),
    "Rain": ("rain", "mm/week", "sum"),
    "Snowfall": ("snowfall", "cm/week", "sum"),
    "Relative humidity": ("relative_humidity_2m", "%", "mean"),
    "Wind speed": ("wind_speed_10m", "km/h", "mean"),
}


def available_properties(hourly):
    """Only show ERA5 fields containing usable measurements."""
    return [
        name for name, (column, _, _) in WEATHER_PROPERTIES.items()
        if column in hourly.columns
        and pd.to_numeric(hourly[column], errors="coerce").notna().any()
    ]

@st.cache_data(show_spinner=False, max_entries=12)
def weekly_weather_series(hourly, property_name):
    """Aggregate hourly observations to the same Sunday-ending weeks as NVE.

    Exclude weeks with fewer than 120 valid hourly measurements (five days),
    because incomplete boundary weeks can distort weekly precipitation totals.
    """
    column, _, aggregation = WEATHER_PROPERTIES[property_name]
    if "time" not in hourly.columns or column not in hourly.columns:
        raise ValueError(f"Weather data requires 'time' and '{column}'.")

    frame = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"], utc=True, errors="coerce"),
        "value": pd.to_numeric(hourly[column], errors="coerce"),
    }).dropna(subset=["time"]).set_index("time").sort_index()

    if frame.empty:
        raise ValueError("There are no valid weather timestamps.")

    grouped = frame["value"].resample("W-SUN")
    weekly = (
        grouped.sum(min_count=1) if aggregation == "sum" else grouped.mean()
    )
    weekly[grouped.count() < 120] = np.nan
    weekly.index = weekly.index.tz_localize(None)
    weekly.name = "weather"
    return weekly

st.cache_data(show_spinner=False, max_entries=12)
def aligned_weekly_data(reservoir, weather, lag_weeks, window_weeks):
    """Align weeks and calculate Pearson r on *consecutive calendar weeks*.

    Positive lag = earlier weather compared against a later reservoir reading.
    Negative lag = earlier reservoir compared against later weather.
    Missing weeks are NOT compressed out of the rolling-window calculation.
    """
    reservoir = pd.Series(
        pd.to_numeric(reservoir, errors="coerce").to_numpy(),
        index=pd.to_datetime(reservoir.index, utc=True).tz_localize(None),
        name="reservoir",
    )
    reservoir = reservoir.sort_index().resample("W-SUN").mean()
    weather = weather.sort_index().resample("W-SUN").mean().rename("weather")

    # Retain an evenly spaced weekly index before shifting and rolling.
    joined = pd.concat([reservoir, weather], axis=1).sort_index()
    joined = joined.asfreq("W-SUN")
    joined["weather_lagged"] = joined["weather"].shift(lag_weeks)
    joined["rolling_r"] = (
        joined["reservoir"]
        .rolling(window=window_weeks, min_periods=window_weeks)
        .corr(joined["weather_lagged"])
    )
    joined["reservoir_pct"] = joined["reservoir"] * 100
    return joined


def make_timeseries_figure(joined, label, unit, lag):
    # Plot only the ERA5 overlap, not the entire 30-year NVE history.
    weather_dates = joined.index[joined["weather_lagged"].notna()]
    plotted = joined.loc[weather_dates.min():weather_dates.max()]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=plotted.index, y=plotted["reservoir_pct"],
                   name="Reservoir filling", mode="lines"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=plotted.index, y=plotted["weather_lagged"],
                   name=f"{label} (lag-adjusted)", mode="lines"),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Weekly reservoir filling vs. {label.lower()} (lag: {lag:+d} weeks)",
        hovermode="x unified", legend=dict(orientation="h", y=1.12),
        height=420,
    )
    fig.update_yaxes(title_text="Reservoir filling (%)", secondary_y=False)
    fig.update_yaxes(title_text=f"{label} ({unit})", secondary_y=True)
    fig.update_xaxes(title_text="Week ending Sunday")
    return fig


def make_rolling_figure(joined, window, lag):
    plotted = joined.loc[joined["rolling_r"].notna()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plotted.index, y=plotted["rolling_r"], mode="lines+markers",
        name="Pearson r", marker=dict(size=5),
        hovertemplate="Week: %{x|%Y-%m-%d}<br>r = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", opacity=0.5)
    fig.update_layout(
        title=f"Rolling Pearson correlation · {window}-week window · lag {lag:+d}",
        xaxis_title="Window end (Sunday)", yaxis_title="Pearson correlation (r)",
        yaxis=dict(range=[-1.05, 1.05]), height=400, hovermode="x unified",
    )
    return fig

@st.cache_data(show_spinner=False, max_entries=12)
def make_window_animation(joined, property_name, unit, window):
    """Animated scatter: each frame represents one consecutive weekly window."""
    valid_ends = joined.index[joined["rolling_r"].notna()].tolist()
    if not valid_ends:
        return None

    # Limit the number of frames for responsiveness with long date intervals.
    stride = max(1, math.ceil(len(valid_ends) / 75))
    selected_ends = valid_ends[::stride]
    if selected_ends[-1] != valid_ends[-1]:
        selected_ends.append(valid_ends[-1])

    def window_points(end):
        pos = joined.index.get_loc(end)
        part = joined.iloc[pos - window + 1:pos + 1]
        return part["weather_lagged"], part["reservoir_pct"]

    first_x, first_y = window_points(selected_ends[0])
    fig = go.Figure(data=[go.Scatter(
        x=first_x, y=first_y, mode="markers", marker=dict(size=11),
        name="Weekly pairs",
        hovertemplate=f"{property_name}: %{{x:.2f}} {unit}<br>Filling: %{{y:.1f}}%<extra></extra>",
    )])

    # Fix axes across frames so movement does not come from autoscaling.
    first_position = joined.index.get_loc(selected_ends[0]) - window + 1
    last_position = joined.index.get_loc(selected_ends[-1])
    visible = joined.iloc[first_position:last_position + 1].dropna(
        subset=["reservoir_pct", "weather_lagged"]
    )
    xmin, xmax = visible["weather_lagged"].min(), visible["weather_lagged"].max()
    ymin, ymax = visible["reservoir_pct"].min(), visible["reservoir_pct"].max()
    dx = max(float(xmax - xmin) * 0.07, 0.5)
    dy = max(float(ymax - ymin) * 0.07, 1.0)

    def title_for(end):
        r = joined.loc[end, "rolling_r"]
        return f"Window ending {end:%Y-%m-%d} · Pearson r = {r:+.3f}"

    fig.frames = [
        go.Frame(
            name=end.strftime("%Y-%m-%d"),
            data=[go.Scatter(x=window_points(end)[0], y=window_points(end)[1])],
            layout=go.Layout(title=title_for(end)),
        )
        for end in selected_ends
    ]
    fig.update_layout(
        title=title_for(selected_ends[0]),
        xaxis=dict(title=f"{property_name} ({unit})", range=[xmin - dx, xmax + dx]),
        yaxis=dict(title="Reservoir filling (%)", range=[ymin - dy, ymax + dy]),
        height=460,
        updatemenus=[dict(
            type="buttons", showactive=False, direction="left",
            x=0, y=1.13,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 270, "redraw": True},
                                  "transition": {"duration": 0},
                                  "fromcurrent": True}]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.05, y=-0.04, len=0.9,
            currentvalue={"prefix": "Window end: "},
            steps=[dict(
                label=end.strftime("%Y-%m-%d"), method="animate",
                args=[[end.strftime("%Y-%m-%d")],
                      {"frame": {"duration": 0, "redraw": True},
                       "mode": "immediate", "transition": {"duration": 0}}],
            ) for end in selected_ends],
        )],
        margin=dict(t=85, b=85),
    )
    return fig


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WEATHER_CSV = (
    ROOT / "assignments" / "assignment4" / "NO3_ERA5_for_2024.csv"
)

@st.cache_data(show_spinner=False)
def load_default_weather_csv():
    return pd.read_csv(DEFAULT_WEATHER_CSV)

def page_sliding_window_correlation():
    region = get_selected_electricity_area()
    st.title("Sliding Window Correlation")
    st.markdown(
        """
        About this analysis:

        Explore the relationship between weather conditions and reservoir filling levels over time. 

        THis analysis calculates **rolling Pearson correlations**
        using weekly reservoir measurements and ERA5 weather data. 

        Adjust the window size, time lag, and weather property to investigate how relationships
        change over different periods. 

        **Note:** Correlation describes statistical relationships, but does not establish causation. 
        
        """
    )
    st.caption(f"Electricity area: {region} · Weekly NVE reservoir data and ERA5 weather")
    st.info(
        "Weather is measured at one sampled location within the electricity area, "
        "not averaged across the entire region. Correlation does not establish causation."
    )

    # A dataset loaded on the Weather Anomalies page is shared through session_state.
    #hourly = st.session_state.get("weather_dataset")

    hourly = st.session_state.get("weather_dataset")

    if DEFAULT_WEATHER_CSV.is_file():

        hourly = load_default_weather_csv()

        if "region" not in hourly.columns:
            hourly["region"] = "NO3"

        st.session_state["weather_dataset"] = hourly

        st.success(
            "Default ERA5 weather dataset loaded automatically"
        )

    else:
        st.warning(
            f"Default weather CSV not found:{DEFAULT_WEATHER_CSV}"
        )

    with st.expander("Weather data — reuse, download or upload", expanded=hourly is None):
        if hourly is not None:
            st.caption("A weather dataset is already present in this Streamlit session.")

        csv = st.file_uploader("Or upload an hourly ERA5 CSV", type=["csv"],
                               key="correlation_weather_upload")
        if csv is not None and st.button("Use uploaded CSV", key="correlation_csv_button"):
            try:
                imported = pd.read_csv(csv)
                required = {"time", "region"}
                if not required.issubset(imported.columns):
                    raise ValueError("CSV must contain 'time' and 'region' columns.")
                st.session_state["weather_dataset"] = imported
                hourly = imported
                st.success("Hourly weather CSV loaded.")
            except Exception as exc:
                st.error(f"Could not load CSV: {exc}")

        try:
            locations = read_locations()
            region_locations = locations.loc[locations["region"] == region].reset_index(drop=True)
        except Exception as exc:
            region_locations = pd.DataFrame()
            st.caption(f"Sampled coordinates unavailable: {exc}")

        if not region_locations.empty:
            sample_index = st.number_input(
                "Sampled location index", min_value=0,
                max_value=len(region_locations) - 1, value=0,
                key="correlation_sample_index",
            )
            left, right = st.columns(2)
            with left:
                start = st.date_input("ERA5 start", value=pd.Timestamp("2024-01-01").date(),
                                      key="correlation_start")
            with right:
                end = st.date_input("ERA5 end", value=pd.Timestamp("2024-12-31").date(),
                                    key="correlation_end")
            if st.button("Download ERA5 for correlation", disabled=start > end,
                         key="correlation_download_button"):
                point = region_locations.iloc[int(sample_index)]
                try:
                    with st.spinner("Downloading hourly ERA5 observations..."):
                        hourly = download_era5_by_coordinate(
                            region, float(point["latitude"]), float(point["longitude"]),
                            start.isoformat(), end.isoformat(),
                        )
                    st.session_state["weather_dataset"] = hourly
                    st.success(f"Loaded {len(hourly):,} hourly weather rows.")
                except Exception as exc:
                    st.error(f"Weather download failed: {exc}")

    if hourly is None:
        st.info("Load weather data here or on the Weather Anomalies page first.")
        return

    if not {"time", "region"}.issubset(hourly.columns):
        st.error("The selected weather dataset must contain time and region columns.")
        return

    source_regions = set(
        hourly["region"].astype(str).str.upper().str.strip()
        .str.replace(" ", "", regex=False)
    )
    if source_regions != {region}:
        st.warning(
            f"Weather data covers {', '.join(sorted(source_regions))}, but the selected "
            f"electricity area is {region}. Load weather data for {region}."
        )
        return

    properties = available_properties(hourly)
    if not properties:
        st.error("No supported ERA5 property with numeric observations was found.")
        return

    try:
        reservoir = get_reservoir_series(region)
    except Exception as exc:
        st.error(f"Could not load reservoir history: {exc}")
        return

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        property_name = st.selectbox(
            "Weather property", options=properties,
            index=properties.index("Temperature") if "Temperature" in properties else 0,
            key="correlation_property",
        )
    with col_b:
        lag = st.slider("Lag (weeks)", min_value=-12, max_value=12, value=0,
                        key="correlation_lag")
    weather = weekly_weather_series(hourly, property_name)
    prejoined = aligned_weekly_data(reservoir, weather, lag_weeks=lag, window_weeks=4)
    paired_weeks = int(prejoined[["reservoir", "weather_lagged"]].dropna().shape[0])
    max_window = min(52, paired_weeks)
    if max_window < 4:
        st.warning("Fewer than four overlapping weekly observations are available.")
        return
    with col_c:
        window = st.slider("Correlation window (weeks)", min_value=4,
                           max_value=max_window, value=min(12, max_window),
                           key="correlation_window")

    joined = aligned_weekly_data(reservoir, weather, lag_weeks=lag, window_weeks=window)
    valid = joined["rolling_r"].dropna()
    if valid.empty:
        st.warning(
            "No complete correlation window is available. Try a shorter window, "
            "smaller lag, or longer ERA5 date range."
        )
        return

    _, unit, aggregation = WEATHER_PROPERTIES[property_name]
    st.caption(
        "Lag interpretation: +4 means weather four weeks earlier than the reservoir "
        "measurement; −4 means weather four weeks later. "
        f"Weather uses weekly {'total' if aggregation == 'sum' else 'mean'} values; "
        "the final partial ERA5 week is excluded when insufficiently sampled."
    )

    a, b, c = st.columns(3)
    a.metric("Latest Pearson r", f"{valid.iloc[-1]:+.3f}")
    b.metric("Complete windows", f"{len(valid)}")
    c.metric("Overlapping paired weeks", f"{paired_weeks}")

    st.subheader("Weekly series")
    st.plotly_chart(make_timeseries_figure(joined, property_name, unit, lag),
                    use_container_width=True)

    st.subheader("Sliding-window Pearson correlation")
    st.plotly_chart(make_rolling_figure(joined, window, lag), use_container_width=True)

    st.subheader("Animated window-by-window relationship")
    st.caption("Press Play to see the weather–reservoir pairs move through time.")
    animated = make_window_animation(joined, property_name, unit, window)
    if animated is not None:
        st.plotly_chart(animated, use_container_width=True)

    st.subheader("Correlation results")
    result = joined.loc[joined["rolling_r"].notna(),
                        ["reservoir_pct", "weather_lagged", "rolling_r"]].copy()
    result.index.name = "Week ending"
    result = result.rename(columns={
        "reservoir_pct": "Reservoir filling (%)",
        "weather_lagged": f"Lag-adjusted {property_name} ({unit})",
        "rolling_r": "Pearson r",
    })
    st.dataframe(result.sort_index(ascending=False), use_container_width=True)
    st.download_button(
        "Download correlation results (CSV)",
        data=result.to_csv().encode("utf-8"),
        file_name=f"correlation_{region}_{property_name.lower().replace(' ', '_')}.csv",
        mime="text/csv", key="correlation_csv_export",
    )

    with st.expander("Suggested notebook log entry"):
        st.markdown(
            f"**Region:** {region}  \n"
            f"**Weather property:** {property_name} ({unit})  \n"
            f"**Window:** {window} weeks  \n"
            f"**Lag:** {lag:+d} weeks  \n"
            f"**Latest rolling Pearson r:** {valid.iloc[-1]:+.3f}  \n"
            f"**Period with valid windows:** {valid.index.min():%Y-%m-%d} "
            f"to {valid.index.max():%Y-%m-%d}  \n\n"
            "**Interpretation:** Describe whether correlation is stable or changes "
            "sign over time, compare at least two lags, and note that the weather "
            "series represents one sampled coordinate rather than an area mean."
        )
