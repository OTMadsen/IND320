"""Interactive River SNARIMAX forecasting page for IND320 Assignment 4.

The default configuration reproduces the validated notebook setup: 260 weeks,
p=2, d=q=0, m=52, sp=1, and a 12-week horizon. Exogenous weather and
exchange-rate inputs (when available) are experimental scenarios, not validated
operational future forecasts.

"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.data import get_reservoir_series
from ui.filters import get_selected_electricity_area


ROOT = Path(__file__).resolve().parent.parent
RESERVOIR_CSV = ROOT / "assignments" / "assignment1" / "reservoirs.csv"
WEATHER_CSV = ROOT / "assignments" / "assignment4" / "NO3_ERA5_for_2024.csv"
CURRENCY_CANDIDATES = (
    ROOT / "assignments" / "assignment4" / "exchange_rates.csv",
    ROOT / "assignments" / "assignment4" / "exchange_rates_2024.csv",
    ROOT / "assignments" / "assignment4" / "currency_weekly.csv",
)

# Source name -> (hourly CSV column, weekly aggregation, unit)
WEATHER_FEATURES = {
    "Temperature (ERA5)": ("temperature_2m", "mean", "°C"),
    "Precipitation (ERA5)": ("precipitation", "sum", "mm/week"),
    "Rain (ERA5)": ("rain", "sum", "mm/week"),
    "Snowfall (ERA5)": ("snowfall", "sum", "cm/week"),
    "Wind speed (ERA5)": ("wind_speed_10m", "mean", "km/h"),
    "Relative humidity (ERA5)": ("relative_humidity_2m", "mean", "%"),
}


def _normalise_region(value):
    return str(value).strip().upper().replace(" ", "")


@st.cache_data(show_spinner=False)
def _local_reservoir_series(region):
    """Load NVE weekly data locally; fall back to existing service if absent."""
    if not RESERVOIR_CSV.exists():
        return get_reservoir_series(region)

    df = pd.read_csv(RESERVOIR_CSV)
    required = {"dato_Id", "omrType", "omrnr", "fyllingsgrad"}
    if not required.issubset(df.columns):
        raise ValueError(f"reservoirs.csv is missing: {sorted(required - set(df.columns))}")

    df["omrType"] = df["omrType"].astype(str).str.strip().str.upper()
    df["omrnr"] = pd.to_numeric(df["omrnr"], errors="coerce")
    df["dato_Id"] = pd.to_datetime(df["dato_Id"], errors="coerce")
    df["fyllingsgrad"] = pd.to_numeric(df["fyllingsgrad"], errors="coerce")
    number = int(region[-1])
    subset = df.loc[(df["omrType"] == "EL") & (df["omrnr"] == number)]
    if subset.empty:
        raise ValueError(f"No EL reservoir observations for {region}")

    weekly = (
        subset.dropna(subset=["dato_Id", "fyllingsgrad"])
        .groupby("dato_Id")["fyllingsgrad"]
        .mean()
        .sort_index()
        .resample("W-SUN").mean()
    )
    # Only fill small, internal gaps, as in the existing NVE data service.
    weekly = weekly.interpolate("time", limit=2, limit_area="inside")
    if weekly.isna().any():
        raise ValueError("Reservoir history has longer gaps that need review.")
    weekly.name = "reservoir"
    return weekly


@st.cache_data(show_spinner=False)
def _read_weather_csv():
    return pd.read_csv(WEATHER_CSV)


def _weather_hourly_for_region(region):
    """Prefer the active Streamlit weather data; use bundled NO3 CSV by default."""
    hourly = st.session_state.get("weather_dataset")
    if hourly is not None and "time" in hourly.columns:
        source_regions = (
            hourly["region"].map(_normalise_region).dropna().unique()
            if "region" in hourly.columns else []
        )
        if len(source_regions) == 1 and source_regions[0] == region:
            return hourly, "Weather data loaded in this Streamlit session"

    if region == "NO3" and WEATHER_CSV.is_file():
        hourly = _read_weather_csv()
        if "region" not in hourly.columns or set(
            hourly["region"].map(_normalise_region).dropna().unique()
        ) == {"NO3"}:
            return hourly, f"Bundled file: {WEATHER_CSV.name}"
    return None, None


def _weather_weekly(hourly):
    """Aggregate hourly measurements to Sunday-ending weeks, without short weeks."""
    times = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
    weekly_features = {}
    for label, (column, aggregation, _unit) in WEATHER_FEATURES.items():
        if column not in hourly.columns:
            continue
        values = pd.to_numeric(hourly[column], errors="coerce")
        data = pd.DataFrame({"date": times, "value": values}).dropna(subset=["date"])
        if data.empty or not data["value"].notna().any():
            continue
        grouped = data.set_index("date").sort_index()["value"].resample("W-SUN")
        weekly = grouped.sum(min_count=1) if aggregation == "sum" else grouped.mean()
        weekly[grouped.count() < 120] = np.nan  # Fewer than five days recorded
        weekly.index = weekly.index.tz_localize(None)
        weekly_features[label] = weekly
    return pd.DataFrame(weekly_features)


@st.cache_data(show_spinner=False)
def _currency_weekly(path_string):
    df = pd.read_csv(path_string)
    date_column = next((name for name in ("Date", "date", "time") if name in df), None)
    if date_column is None:
        raise ValueError("Currency CSV needs a 'Date' column")
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    columns = [name for name in ("EUR_NOK", "USD_NOK") if name in df.columns]
    if not columns:
        raise ValueError("Currency CSV needs EUR_NOK or USD_NOK")
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[date_column]).set_index(date_column).sort_index()
    weekly = df[columns].resample("W-SUN").last()
    return weekly.rename(columns={"EUR_NOK": "EUR/NOK", "USD_NOK": "USD/NOK"})


def _calendar_features(date):
    """Cyclic calendar inputs are available at prediction time."""
    week = int(pd.Timestamp(date).isocalendar().week)
    angle = 2 * math.pi * (week - 1) / 52.1775
    return {"week_sin": math.sin(angle), "week_cos": math.cos(angle)}


def _forecast_model(train, horizon, p, d, q, m, sp, sd, sq,
                    selected_features, scenario="Hold last observed value"):
    """Train online on ordered historical weeks and make a genuine future forecast."""
    from river import time_series  # Keep the rest of the app importable if river is missing.

    target = train["reservoir"]
    if len(train) < 26 or target.isna().any():
        raise ValueError("At least 26 complete historical weeks are required.")
    if not (train.index.to_series().diff().dropna() == pd.Timedelta(days=7)).all():
        raise ValueError("Training data contains missing calendar weeks. Select a continuous period.")

    model = time_series.SNARIMAX(p=p, d=d, q=q, m=m, sp=sp, sd=sd, sq=sq)
    for week, y in target.items():
        x = {}
        if "Calendar seasonality" in selected_features:
            x.update(_calendar_features(week))
        for name in selected_features:
            if name in train.columns:
                x[name] = float(train.at[week, name])
        model.learn_one(float(y), x=x or None)

    dates = pd.date_range(train.index[-1] + pd.Timedelta(weeks=1), periods=horizon,
                          freq="W-SUN")
    future_xs = []
    for week in dates:
        x = {}
        if "Calendar seasonality" in selected_features:
            x.update(_calendar_features(week))
        for name in selected_features:
            if name in train.columns:
                values = train[name]
                if scenario == "Historical training average":
                    x[name] = float(values.mean())
                else:
                    x[name] = float(values.iloc[-1])
        future_xs.append(x)

    raw = model.forecast(horizon=horizon, xs=future_xs)
    preds = pd.Series(np.clip(np.asarray(raw, dtype=float), 0.0, 1.0),
                      index=dates, name="River forecast")
    if not np.isfinite(preds.to_numpy()).all():
        raise ValueError("River returned non-finite predictions; try simpler model settings.")
    return preds


def _forecast_figure(history, forecast):
    fig = go.Figure()
    recent = history.iloc[-min(104, len(history)):]
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent.to_numpy() * 100, mode="lines",
        name="Historical reservoir filling",
    ))
    fig.add_trace(go.Scatter(
        x=[recent.index[-1], *forecast.index],
        y=[float(recent.iloc[-1] * 100), *(forecast.to_numpy() * 100)],
        mode="lines+markers", name="River SNARIMAX forecast",
        line=dict(dash="dash"),
    ))
    fig.add_vline(x=recent.index[-1].timestamp() * 1000, line_dash="dot",
                  annotation_text="Forecast origin", annotation_position="top left")
    fig.update_layout(
        title="Weekly NVE reservoir filling — historical and forecast",
        xaxis_title="Week ending Sunday", yaxis_title="Reservoir filling (%)",
        yaxis=dict(range=[0, 100]), hovermode="x unified", height=450,
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def _animated_figure(history, forecast):
    """Plotly animation reveals the forecast one week at a time."""
    recent = history.iloc[-min(52, len(history)):]
    dates = [recent.index[-1], *list(forecast.index)]
    values = [float(recent.iloc[-1] * 100), *(forecast.to_numpy() * 100)]
    fig = go.Figure(data=[
        go.Scatter(x=recent.index, y=recent.to_numpy() * 100,
                   mode="lines", name="Observed"),
        go.Scatter(x=dates[:1], y=values[:1], mode="lines+markers",
                   name="Forecast unfolding", line=dict(dash="dash")),
    ])
    fig.frames = [
        go.Frame(
            data=[go.Scatter(x=dates[:i + 1], y=values[:i + 1],
                             mode="lines+markers", line=dict(dash="dash"))],
            traces=[1], name=str(i),
        )
        for i in range(1, len(dates))
    ]
    fig.update_layout(
        title="River forecast — week-by-week animation",
        xaxis_title="Week ending Sunday", yaxis_title="Reservoir filling (%)",
        yaxis=dict(range=[0, 100]), height=460,
        updatemenus=[dict(type="buttons", direction="left", showactive=False,
                          x=0.01, y=1.13,
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, {"frame": {"duration": 550, "redraw": True},
                                                "transition": {"duration": 0},
                                                "fromcurrent": True}]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], {"mode": "immediate",
                                                  "frame": {"duration": 0, "redraw": False}}]),
                          ])],
        sliders=[dict(currentvalue={"prefix": "Forecast week: "},
                      steps=[dict(method="animate", label=str(i),
                                  args=[[str(i)], {"mode": "immediate",
                                                   "frame": {"duration": 0, "redraw": True}}])
                             for i in range(1, len(dates))])],
    )
    return fig


def page_river_forecasting():
    st.title("River SNARIMAX Forecasting")
    region = get_selected_electricity_area()
    st.caption(f"Electricity area: {region} · NVE reservoir filling · weekly (W-SUN)")
    st.write(
        "Forecast future reservoir filling with River's online SNARIMAX model. "
        "The default settings follow our evaluated Assignment 4 notebook."
    )

    try:
        series = _local_reservoir_series(region)
    except Exception as exc:
        st.error(f"Could not load reservoir history: {exc}")
        return
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.resample("W-SUN").mean().rename("reservoir")
    if len(series) < 52 or series.isna().any():
        st.error("A continuous reservoir history of at least 52 weeks is needed.")
        return

    hourly, weather_source = _weather_hourly_for_region(region)
    weekly_weather = _weather_weekly(hourly) if hourly is not None else pd.DataFrame()
    currency_path = next((p for p in CURRENCY_CANDIDATES if p.is_file()), None)
    weekly_currency = pd.DataFrame()
    if currency_path:
        try:
            weekly_currency = _currency_weekly(str(currency_path))
        except (ValueError, KeyError, TypeError) as exc:
            st.warning(f"Could not use currency CSV: {exc}")

    option_labels = ["Calendar seasonality"]
    option_labels.extend(weekly_weather.columns.tolist())
    option_labels.extend(weekly_currency.columns.tolist())

    selected_features = st.multiselect(
        "Additional predictor variables (optional)",
        options=option_labels, default=[],
        help="Reservoir history is always used. Calendar features are known for future weeks. "
             "ERA5/currency features use an explicitly selected future scenario.",
    )
    external = [name for name in selected_features if name != "Calendar seasonality"]
    aligned = series.to_frame()
    if any(name in weekly_weather for name in external):
        aligned = aligned.join(weekly_weather[[n for n in external if n in weekly_weather]])
    if any(name in weekly_currency for name in external):
        aligned = aligned.join(weekly_currency[[n for n in external if n in weekly_currency]])
    aligned = aligned.dropna(subset=["reservoir", *external])
    if aligned.empty:
        st.error("No overlapping NVE and external measurements are available for this selection.")
        return

    latest = aligned.index.max()
    earliest = aligned.index.min()
    map_date = st.session_state.get("selected_date", latest.date())
    try:
        proposed = pd.Timestamp(map_date)
    except (ValueError, TypeError):
        proposed = latest
    proposed = min(max(proposed, earliest), latest)
    possible_origins = aligned.index[aligned.index <= proposed]
    default_origin = possible_origins[-1] if len(possible_origins) else latest

    col1, col2 = st.columns(2)
    with col1:
        chosen_date = st.date_input(
            "Forecast origin (last observed week)",
            value=default_origin.date(), min_value=earliest.date(), max_value=latest.date(),
            help="Defaults to the selected map date when available; otherwise the newest compatible week.",
        )
    origin_options = aligned.index[aligned.index <= pd.Timestamp(chosen_date)]
    if origin_options.empty:
        st.warning("There are no observations before the chosen date.")
        return
    origin = origin_options[-1]
    st.session_state["selected_date"] = origin.date()

    aligned = aligned.loc[:origin]
    complete_weeks = (aligned.index.to_series().diff() == pd.Timedelta(days=7))
    # Keep the last uninterrupted span, rather than making missing weeks disappear.
    breaks = np.flatnonzero(~complete_weeks.to_numpy())
    if len(breaks):
        aligned = aligned.iloc[int(breaks[-1]):]
    n_available = len(aligned)
    if n_available < 26:
        st.warning(f"Only {n_available} consecutive overlapping weeks. Need at least 26.")
        return

    choices = [w for w in (26, 52, 78, 104, 156, 208, 260, 364, 520) if w <= n_available]
    if not choices:
        st.error("Not enough training weeks.")
        return
    with col2:
        history_weeks = st.select_slider(
            "Training history (weeks)", options=choices,
            value=260 if 260 in choices else choices[-1],
        )
    horizon = st.slider("Forecast horizon (weeks)", min_value=1, max_value=26,
                        value=12, step=1)

    with st.expander("River SNARIMAX model parameters", expanded=False):
        a, b, c, d_col = st.columns(4)
        with a:
            p = int(st.number_input("AR order (p)", min_value=0, max_value=16, value=2))
        with b:
            d = int(st.number_input("Differencing (d)", min_value=0, max_value=2, value=0))
        with c:
            q = int(st.number_input("MA order (q)", min_value=0, max_value=12, value=0))
        with d_col:
            m = int(st.number_input("Seasonal period (m)", min_value=2, max_value=104, value=52))
        a, b, c = st.columns(3)
        with a:
            sp = int(st.number_input("Seasonal AR (sp)", min_value=0, max_value=3, value=1))
        with b:
            sd = int(st.number_input("Seasonal differencing (sd)", min_value=0, max_value=2, value=0))
        with c:
            sq = int(st.number_input("Seasonal MA (sq)", min_value=0, max_value=3, value=0))

    scenario = "Hold last observed value"
    if external:
        st.warning(
            "Experimental external-input scenario: ERA5 and exchange rates are historical data, "
            "not known future observations. These models were not validated in our notebook "
            "as operational forecasts."
        )
        scenario = st.selectbox(
            "Assumption for future external variables",
            ["Hold last observed value", "Historical training average"],
        )
        if weather_source:
            st.caption(f"ERA5 source: {weather_source}")
        if currency_path:
            st.caption(f"Currency source: {currency_path.name}")
    if history_weeks < m * max(1, sp, sd, sq):
        st.warning("The training history is shorter than the seasonal memory; the "
                   "model may not learn the seasonal component reliably.")
    if external and latest < series.index.max():
        st.info(
            f"External inputs end {latest:%Y-%m-%d}, so this scenario is anchored "
            "to their last overlapping observation rather than the newest NVE week."
        )

    training = aligned.iloc[-history_weeks:]
    # Ensure user sees the correct last observation date and avoids false date labels.
    st.caption(f"Training: {training.index[0]:%Y-%m-%d} to "
               f"{training.index[-1]:%Y-%m-%d} ({len(training)} weekly observations).")
    try:
        predictions = _forecast_model(
            training, horizon, p, d, q, m, sp, sd, sq, selected_features, scenario
        )
    except ImportError:
        st.error("The River package is not installed. Add 'river' to requirements.txt "
                 "and run `python -m pip install river` locally.")
        return
    except Exception as exc:
        st.error(f"River forecasting failed: {exc}")
        return

    x, y, z = st.columns(3)
    x.metric("Latest observed filling", f"{float(training['reservoir'].iloc[-1]) * 100:.1f}%")
    y.metric(f"Forecast after {horizon} weeks", f"{float(predictions.iloc[-1]) * 100:.1f}%")
    z.metric("Forecast start", f"{predictions.index[0]:%d %b %Y}")

    tab_forecast, tab_animation, tab_check = st.tabs(
        ["Forecast", "Animated forecast", "Historical backtest"]
    )
    with tab_forecast:
        st.plotly_chart(_forecast_figure(training["reservoir"], predictions),
                        use_container_width=True)
        output = pd.DataFrame({
            "Week ending": predictions.index.date,
            "Forecast filling (%)": (predictions.to_numpy() * 100).round(2),
        })
        st.dataframe(output, hide_index=True, use_container_width=True)
        st.download_button(
            "Download forecast CSV", output.to_csv(index=False).encode("utf-8"),
            file_name=f"river_forecast_{region}_{origin:%Y%m%d}.csv", mime="text/csv",
        )
        st.caption("Predictions are clipped to the physically meaningful 0–100% range. "
                   "Clipping does not provide a confidence interval.")

    with tab_animation:
        st.plotly_chart(_animated_figure(training["reservoir"], predictions),
                        use_container_width=True)
        st.caption("Press Play to reveal the forecast one week at a time.")

    with tab_check:
        st.write("Compare River against a persistence baseline on historical weeks "
                 "not included in model training.")
        test_weeks = st.slider("Historical holdout (weeks)", 4, 12, 12)
        if n_available < history_weeks + test_weeks:
            st.info(f"Need {history_weeks + test_weeks} consecutive observations "
                    "to backtest with a {history_weeks}-week training window; "
                    f"currently have {n_available}.")
        else:
            train_backtest = aligned.iloc[-(history_weeks + test_weeks):-test_weeks]
            actual = aligned["reservoir"].iloc[-test_weeks:]
            try:
                predicted = _forecast_model(
                    train_backtest, test_weeks, p, d, q, m, sp, sd, sq,
                    selected_features, scenario
                )
                predicted.index = actual.index  # Both are consecutive W-SUN weeks.
                persistence = pd.Series(
                    float(train_backtest["reservoir"].iloc[-1]), index=actual.index
                )
                river_mae = float((predicted - actual).abs().mean() * 100)
                baseline_mae = float((persistence - actual).abs().mean() * 100)
                a, b = st.columns(2)
                a.metric("River MAE (percentage points)", f"{river_mae:.2f}")
                b.metric("Persistence MAE (percentage points)", f"{baseline_mae:.2f}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual.index, y=actual * 100,
                                         mode="lines+markers", name="Actual"))
                fig.add_trace(go.Scatter(x=predicted.index, y=predicted * 100,
                                         mode="lines+markers", name="River"))
                fig.add_trace(go.Scatter(x=persistence.index, y=persistence * 100,
                                         mode="lines", name="Persistence", line=dict(dash="dash")))
                fig.update_layout(height=380, yaxis_title="Reservoir filling (%)",
                                  xaxis_title="Held-out week", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("This is a single held-out window, not the full rolling-origin "
                           "evaluation in the assignment notebook. External-input scenarios "
                           "hold external values fixed from the training period.")
            except Exception as exc:
                st.error(f"Historical backtest failed: {exc}")

    st.info(
        "Forecasts are exploratory, not operational NVE predictions. "
        "The latest NVE reference week is not necessarily its publication date, "
        "and model accuracy can change with region and horizon."
    )
