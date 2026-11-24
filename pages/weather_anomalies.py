import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from ui.filters import get_selected_electricity_area
from services.weather import read_locations, download_era5_by_coordinate, weather_daily_from_hourly
from analysis.anomalies import plot_precipitation_spc, plot_temperature_lof

# Hourly ERA5 fields already downloaded by services/weather.py.
# Aggregation uses the same UTC calendar-day convention as the original page.

WEATHER_PROPERTIES = {
    "Temperature": ("temperature_2m", "°C", "mean"),
    "Precipitation": ("precipitation", "mm/day", "sum"),
    "Rain": ("rain", "mm/day", "sum"),
    "Snowfall": ("snowfall", "cm/day", "sum"),
    "Relative humidity": ("relative_humidity_2m", "%", "mean"),
    "Wind speed": ("wind_speed_10m", "km/h", "mean"),
}


def available_weather_properties(hourly):
    """Offer only fields present with at least one numeric observation."""
    return [
        label for label, (column, _, _) in WEATHER_PROPERTIES.items()
        if column in hourly.columns
        and pd.to_numeric(hourly[column], errors="coerce").notna().any()
    ]


def selected_daily_weather(hourly, weather_daily, label):
    """Produce a daily series for either anomaly algorithm."""
    column, unit, aggregation = WEATHER_PROPERTIES[label]

    # Reuse the app's existing daily aggregation when possible.
    if column == "temperature_2m":
        series = weather_daily["temperature_c"]
    elif column == "precipitation":
        series = weather_daily["precipitation_mm"]
    else:
        frame = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"], errors="coerce", utc=True),
            "value": pd.to_numeric(hourly[column], errors="coerce"),
        })
        frame = frame.dropna(subset=["time"]).set_index("time").sort_index()
        grouped = frame["value"].resample("D")
        series = (grouped.sum(min_count=1) if aggregation == "sum" else grouped.mean())

    return series.dropna().sort_index(), unit, aggregation


def relabel_anomaly_figure(fig, label, unit, aggregation):
    """Correct the plot labels used by the original fixed-property functions."""
    ax = fig.axes[0]
    ax.set_ylabel(f"{label} ({unit})")
    if ax.lines:
        descriptor = "daily total" if aggregation == "sum" else "daily mean"
        ax.lines[0].set_label(f"{label} ({descriptor})")
        ax.legend()
    fig.tight_layout()


def page_weather_anomalies():
    region = get_selected_electricity_area()
    st.title("Weather Anomalies")
    st.caption(f"Electricity area: {region} · Open-Meteo ERA5 · UTC")
    st.info(
        "Each download uses one sampled location within the selected area. "
        "It is not a spatial average of the entire electricity region."
    )

    with st.expander("Weather data and download controls", expanded=True):
        try:
            locations = read_locations()
        except Exception as exc:
            st.error(f"Could not load sampled locations: {exc}")
            return
        st.success("Sampled electricity locations loaded automatically")

        region_locations = (
            locations[locations["region"] == region].reset_index(drop=True)
            if locations is not None else pd.DataFrame()
        )
        if region_locations.empty:
            st.warning(
                "No sampled coordinates available for this area. "
                "Place electricity_sampled_locations.csv in the project "
                "or upload it above."
            )

        location_index = st.number_input(
            "Sampled location index", min_value=0,
            max_value=max(0, len(region_locations) - 1),
            value=0, step=1, disabled=region_locations.empty,
        )
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "Weather start date", value=pd.Timestamp("2024-01-01").date(),
                key="era5_start",
            )
        with col_end:
            end_date = st.date_input(
                "Weather end date", value=pd.Timestamp("2024-12-31").date(),
                key="era5_end",
            )

        if st.button(
            f"Download ERA5 for {region}",
            disabled=region_locations.empty or start_date > end_date,
            type="primary",
        ):
            point = region_locations.iloc[int(location_index)]
            try:
                with st.spinner("Downloading ERA5 hourly observations..."):
                    frame = download_era5_by_coordinate(
                        region, float(point["latitude"]),
                        float(point["longitude"]),
                        start_date.isoformat(), end_date.isoformat(),
                    )
                st.session_state["weather_dataset"] = frame
                st.success(
                    f"Loaded {len(frame):,} hourly records for {region} "
                    f"at {point['latitude']:.4f}, {point['longitude']:.4f}."
                )
            except Exception as exc:
                st.error(f"Weather download failed: {exc}")

        imported = st.file_uploader(
            "Alternatively, use an existing hourly ERA5 CSV",
            type=["csv"], key="hourly_weather_csv",
        )
        if imported is not None and st.button("Use uploaded weather CSV"):
            try:
                imported_data = pd.read_csv(imported)
                if not {"region", "time", "precipitation", "temperature_2m"}.issubset(
                    imported_data.columns
                ):
                    raise ValueError(
                        "CSV requires region, time, precipitation, temperature_2m."
                    )
                source_regions = set(
                    imported_data["region"].astype(str).str.strip()
                    .str.replace(" ", "", regex=False)
                )
                if source_regions != {region}:
                    raise ValueError(
                        f"CSV covers {source_regions}; selected area is {region}."
                    )
                st.session_state["weather_dataset"] = imported_data
                st.success(f"Loaded uploaded weather data for {region}.")
            except Exception as exc:
                st.error(f"CSV import failed: {exc}")

    hourly = st.session_state.get("weather_dataset")
    if hourly is None:
        st.info("Download or upload weather data to see the anomaly analyses.")
        return

    source_regions = set(
        hourly["region"].astype(str).str.strip()
        .str.replace(" ", "", regex=False)
    )
    if source_regions != {region}:
        st.info(
            f"The loaded weather data belongs to {', '.join(sorted(source_regions))}; "
            f"select that area or load data for {region}."
        )
        return

    try:
        weather_daily = weather_daily_from_hourly(hourly)
    except Exception as exc:
        st.error(f"Could not aggregate daily weather: {exc}")
        return

    st.caption(
        f"{len(weather_daily)} daily observations: "
        "precipitation summed per day; temperature averaged per day."
    )

    properties = available_weather_properties(hourly)
    if not properties:
        st.warning("The loaded dataset contains no usable weather properties.")
        return

    st.caption(
        "Choose an ERA5 property independently for each analysis. "
        "Rain, snowfall and precipitation use daily totals; "
        "temperature, humidity and wind speed use daily means."
    )

    tab_spc, tab_lof = st.tabs(["DCT / SPC", "LOF"])
    with tab_spc:
        spc_property = st.selectbox(
            "Weather property for DCT / SPC",
            options=properties,
            index=properties.index("Temperature") if "Temperature" in properties else 0,
            key="spc_weather_property",
        )
        cutoff = st.slider(
            "DCT cutoff (cycles/year)", 1, 24, 6,
        )
        n_std = st.slider(
            "SPC threshold (robust standard deviations)", 1.0, 6.0, 3.0, 0.5,
        )
        try:
            spc_series, spc_unit, spc_aggregation = selected_daily_weather(
                hourly, weather_daily, spc_property
            )
            # Reuse the original SPC algorithm, which expects this field name.
            spc_input = pd.DataFrame({"precipitation_mm": spc_series})
            fig, summary = plot_precipitation_spc(spc_input, cutoff, n_std)
            relabel_anomaly_figure(fig, spc_property, spc_unit, spc_aggregation)
            st.pyplot(fig)
            plt.close(fig)
            a, b, c = st.columns(3)
            a.metric("Days", summary["total_observations"])
            b.metric("SPC outliers", summary["outlier_count"])
            c.metric("Outlier share", f"{summary['outlier_percentage']:.2f}%")
            spc_outliers = summary["outliers"].rename(
                columns={"precipitation_mm": f"{spc_property} ({spc_unit})"}
            )
            st.dataframe(spc_outliers, use_container_width=True)
            st.caption(
                f"DCT/SPC analyses daily {spc_property.lower()}. "
                "Robust limits are estimated from DCT residuals and "
                "displayed against the observed values."
            )
        except Exception as exc:
            st.error(f"SPC analysis failed: {exc}")

    with tab_lof:
        lof_property = st.selectbox(
            "Weather property for LOF",
            options=properties,
            index=properties.index("Temperature") if "Temperature" in properties else 0,
            key="lof_weather_property",
        )
        lof_series, lof_unit, lof_aggregation = selected_daily_weather(
            hourly, weather_daily, lof_property
        )
        contamination = st.slider(
            "Expected anomaly proportion (%)",
            min_value=0.5, max_value=10.0, value=1.0, step=0.5,
        ) / 100.0
        n_neighbors = st.slider(
            "Number of LOF neighbours", min_value=2,
            max_value=min(100, max(2, len(lof_series) - 1)),
            value=min(20, max(2, len(lof_series) - 1)),
        )
        try:
            # Reuse the existing LOF algorithm (value and day-to-day change).
            lof_input = pd.DataFrame({"temperature_c": lof_series})
            fig, summary = plot_temperature_lof(
                lof_input, contamination, n_neighbors
            )
            relabel_anomaly_figure(fig, lof_property, lof_unit, lof_aggregation)
            st.pyplot(fig)
            plt.close(fig)
            a, b, c = st.columns(3)
            a.metric("Days", summary["total_observations"])
            b.metric("LOF anomalies", summary["outlier_count"])
            c.metric("Anomaly share", f"{summary['outlier_percentage']:.2f}%")
            lof_outliers = summary["outliers"].rename(
                columns={"temperature_c": f"{lof_property} ({lof_unit})"}
            )
            st.dataframe(lof_outliers, use_container_width=True)
            st.caption(
                f"LOF uses daily {lof_property.lower()} values and "
                "their day-to-day changes. Detected anomalies need not "
                "be measurement errors."
            )
        except Exception as exc:
            st.error(f"LOF analysis failed: {exc}")
