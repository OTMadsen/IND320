import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from ui.filters import get_selected_electricity_area
from services.weather import read_locations, download_era5_by_coordinate, weather_daily_from_hourly
from analysis.anomalies import plot_precipitation_spc, plot_temperature_lof


def page_weather_anomalies():
    region = get_selected_electricity_area()
    st.title("Weather Anomalies")
    st.caption(f"Electricity area: {region} · Open-Meteo ERA5 · UTC")
    st.info(
        "Each download uses one sampled location within the selected area. "
        "It is not a spatial average of the entire electricity region."
    )

    with st.expander("Weather data and download controls", expanded=True):
        point_file = st.file_uploader(
            "Optional sampled locations CSV (region, latitude, longitude)",
            type=["csv"], key="location_csv",
        )
        try:
            locations = read_locations(point_file)
        except Exception as exc:
            st.error(f"Could not read sampled locations: {exc}")
            locations = None

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

    tab_spc, tab_lof = st.tabs(["Precipitation · DCT / SPC",
                                "Temperature · LOF"])
    with tab_spc:
        cutoff = st.slider(
            "DCT cutoff (cycles/year)", 1, 24, 6,
        )
        n_std = st.slider(
            "SPC threshold (robust standard deviations)", 1.0, 6.0, 3.0, 0.5,
        )
        try:
            fig, summary = plot_precipitation_spc(weather_daily, cutoff, n_std)
            st.pyplot(fig)
            plt.close(fig)
            a, b, c = st.columns(3)
            a.metric("Days", summary["total_observations"])
            b.metric("SPC outliers", summary["outlier_count"])
            c.metric("Outlier share", f"{summary['outlier_percentage']:.2f}%")
            st.dataframe(summary["outliers"], use_container_width=True)
            st.caption(
                "SPC limits are calculated from robust statistics on DCT "
                "residuals, but displayed against original precipitation."
            )
        except Exception as exc:
            st.error(f"SPC analysis failed: {exc}")

    with tab_lof:
        contamination = st.slider(
            "Expected anomaly proportion (%)",
            min_value=0.5, max_value=10.0, value=1.0, step=0.5,
        ) / 100.0
        n_neighbors = st.slider(
            "Number of LOF neighbours", min_value=2,
            max_value=min(100, max(2, len(weather_daily) - 1)),
            value=min(20, max(2, len(weather_daily) - 1)),
        )
        try:
            fig, summary = plot_temperature_lof(
                weather_daily, contamination, n_neighbors
            )
            st.pyplot(fig)
            plt.close(fig)
            a, b, c = st.columns(3)
            a.metric("Days", summary["total_observations"])
            b.metric("LOF anomalies", summary["outlier_count"])
            c.metric("Anomaly share", f"{summary['outlier_percentage']:.2f}%")
            st.dataframe(summary["outliers"], use_container_width=True)
            st.caption(
                "LOF uses daily mean temperature and day-to-day changes. "
                "Detected anomalies need not be measurement errors."
            )
        except Exception as exc:
            st.error(f"LOF analysis failed: {exc}")
