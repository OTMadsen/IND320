"""Electricity area map with linked page selection, NVE data and ERA5 weather.



ERA5 is measured at ONE sampled location per price area, not the regional mean.
The Open-Meteo ERA5 data request is user-triggered, then can be bundled as a
CSV for immediate loading on future visits.
"""

import copy
import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

from services.weather import read_locations, download_era5_by_coordinate

#All the paths for the datasets we wish to use, all in the different assignments folders
ROOT = Path(__file__).resolve().parent.parent
GEOJSON_PATH = ROOT / "assignments" / "assignment4" / "electricity_areas.geojson"
RENDER_GEOJSON_PATH = ROOT / "assignments" /"assignment4" /"electricity_areas_render.geojson" #Due to some plotting issues I asked AI to error-search the original file, and rendered a new one
RESERVOIR_PATH = ROOT / "assignments" / "assignment1" / "reservoirs.csv"
WEATHER_2024_PATH = ROOT / "assignments" / "assignment4" / "NO3_ERA5_for_2024.csv"
WEEKLY_AREAS_PATH = ROOT / "assignments" / "assignment4" / "area_weather_weekly_2024.csv"

# Label display
WEATHER_FIELDS = {
    "Temperature": ("temperature_2m", "mean", "°C"),
    "Precipitation": ("precipitation", "sum", "mm/week"),
    "Rain": ("rain", "sum", "mm/week"),
    "Snowfall": ("snowfall", "sum", "cm/week"),
    "Wind speed": ("wind_speed_10m", "mean", "km/h"),
    "Relative humidity": ("relative_humidity_2m", "mean", "%"),
}


def normalise_region(value):
    """Normalize 'NO 3' and 'NO3' to 'NO3'."""
    return str(value).upper().strip().replace(" ", "")


@st.cache_data(show_spinner=False)
def load_electricity_areas():
    """Keep the original, verified GeoJSON for the true regional outlines."""
    if not GEOJSON_PATH.is_file():
        raise FileNotFoundError(f"Missing electricity-area GeoJSON: {GEOJSON_PATH}")

    areas = gpd.read_file(GEOJSON_PATH)
    if areas.crs is None:
        raise ValueError("GeoJSON is missing its coordinate reference system.")
    if "budomr" not in areas.columns:
        raise ValueError("GeoJSON must contain a 'budomr' column.")
    if len(areas) != 5:
        raise ValueError("Expected five electricity-area polygons.")

    # Do NOT simplify with a 750-degree tolerance in EPSG:4326.
    # Filled map rendering is handled by the separately triangulated file.
    return json.loads(areas.to_crs(epsg=4326).to_json())


@st.cache_data(show_spinner=False)
def load_render_triangles():
    "This is used for map colouring"

    if not RENDER_GEOJSON_PATH.is_file():
        raise FileNotFoundError(
            f"Missing rendering GeoJSON: {RENDER_GEOJSON_PATH}"
        )

    with RENDER_GEOJSON_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)

    
@st.cache_data(show_spinner=False)
def load_map_reservoirs():
    if not RESERVOIR_PATH.is_file():
        raise FileNotFoundError(f"Missing NVE reservoir data: {RESERVOIR_PATH}")
    df = pd.read_csv(RESERVOIR_PATH)
    if "omrType" not in df.columns and "Area_Type" in df.columns:
        df = df.rename(columns={
            "Area_Type": "omrType", "Area_Number": "omrnr",
            "Date": "dato_Id", "Fillings_Degree": "fyllingsgrad",
            "Filling_TWh": "fylling_TWh",
        })
    needed = {"omrType", "omrnr", "dato_Id", "fyllingsgrad"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Reservoir data is missing {sorted(needed - set(df.columns))}")
    # Do not restrict the source data to EL: NO and VASS need it too.
    df["omrType"] = df["omrType"].astype(str).str.upper().str.strip()
    df["dato_Id"] = pd.to_datetime(df["dato_Id"], errors="coerce")
    df["omrnr"] = pd.to_numeric(df["omrnr"], errors="coerce")
    df["fyllingsgrad"] = pd.to_numeric(df["fyllingsgrad"], errors="coerce")
    return df.dropna(subset=["dato_Id", "omrnr", "fyllingsgrad"])


def reservoir_colour(value):
    if value is None or pd.isna(value):
        return [145, 145, 145, 100]
    value = max(0.0, min(1.0, float(value)))
    low, high = (231, 145, 89), (43, 153, 120)
    return [round(a + value * (b - a)) for a, b in zip(low, high)] + [145]


def weather_colour(value, low, high):
    if value is None or pd.isna(value):
        return [145, 145, 145, 95]
    # One known area alone does not give a meaningful regional colour ranking.
    if low is None or high is None or high <= low:
        return [80, 145, 210, 155]
    t = max(0.0, min(1.0, (float(value) - low) / (high - low)))
    cold, warm = (57, 113, 190), (238, 145, 76)
    return [round(a + t * (b - a)) for a, b in zip(cold, warm)] + [155]


def hourly_to_weekly(hourly, region):
    """Prepare complete-ish Sunday-ending weeks from one station's hourly ERA5."""
    if "time" not in hourly.columns:
        raise ValueError("ERA5 CSV needs a 'time' column.")
    dt = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
    output = None
    for label, (column, kind, _unit) in WEATHER_FIELDS.items():
        if column not in hourly.columns:
            continue
        values = pd.to_numeric(hourly[column], errors="coerce")
        data = pd.DataFrame({"time": dt, "value": values})
        data = data.dropna(subset=["time"]).set_index("time").sort_index()
        if data.empty:
            continue
        grouped = data["value"].resample("W-SUN")
        weekly = (grouped.sum(min_count=1) if kind == "sum" else grouped.mean())
        weekly[grouped.count() < 120] = float("nan")  # Avoid partial weeks.
        if output is None:
            output = pd.DataFrame(index=weekly.index)
        output[column] = weekly
    if output is None:
        raise ValueError("ERA5 CSV contains no supported weather properties.")
    output.index = output.index.tz_localize(None)
    output.index.name = "week"
    output = output.reset_index()
    output["region"] = normalise_region(region)
    return output


@st.cache_data(show_spinner=False)
def load_bundled_weather():
    """Prefer committed five-area weekly CSV; fall back to existing NO3 hourly CSV."""
    if WEEKLY_AREAS_PATH.is_file():
        frame = pd.read_csv(WEEKLY_AREAS_PATH)
        if not {"week", "region"}.issubset(frame.columns):
            raise ValueError("area_weather_weekly_2024.csv needs 'week' and 'region'.")
        frame["week"] = pd.to_datetime(frame["week"], errors="coerce")
        frame["region"] = frame["region"].map(normalise_region)
        return frame.dropna(subset=["week"])
    if WEATHER_2024_PATH.is_file():
        hourly = pd.read_csv(WEATHER_2024_PATH)
        if "region" in hourly and not set(hourly["region"].dropna().map(normalise_region)) <= {"NO3"}:
            raise ValueError("The default hourly file must contain NO3 measurements only.")
        return hourly_to_weekly(hourly, "NO3")
    return pd.DataFrame(columns=["week", "region"])


def prepare_all_areas_from_era5(progress=None):
    points = read_locations()
    groups = []
    
    if WEATHER_2024_PATH.is_file():
        groups.append(hourly_to_weekly(pd.read_csv(WEATHER_2024_PATH), "NO3"))
    for num in (1, 2, 3, 4, 5):
        region = f"NO{num}"
        if region == "NO3" and WEATHER_2024_PATH.is_file():
            if progress is None:
                progress.progess(
                    num / 5,
                    text = f"Prepared {num} of 5 electricity areas"
                )
            continue
        
        subset = points.loc[points["region"].map(normalise_region) == region]
        if subset.empty:
            raise ValueError(f"No sampled location is available for {region}.")
        point = subset.iloc[0]
        hourly = download_era5_by_coordinate(
            region,
            float(point["latitude"]), float(point["longitude"]),
            "2024-01-01", "2024-12-31",
        )
        groups.append(hourly_to_weekly(hourly, region))
        if progress is not None:
            progress.progress(
                num / 5, 
                text=f"Prepared {num} of 5 electricity areas"
            )
    return pd.concat(groups, ignore_index=True).sort_values(["week", "region"])


def page_area_map():
    st.title("Interactive Electricity Area Map")
    if "selected_area_id" not in st.session_state:
        st.session_state.selected_area_id = None
    if "selected_area_type" not in st.session_state:
        st.session_state.selected_area_type = "EL"
    if "selected_electricity_area" not in st.session_state:
        st.session_state["selected_electricity_area"] = "NO3"

    try:
        reservoirs = load_map_reservoirs()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    dates = sorted(reservoirs["dato_Id"].dt.date.unique().tolist())
    if not dates:
        st.warning("No reservoir measurement dates are available.")
        return
    saved_date = st.session_state.get("selected_date")
    try:
        saved_date = pd.Timestamp(saved_date).date() if saved_date is not None else None
    except (ValueError, TypeError):
        saved_date = None
    if saved_date not in dates:
        saved_date = dates[-1]

    selected_date = st.select_slider(
        "Select reservoir measurement week", options=dates, value=saved_date,
        format_func=lambda value: value.strftime("%d %B %Y"),
        key="reservoir_week_slider",
    )
    st.session_state["selected_date"] = selected_date
    measurements = reservoirs.loc[reservoirs["dato_Id"].dt.date == selected_date]

    # The EL map is geographic; NO and VASS still use statistical tables.
    choices = {
        "Electricity price areas": "EL",
        "Norway total": "NO",
        "Watercourse regions": "VASS",
    }
    classification = st.selectbox("Select area classification", list(choices))
    selected_type = choices[classification]
    if st.session_state.selected_area_type != selected_type:
        st.session_state.selected_area_id = None
        st.session_state.selected_area_type = selected_type
    area_measurements = measurements.loc[measurements["omrType"] == selected_type].copy()
    
    if selected_type == "VASS":

        st.subheader(f"{classification} - {selected_date}")

        columns = [
            c for c in ("omrnr", "fyllingsgrad", "fylling_TWh")
            if c in area_measurements.columns
        ]

        st.dataframe(
            area_measurements[columns],
            hide_index=True
        )

        st.info(
            "VASS regions 1–3 require their own verified boundaries. "
            "The electricity-area polygons would not be appropriate here."
        )

        return

    
    # Norway total: show the nationwide reservoir measurement
    if selected_type == "NO":

        st.subheader(f"Norway total — {selected_date}")

        national = area_measurements.loc[
            area_measurements["omrnr"] == 0
        ]

        if national.empty:
            st.warning("No nationwide reservoir data available.")
            return

        filling = float(national["fyllingsgrad"].mean())

        st.metric(
            "Norway-wide reservoir filling",
            f"{filling * 100:.1f}%"
        )

        columns = [
            c for c in ("omrnr", "fyllingsgrad", "fylling_TWh")
            if c in national.columns
        ]

        st.dataframe(
            national[columns],
            hide_index=True
        )

        st.info(
            "This is the nationwide NVE reservoir aggregate. "
            "The nationwide map is omitted until suitable boundaries are available."
        )

        return


    metric = st.radio(
        "Colour electricity areas by",
        ["Reservoir filling", "ERA5 weather"], horizontal=True,
        key="area_map_colour_metric",
    )
    filling_by_area = area_measurements.groupby("omrnr")["fyllingsgrad"].mean().to_dict()
    weather_by_area = {}
    weather_min = weather_max = None
    weather_label = None
    unit = ""

    if metric == "ERA5 weather":
        try:
            weather = st.session_state.get("area_weather_weekly")
            if weather is None:
                weather = load_bundled_weather()
        except (ValueError, KeyError) as exc:
            st.error(f"Could not read the weather CSV: {exc}")
            return

        if st.button("Prepare 2024 ERA5 data for all five electricity areas"):

            progress = st.progress(
                0,
                text="Preparing ERA5 weather observations..."
            )

            try:
                with st.spinner(
                    "Downloading and processing weather data",
                    show_time=True
                ):
                    weather = prepare_all_areas_from_era5(
                        progress=progress
                    )

                    st.session_state["area_weather_weekly"] = weather

                st.success(
                    "ERA 5 weather prepared for all five electricity areas!"
                )

            except Exception as exc:
                st.error(f"Could not prepare ERA5 data: {exc}")

            finally:
                progress.empty()

        available = [
            label for label, (column, _method, _unit) in WEATHER_FIELDS.items()
            if column in weather.columns and pd.to_numeric(weather[column], errors="coerce").notna().any()
        ]
        if not available:
            st.info("No ERA5 values available. Add NO3_ERA5_for_2024.csv or "
                    "prepare weather data using the button above.")
            return
        property_label = st.selectbox("Weather property", available, key="map_weather_property")
        column, _method, unit = WEATHER_FIELDS[property_label]
        weather = weather.copy()
        weather["week"] = pd.to_datetime(weather["week"], errors="coerce")
        weather["region"] = weather["region"].map(normalise_region)
        weeks = sorted(weather.loc[
            pd.to_numeric(weather[column], errors="coerce").notna(), "week"
        ].dropna().dt.date.unique().tolist())
        if not weeks:
            st.warning("No complete ERA5 weeks available for this property.")
            return
        chosen_week = st.select_slider(
            "Select ERA5 weather week (2024)", options=weeks, value=weeks[-1],
            format_func=lambda value: value.strftime("%d %B %Y"),
            key="map_weather_week_slider",
        )
        weekly = weather.loc[weather["week"].dt.date == chosen_week].copy()
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce")
        weather_by_area = weekly.dropna(subset=[column]).groupby("region")[column].mean().to_dict()
        readings = list(weather_by_area.values())
        if len(readings) >= 2:
            weather_min, weather_max = min(readings), max(readings)
        weather_label = property_label
        st.caption(
            f"ERA5 for week ending {chosen_week}: {len(weather_by_area)} of 5 areas covered. "
            "One sampled location per area; NOT a regional weather average. "
            "Grey polygons have no data. This week selector is separate from the NVE week."
        )
        if len(readings) >= 2:
            st.caption(f"Blue = lower {property_label.lower()} ({weather_min:.1f} {unit}) · "
                       f"orange = higher ({weather_max:.1f} {unit})")
        if len(weather_by_area) < 5:
            st.warning(
                "Only some electricity areas have ERA5 data. "
                "Use the preparation button to collect NO1–NO5; do not interpret grey as zero."
            )
        if len(weather_by_area) == 5:
            st.download_button(
                "Download weekly ERA5 CSV for permanent default",
                data=weather.to_csv(index=False).encode("utf-8"),
                file_name="area_weather_weekly_2024.csv", mime="text/csv",
                help="Save the file to assignments/assignment4/ and commit it to GitHub "
                     "so the map loads automatically for new visitors.",
            )

    try:
        geojson = copy.deepcopy(load_render_triangles())
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Add electricity_areas_render.geojson to assignments/assignment4/.")
        return
    for feature in geojson["features"]:
        props = feature["properties"]
        region = normalise_region(props.get("budomr", ""))
        match = re.fullmatch(r"NO(\d+)", region)
        area_number = int(match.group(1)) if match else None
        if metric == "Reservoir filling":
            value = filling_by_area.get(area_number)
            props["fill_color"] = reservoir_colour(value)
            props["metric_label"] = (f"{value * 100:.1f}%" if value is not None else "No data")
            props["metric_title"] = "Reservoir filling"
        else:
            value = weather_by_area.get(region)
            props["fill_color"] = weather_colour(value, weather_min, weather_max)
            props["metric_label"] = (f"{value:.1f} {unit}" if value is not None else "No data")
            props["metric_title"] = weather_label


  
    base_layer = pdk.Layer(
            "GeoJsonLayer",
                id="electricity_areas",
                data=geojson,
                pickable=True,

    # Prevent individual triangles from being highlighted
                auto_highlight=False,
                highlight_color=[0, 0, 0, 0],

                stroked=False,
                filled=True,
                extruded=False,
                opacity=1,
                get_fill_color="properties.fill_color",
    )

    # Draw the real region boundaries over the triangles.
    outline_layer = pdk.Layer(
        "GeoJsonLayer",
        id="electricity_outlines",
        data=load_electricity_areas(),
        pickable=False,
        stroked=True,
        filled=False,
        extruded=False,
        get_line_color=[30, 60, 100, 200],
        get_line_width=1,
        line_width_units="pixels",
        line_width_min_pixels=1,
    )
    layers = [base_layer, outline_layer]

    # Highlight only the selected region's actual outline, not every triangle.
    selected_area = st.session_state.selected_area_id
    if selected_area is not None:
        selected_geo = [
            feature for feature in load_electricity_areas()["features"]
            if normalise_region(feature["properties"].get("budomr", ""))
            == normalise_region(selected_area)
        ]
        if selected_geo:
            highlight_layer = pdk.Layer(
                "GeoJsonLayer",
                id="selected_area_highlight",
                data={"type": "FeatureCollection", "features": selected_geo},
                pickable=False,
                stroked=True,
                filled=False,
                extruded=False,
                get_line_color=[255, 165, 0, 150],
                get_line_width=1,
                line_width_units="pixels",
                line_width_min_pixels=1,
            )
            layers.append(highlight_layer)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=64.5, longitude=12.5, zoom=3.5, pitch=0
        ),
        map_style=None,
        tooltip={
            "html": "<b>Electricity area:</b> {budomr}"
                    "<br/><b>{metric_title}:</b> {metric_label}",
            "style": {"color": "white"},
        },
    )

    event = st.pydeck_chart(
        deck,
        height=600,
        key="electricity_area_map",
        on_select="rerun",
        selection_mode="single-object",
    )

    # Selection indices now refer to the triangle features, not the 5 outlines.
    indices = event.selection.indices.get("electricity_areas", [])
    if indices:
        feature_index = indices[0]
        if 0 <= feature_index < len(geojson["features"]):
            clicked_area = str(
                geojson["features"][feature_index]["properties"]["budomr"]
            ).strip()
            code = normalise_region(clicked_area)
            if (
                code != st.session_state.get("selected_electricity_area")
                or clicked_area != st.session_state.selected_area_id
            ):
                st.session_state.selected_area_id = clicked_area
                st.session_state.selected_area_type = "EL"
                # Shared state key read by other Streamlit analysis pages.
                # The sidebar widget in app.py must use its separate key.
                st.session_state["selected_electricity_area"] = code
                st.rerun()

    st.caption(
        "Shared electricity area for other analysis pages: "
        f"**{st.session_state['selected_electricity_area']}**"
    )
    if st.session_state.selected_area_id is not None:
        st.success(
            f"Selected electricity area: {st.session_state.selected_area_id}"
        )
    else:
        st.info("Click an electricity area to select it for the other pages.")

    if metric == "Reservoir filling":
        st.caption(
            "Orange = lower reservoir filling · teal = higher filling · grey = missing data"
        )
