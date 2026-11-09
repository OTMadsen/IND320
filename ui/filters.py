import streamlit as st


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


def get_selected_electricity_area():
    return st.session_state.get("selected_electricity_area", "NO3")
