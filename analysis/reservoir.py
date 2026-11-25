import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import spectrogram
from statsmodels.tsa.seasonal import STL


def plot_reservoir_stl(series, seasonal_smoother=13,
                       trend_smoother=105, robust=True):
    if seasonal_smoother < 3 or seasonal_smoother % 2 == 0:
        raise ValueError("Seasonal smoother must be odd and at least 3.")
    if trend_smoother <= 52 or trend_smoother % 2 == 0:
        raise ValueError("Trend smoother must be odd and greater than 52.")

    result = STL(
        series, period=52, seasonal=seasonal_smoother,
        trend=trend_smoother, robust=robust
    ).fit()
    # Display the same STL components as four synchronized, interactive plots.
    labels = ["Observed reservoir filling", "Trend", "Seasonal", "Residual"]
    components = [series, result.trend, result.seasonal, result.resid]
    colours = ["#3978BE", "#2B9978", "#F0A32F", "#E35D60"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        subplot_titles=labels,
    )
    for row, (label, data, colour) in enumerate(
        zip(labels, components, colours), start=1
    ):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=np.asarray(data, dtype=float),
                mode="lines",
                name=label,
                line=dict(color=colour, width=1.7),
                hovertemplate="%{x|%Y-%m-%d}<br>Value: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text="Filling degree" if row <= 2 else "Deviation",
            row=row,
            col=1,
        )

    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_layout(
        title="Reservoir filling — STL decomposition",
        height=900,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=30, r=20, t=90, b=45),
    )
    return fig


def plot_reservoir_spectrogram(series, window_length=104,
                               window_overlap=52):
    series = series.sort_index()
    if series.isna().any():
        raise ValueError("The series contains missing observations.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series must have a DatetimeIndex.")
    if window_length < 4 or window_length > len(series):
        raise ValueError("Window length must be 4 or more and fit the data.")
    if not 0 <= window_overlap < window_length:
        raise ValueError("Overlap must be smaller than the window length.")
    if not (series.index.to_series().diff().dropna()
            == pd.Timedelta(days=7)).all():
        raise ValueError("The data must be sampled every 7 days.")

    frequencies, times, power = spectrogram(
        series.to_numpy(dtype=float), fs=1.0, window="hann",
        nperseg=window_length, noverlap=window_overlap,
        detrend="linear", scaling="spectrum"
    )
    dates = series.index[0] + pd.to_timedelta(times * 7, unit="D")
    power_db = 10 * np.log10(np.maximum(power, 1e-12))

    # Interactive heatmap: same spectrogram power values as the former Matplotlib plot.
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=dates,
                y=frequencies * 52.18,
                z=power_db,
                colorscale="Viridis",
                colorbar=dict(title="Power (dB)"),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Frequency: %{y:.2f} cycles/year<br>"
                    "Power: %{z:.1f} dB<extra></extra>"
                ),
            )
        ]
    )
    fig.add_hline(
        y=1.0,
        line_color="white",
        line_dash="dash",
        line_width=2,
        annotation_text="Annual cycle (1/year)",
        annotation_position="top right",
        annotation_font_color="white",
    )
    fig.update_layout(
        title="Reservoir filling — spectrogram",
        xaxis_title="Date",
        yaxis_title="Frequency (cycles/year)",
        height=560,
        margin=dict(l=30, r=20, t=70, b=50),
    )
    fig.update_yaxes(range=[0, 6])
    return fig
