import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, name, data in zip(
        axes,
        ["Observed reservoir filling", "Trend", "Seasonal", "Residual"],
        [series, result.trend, result.seasonal, result.resid],
    ):
        ax.plot(series.index, np.asarray(data), linewidth=1.0)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Filling degree")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
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
    date_numbers = mdates.date2num(dates.to_pydatetime())
    power_db = 10 * np.log10(np.maximum(power, 1e-12))

    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(
        date_numbers, frequencies * 52.18, power_db,
        shading="auto", cmap="viridis"
    )
    fig.colorbar(mesh, ax=ax, label="Spectral power (dB)")
    ax.axhline(1.0, color="white", linestyle="--", label="Annual cycle")
    ax.set_ylim(0, 6)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Frequency (cycles/year)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
