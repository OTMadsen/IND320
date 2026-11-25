import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.fft import dct, idct
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


def plot_precipitation_spc(weather, cutoff_cycles_per_year=6, n_std=3):
    series = weather["precipitation_mm"].dropna().sort_index()
    if len(series) < 30:
        raise ValueError("At least 30 daily measurements are required.")
    if cutoff_cycles_per_year <= 0 or n_std <= 0:
        raise ValueError("The cutoff and standard-deviation multiplier must be positive.")

    values = series.to_numpy(dtype=float)
    n = len(values)
    coefficients = dct(values, norm="ortho")
    frequencies = np.arange(n) / (2 * n)
    low_frequency = coefficients.copy()
    low_frequency[frequencies >= cutoff_cycles_per_year / 365.25] = 0
    baseline = idct(low_frequency, norm="ortho")
    residual = values - baseline
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0:
        raise ValueError("MAD is zero: robust SPC limits cannot be estimated.")

    lower = baseline + median - n_std * robust_sigma
    upper = baseline + median + n_std * robust_sigma
    mask = (values < lower) | (values > upper)

    # Interactive Plotly chart; the DCT/SPC calculations above are unchanged.
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=values, mode="lines",
        name="Daily precipitation",
        line=dict(color="#3978BE", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=series.index, y=upper, mode="lines",
        name="Upper SPC limit",
        line=dict(color="#F0A32F", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=series.index, y=lower, mode="lines",
        name="Lower SPC limit",
        line=dict(color="#9966CC", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=series.index[mask], y=values[mask], mode="markers",
        name="SPC outliers",
        marker=dict(color="#E35D60", size=9),
    ))
    fig.update_layout(
        title="Daily precipitation — DCT and SPC anomaly detection",
        xaxis_title="Date",
        yaxis_title="Daily precipitation (mm)",
        hovermode="x unified", height=480,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=20, r=20, t=90, b=40),
    )

    outliers = pd.DataFrame(
        {"precipitation_mm": values[mask],
         "lower_limit": lower[mask], "upper_limit": upper[mask],
         "direction": np.where(values[mask] > upper[mask], "High", "Low")},
        index=series.index[mask],
    )
    return fig, {
        "total_observations": n,
        "outlier_count": int(mask.sum()),
        "outlier_percentage": 100 * mask.mean(),
        "robust_sigma": robust_sigma,
        "outliers": outliers,
    }


def plot_temperature_lof(weather, contamination=0.01, n_neighbors=20):
    if not 0 < contamination <= 0.5:
        raise ValueError("Contamination must be greater than 0 and at most 0.5.")
    series = weather["temperature_c"].dropna().sort_index()
    if len(series) < 3:
        raise ValueError("At least three daily measurements are required.")

    features = pd.DataFrame({
        "temperature": series,
        "daily_change": series.diff().fillna(0),
    })
    scaled = RobustScaler().fit_transform(features)
    lof = LocalOutlierFactor(
        n_neighbors=min(n_neighbors, len(series) - 1),
        contamination=contamination,
    )
    labels = lof.fit_predict(scaled)
    mask = labels == -1

    # Interactive Plotly chart; the LOF calculations above are unchanged.
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.to_numpy(), mode="lines",
        name="Daily mean temperature",
        line=dict(color="#3978BE", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=series.index[mask], y=series.to_numpy()[mask], mode="markers",
        name="LOF anomalies",
        marker=dict(color="#E35D60", size=10),
    ))
    fig.update_layout(
        title="Daily mean temperature — LOF anomaly detection",
        xaxis_title="Date",
        yaxis_title="Daily mean temperature (°C)",
        hovermode="x unified", height=480,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=20, r=20, t=90, b=40),
    )

    outliers = pd.DataFrame(
        {"temperature_c": series.values[mask],
         "lof_score": -lof.negative_outlier_factor_[mask]},
        index=series.index[mask],
    )
    return fig, {
        "total_observations": len(series),
        "outlier_count": int(mask.sum()),
        "outlier_percentage": 100 * mask.mean(),
        "outliers": outliers,
    }
