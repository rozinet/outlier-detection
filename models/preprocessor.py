"""Data preprocessing utilities."""

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def preprocess_device_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Multi-resolution preprocessing.

    Three signal levels per channel:
      {col}_raw       - original data (for sensor malfunction detection)
      {col}_denoised  - median filter only (for intrusion + rapid change detectors)
      {col}           - trend: EWMA or causal rolling mean on denoised (for condensation + drying)

    Args:
        df: Input DataFrame with sensor data
        config: Configuration dictionary

    Returns:
        Preprocessed DataFrame
    """
    column_names = ["temp", "hum_ambient", "hum_cavity", "moisture"]

    median_k = config.get("median_filter_window", 7)
    smooth_k = config.get("smoothing_window", 6)
    use_ewma = config.get("use_ewma_trend", True)
    ewma_halflife = config.get("ewma_halflife_hours", 6) * 12  # hours -> samples

    processed = df.copy()

    for col in column_names:
        if col not in processed.columns:
            continue
        raw = processed[col].values.copy()
        processed[f"{col}_raw"] = raw

        mask = np.isnan(raw)
        if mask.all():
            continue
        filled = pd.Series(raw).ffill().bfill().values
        filtered = median_filter(filled, size=median_k)
        filtered[mask] = np.nan

        # Level 2 - denoised (median filter only, preserves edges and peaks)
        processed[f"{col}_denoised"] = filtered

        # Level 3 - trend (EWMA or causal rolling mean on denoised)
        denoised_series = pd.Series(filtered, index=df.index)
        if use_ewma:
            trend = denoised_series.ewm(halflife=ewma_halflife, min_periods=1).mean()
        else:
            trend = denoised_series.rolling(smooth_k, min_periods=1, center=False).mean()
        processed[col] = trend

    return processed


def compute_seasonal_baseline(series: pd.Series, window_days: int = 30) -> pd.DataFrame:
    """
    Compute seasonal baseline for a sensor channel.

    Args:
        series: Time series data
        window_days: Window size in days

    Returns:
        DataFrame with columns: 'baseline' (rolling median), 'std' (rolling std),
        'deviation' (how far current value is above baseline in std units).
    """
    samples_per_day = 288  # 5-min intervals
    window = window_days * samples_per_day

    baseline = series.rolling(window, min_periods=window // 4, center=False).median()
    rolling_std = series.rolling(window, min_periods=window // 4, center=False).std()
    rolling_std = rolling_std.clip(lower=1.0)  # avoid division by zero

    deviation = (series - baseline) / rolling_std

    return pd.DataFrame({
        "baseline": baseline,
        "std": rolling_std,
        "deviation": deviation,
    }, index=series.index)


def compute_fleet_seasonal_profile(
    devices: dict[str, pd.DataFrame],
    column: str = "hum_cavity",
) -> pd.DataFrame | None:
    """
    Compute fleet-wide seasonal profile: median and std by month-of-year.

    Args:
        devices: Dictionary of device DataFrames
        column: Column name to analyze

    Returns:
        DataFrame indexed by month (1-12) with columns: 'median', 'q25', 'q75'
        or None if no data
    """
    all_monthly = []
    for did, df in devices.items():
        if column not in df.columns:
            continue
        monthly = df[column].dropna().resample("MS").mean()
        if len(monthly) > 0:
            monthly_df = pd.DataFrame({"value": monthly})
            monthly_df["month"] = monthly_df.index.month
            all_monthly.append(monthly_df)

    if not all_monthly:
        return None

    combined = pd.concat(all_monthly, ignore_index=True)
    profile = combined.groupby("month")["value"].agg([
        "median",
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ])
    profile.columns = ["median", "q25", "q75"]
    return profile
