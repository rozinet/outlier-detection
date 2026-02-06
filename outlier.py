"""
Building Problem Detection for Senzomatic Sensor Data (v6).

Detects specific building pathologies from HT sensor readings:
  1. MOISTURE INTRUSION  - water entering construction (moisture_resistance drop AND cavity humidity spike)
  2. CONDENSATION RISK   - cavity humidity sustained above thresholds (seasonal-aware)
  3. DRYING FAILURE      - post-construction moisture not decreasing as expected
  4. SENSOR MALFUNCTION  - flatlines (saturation-aware), impossible jumps, out-of-range values
  5. RAPID MOISTURE CHANGE - derivative-based detector for acute moisture events

v6 improvements over v5:
  G1. Multi-resolution signals: raw/denoised/trend per channel; acute detectors use denoised
  G2. Time-based deltas: shift(freq=...) instead of diff(N) for correct gap handling
  G3. Condensation hysteresis: enter at threshold, exit at threshold-2% to reduce toggling
  G4. Fleet seasonal reference in detection: catches always-wet-but-stable cavities
  G5. CUSUM change-point detection for moisture intrusion and rapid change (catches slow leaks)
  G6. Exponential drying curve fitting: M(t)=a+b*exp(-t/tau) instead of linear slope
  G7. Vapor pressure / absolute humidity as supplementary condensation signal
  G8. Residual-based sensor fault detection: Hampel filter, drift, stuck-at-mid-value
  G9. Robust installation outliers: MAD z-scores on rich feature vectors (p95, %time, streaks)
  G10. Bug fixes: chronic WARNING uses combined flag, causal baselines, safe 7d/14d delta skip

v5 improvements over v4:
  F1. Increased WARNING condensation merge gap to 21 days (from 7 days)
  F2. Added recurring condensation pattern detection (>6 WARNING episodes in 12 months -> single recurring episode)
  F3. Lowered chronic merge threshold to 40% for WARNING (from 50%), kept 50% for DANGER/CRITICAL
  F4. Fixed moisture intrusion AND logic: requires drop >1% AND rise >1%, filters 0 or negative changes
  F5. Fixed rapid moisture change description: now reports actual start/end values and peak drop separately
  F6. Visual: recurring condensation shown as semi-transparent band with tick marks instead of fragmented bars

v4 improvements over v3:
  A1. Seasonal baseline subtraction for condensation (flags deviation from device norm)
  A2. Chronic condensation merged into single episode when >50% time above threshold
  A3. Saturation vs flatline cross-validation (100% humidity = sensor saturation, not malfunction)
  A4. Moisture intrusion requires AND logic (moisture drop AND cavity rise)
  A5. Rate-of-change detector for rapid moisture resistance transitions
  B6. Dual-axis overlay panel (cavity humidity + moisture resistance)
  B7. Chronic vs acute visual separation on timeline (hatching for chronic)
  B8. Seasonal reference band on cavity humidity panel (fleet median +/- 1std)
  B9. Color-coded time series lines by threshold exceedance
  B10. Per-installation comparison panel (box plots)
  C11. Composite health score per device (0-100)
  C12. Inter-device comparison within installation (outlier flagging)

Sensors per device (HT02):
  - temperature_ambient_celsius   (ambient temperature at sensor)
  - rel_humidity_ambient_pct      (ambient relative humidity)
  - rel_humidity_cavity_pct       (relative humidity inside wall cavity)
  - moisture_resistance_pct       (material moisture resistance; higher = drier)
"""

import json
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SENSOR_TYPES = [
    "temperature_ambient_celsius",
    "rel_humidity_ambient_pct",
    "rel_humidity_cavity_pct",
    "moisture_resistance_pct",
]
COLUMN_NAMES = ["temp", "hum_ambient", "hum_cavity", "moisture"]

CONFIG = {
    "resample_interval": "5min",

    # --- Preprocessing (G1: multi-resolution) ---
    "median_filter_window": 7,              # median filter kernel size (samples); removes short spikes
    "smoothing_window": 6,                  # rolling mean window (samples, ~30min); smooths remaining noise
    "ewma_halflife_hours": 6,              # G1: EWMA half-life for trend signal (hours)
    "use_ewma_trend": True,                # G1: True = EWMA for trend, False = causal rolling mean
    "min_data_hours": 48,                   # skip devices with less data than this

    # --- Moisture intrusion (F4: stricter AND logic) ---
    "moisture_drop_threshold_24h": 3.0,     # pct-points drop in 24h
    "moisture_drop_threshold_7d": 5.0,      # pct-points drop in 7 days
    "cavity_rise_threshold_24h": 8.0,       # pct-points rise in 24h
    "moisture_intrusion_min_hours": 24,     # at least 1 day
    "moisture_drop_min": 1.0,               # F4: minimum drop to consider (filters 0 or noise)
    "cavity_rise_min": 1.0,                 # F4: minimum rise to consider (filters 0 or noise)

    # --- Condensation risk (A1: seasonal-aware, A2+F1-F3: chronic/recurring merge) ---
    "condensation_warning_pct": 80.0,       # cavity hum % -> mold risk
    "condensation_danger_pct": 90.0,        # cavity hum % -> condensation likely
    "condensation_critical_pct": 95.0,      # cavity hum % -> active condensation
    "condensation_min_hours": 48,           # at least 2 days
    "condensation_chronic_pct_warning": 40.0,   # F3: % time above threshold for WARNING chronic (lowered from 50%)
    "condensation_chronic_pct_severe": 50.0,    # F3: % time for DANGER/CRITICAL chronic
    "condensation_chronic_min_months": 6,   # minimum months for chronic assessment
    "condensation_recurring_min_episodes": 6,   # F2: # episodes in 12mo to trigger recurring pattern
    "condensation_warning_merge_gap_days": 21,  # F1: merge gap for WARNING (increased from 7)
    "seasonal_baseline_window_days": 30,    # A1: rolling window for seasonal baseline
    "condensation_hysteresis_band": 2.0,   # G3: pct-points below entry threshold for exit
    "condensation_use_abs_humidity": True,  # G7: supplement RH with absolute humidity
    "abs_humidity_warning_gkg": 14.0,      # G7: g/kg absolute humidity WARNING threshold
    "fleet_seasonal_offset_pct": 8.0,      # G4: above fleet monthly median + this = outlier

    # --- Rapid moisture change (A5) ---
    "rapid_moisture_drop_3d": 4.0,          # pct-points drop in 3 days (derivative-based)
    "rapid_moisture_drop_14d": 8.0,         # pct-points drop in 14 days
    "rapid_change_min_hours": 12,           # minimum episode duration

    # --- Rapid moisture change: CUSUM (G5) ---
    "cusum_threshold": 5.0,                # G5: CUSUM alarm threshold
    "cusum_drift": 0.5,                    # G5: CUSUM allowance/drift parameter
    "cusum_confirmation_window_hours": 72, # G5: window after change point for confirmation

    # --- Drying failure (G6: exponential curve) ---
    "drying_eval_window_weeks": 4,
    "drying_plateau_tolerance": 0.5,
    "drying_reversal_threshold": 1.0,
    "drying_tau_warning_days": 365,        # G6: exp time constant > this = slow drying (WARNING)
    "drying_tau_danger_days": 730,         # G6: exp time constant > this = very slow (DANGER)
    "drying_plateau_pct": 80.0,            # G6: exp plateau above this = failure
    "drying_initial_wet_threshold": 85.0,  # G6: cavity RH above this = initial wet period
    "drying_exp_fit_min_days": 60,         # G6: minimum data span for exp fit

    # --- Sensor malfunction (A3: saturation-aware) ---
    "flatline_window_hours": 24,
    "jump_threshold_temp": 10.0,
    "jump_threshold_humidity": 25.0,
    "jump_threshold_moisture": 20.0,
    "jump_min_count": 3,
    "temp_range": (-40.0, 60.0),
    "humidity_range": (0.0, 100.0),
    "moisture_range": (0.0, 100.0),
    "saturation_values": {                  # A3: values that indicate sensor saturation, not malfunction
        "hum_ambient": [0.0, 100.0],
        "hum_cavity": [0.0, 100.0],
        "moisture": [0.0, 100.0],
    },

    # --- Sensor malfunction: v6 additions (G8) ---
    "hampel_window": 25,                   # G8: Hampel filter window size (samples)
    "hampel_threshold": 3.0,              # G8: Hampel filter MAD multiplier
    "sensor_residual_window_days": 14,     # G8: window for drift detection model
    "sensor_drift_threshold_std": 3.0,     # G8: residual mean shift > N std = drift

    # --- Installation outliers (G9) ---
    "outlier_mad_zscore_threshold": 3.0,   # G9: MAD z-score > this = outlier
    "outlier_min_devices": 3,              # G9: minimum devices per installation

    # --- Episode merging ---
    "episode_merge_gap_hours": 6,

    # --- Health score (C11) ---
    "health_score_weights": {
        "condensation_risk": {"warning": 5, "danger": 15, "critical": 30},
        "moisture_intrusion": {"warning": 10, "danger": 25, "critical": 40},
        "drying_failure": {"warning": 8, "danger": 20, "critical": 35},
        "sensor_malfunction": {"warning": 3, "danger": 10, "critical": 20},
        "rapid_moisture_change": {"warning": 8, "danger": 20, "critical": 35},
    },
}

# Severity levels
OK = "ok"
WARNING = "warning"
DANGER = "danger"
CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """A detected building problem."""
    problem_type: str       # moisture_intrusion | condensation | drying_failure | sensor_malfunction | rapid_moisture_change
    severity: str           # ok | warning | danger | critical
    device_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    description: str
    details: dict = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600

    @property
    def is_chronic(self) -> bool:
        return self.duration_hours > 90 * 24  # >90 days

    def __str__(self):
        dur = f"{self.duration_hours:.0f}h"
        tag = " [CHRONIC]" if self.is_chronic else ""
        return f"[{self.severity.upper():8s}] {self.problem_type:22s} | {self.device_id[:12]} | {self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d} ({dur}){tag} | {self.description}"


# ---------------------------------------------------------------------------
# Step 1: Data Loading
# ---------------------------------------------------------------------------

def load_single_channel(filepath: str) -> pd.Series | None:
    """Load a single sensor channel from a .json_line file."""
    try:
        if os.path.getsize(filepath) == 0:
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.strip()
        if not content:
            return None
        first_line = content.split("\n", 1)[0]
        data = json.loads(first_line)
        timestamps = pd.to_datetime(data["timestamps"], unit="ms")
        values = np.array(data["values"], dtype=np.float64)
        series = pd.Series(values, index=timestamps, name=data["metric"]["__name__"])
        series = series[~series.index.duplicated(keep="first")].sort_index()
        return series
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        print(f"  Warning: failed to load {os.path.basename(filepath)}: {e}")
        return None


def load_device_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all devices, align channels on 5-min grid."""
    device_files: dict[str, dict[str, str]] = {}
    installation_map: dict[str, str] = {}

    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".json_line"):
                continue
            device_id = fname[:36]
            sensor_type = fname[37:].replace(".json_line", "")
            if sensor_type not in SENSOR_TYPES:
                continue
            device_files.setdefault(device_id, {})[sensor_type] = os.path.join(root, fname)
            if device_id not in installation_map:
                try:
                    with open(os.path.join(root, fname), "r") as f:
                        line = f.readline().strip()
                        if line:
                            meta = json.loads(line)["metric"]
                            installation_map[device_id] = meta.get("installation_id", "unknown")
                except Exception:
                    pass

    print(f"Found {len(device_files)} devices")
    devices = {}
    for device_id, files_map in device_files.items():
        channels = {}
        for sensor_type, col_name in zip(SENSOR_TYPES, COLUMN_NAMES):
            if sensor_type in files_map:
                series = load_single_channel(files_map[sensor_type])
                if series is not None and len(series) > 0:
                    channels[col_name] = series
        if len(channels) < 2:
            continue
        df = pd.DataFrame(channels)
        df = df.resample(CONFIG["resample_interval"]).mean()
        df = df.dropna(how="all")
        df = df.ffill(limit=6)
        df.attrs["device_id"] = device_id
        df.attrs["installation_id"] = installation_map.get(device_id, "unknown")
        devices[device_id] = df

    print(f"Loaded {len(devices)} devices")
    return devices


# ---------------------------------------------------------------------------
# Step 2: Preprocessing - noise removal
# ---------------------------------------------------------------------------

def preprocess_device_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    G1: Multi-resolution preprocessing.

    Three signal levels per channel:
      {col}_raw       - original data (for sensor malfunction detection)
      {col}_denoised  - median filter only (for intrusion + rapid change detectors)
      {col}           - trend: EWMA or causal rolling mean on denoised (for condensation + drying)
    """
    from scipy.ndimage import median_filter

    median_k = CONFIG["median_filter_window"]
    smooth_k = CONFIG["smoothing_window"]
    use_ewma = CONFIG.get("use_ewma_trend", True)
    ewma_halflife = CONFIG.get("ewma_halflife_hours", 6) * 12  # hours -> samples

    processed = df.copy()

    for col in COLUMN_NAMES:
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

        # G1: Level 2 - denoised (median filter only, preserves edges and peaks)
        processed[f"{col}_denoised"] = filtered

        # G1: Level 3 - trend (EWMA or causal rolling mean on denoised)
        denoised_series = pd.Series(filtered, index=df.index)
        if use_ewma:
            trend = denoised_series.ewm(halflife=ewma_halflife, min_periods=1).mean()
        else:
            trend = denoised_series.rolling(smooth_k, min_periods=1, center=False).mean()
        processed[col] = trend

    return processed


# ---------------------------------------------------------------------------
# A1: Seasonal baseline computation
# ---------------------------------------------------------------------------

def compute_seasonal_baseline(series: pd.Series, window_days: int = 30) -> pd.DataFrame:
    """
    Compute seasonal baseline for a sensor channel.

    Returns DataFrame with columns: 'baseline' (rolling median), 'std' (rolling std),
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
    Compute fleet-wide seasonal profile (B8): median and std by month-of-year.

    Returns DataFrame indexed by month (1-12) with columns: 'median', 'q25', 'q75'.
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
    profile = combined.groupby("month")["value"].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    profile.columns = ["median", "q25", "q75"]
    return profile


# ---------------------------------------------------------------------------
# Detector 1: MOISTURE INTRUSION (A4: AND logic)
# ---------------------------------------------------------------------------

def detect_moisture_intrusion(df: pd.DataFrame, device_id: str) -> list[Problem]:
    """
    Detect water entering construction.

    G1: Uses denoised signal (preserves peaks better than trend).
    G2: Time-based deltas instead of fixed-step diff.
    G5: CUSUM change-point detection for slow leaks.
    F4: Stricter AND logic requires BOTH moisture drop >1% AND cavity rise >1%.
    """
    problems = []

    has_moisture = "moisture" in df.columns
    has_cavity = "hum_cavity" in df.columns

    if not has_moisture and not has_cavity:
        return problems

    # G1: Use denoised signal for acute detection (preserves edges)
    moisture_col = "moisture_denoised" if "moisture_denoised" in df.columns else "moisture"
    cavity_col = "hum_cavity_denoised" if "hum_cavity_denoised" in df.columns else "hum_cavity"

    data_span = df.index[-1] - df.index[0]

    # G2: Time-based deltas
    if has_moisture:
        moisture_delta_24h = _time_delta(df[moisture_col], "24h")
        if data_span >= pd.Timedelta("7D"):
            moisture_delta_7d = _time_delta(df[moisture_col], "7D")
        else:
            moisture_delta_7d = pd.Series(np.nan, index=df.index)  # G10: skip 7d if insufficient span
    else:
        moisture_delta_24h = pd.Series(0, index=df.index)
        moisture_delta_7d = pd.Series(np.nan, index=df.index)

    if has_cavity:
        cavity_delta_24h = _time_delta(df[cavity_col], "24h")
    else:
        cavity_delta_24h = pd.Series(0, index=df.index)

    # F4: Require meaningful changes (filter 0 or noise)
    moisture_dropping = moisture_delta_24h < -CONFIG["moisture_drop_threshold_24h"]
    moisture_dropping_7d = moisture_delta_7d < -CONFIG["moisture_drop_threshold_7d"]
    moisture_dropping_7d = moisture_dropping_7d.fillna(False)
    cavity_rising = cavity_delta_24h > CONFIG["cavity_rise_threshold_24h"]

    # F4: Additional filter for minimum change magnitude
    moisture_meaningful_drop = moisture_delta_24h < -CONFIG["moisture_drop_min"]
    cavity_meaningful_rise = cavity_delta_24h > CONFIG["cavity_rise_min"]

    # F4: AND logic when both channels available
    if has_moisture and has_cavity:
        flag = (moisture_dropping & cavity_rising & moisture_meaningful_drop & cavity_meaningful_rise) | moisture_dropping_7d
    elif has_moisture:
        flag = moisture_dropping_7d
    else:
        flag = cavity_delta_24h > CONFIG["cavity_rise_threshold_24h"] * 1.5

    # G5: CUSUM change-point detection for slow leaks
    if has_cavity and has_moisture:
        cavity_trend = df["hum_cavity"]  # trend signal for CUSUM
        cusum_cps = _cusum_changepoints(
            cavity_trend,
            threshold=CONFIG.get("cusum_threshold", 5.0),
            drift=CONFIG.get("cusum_drift", 0.5),
            direction="up",
        )
        confirm_samples = int(CONFIG.get("cusum_confirmation_window_hours", 72) * 12)
        for cp_time, _ in cusum_cps:
            cp_idx = df.index.get_indexer([cp_time], method="nearest")[0]
            window_end = min(cp_idx + confirm_samples, len(df) - 1)
            if window_end > cp_idx:
                moist_window = df[moisture_col].iloc[cp_idx:window_end]
                if len(moist_window) > 0:
                    moisture_drop = moist_window.iloc[0] - moist_window.min()
                    if moisture_drop > CONFIG["moisture_drop_min"]:
                        cusum_flag = pd.Series(False, index=df.index)
                        cusum_flag.iloc[cp_idx:window_end] = True
                        flag = flag | cusum_flag

    problems.extend(_extract_episodes(
        flag, df, device_id,
        problem_type="moisture_intrusion",
        min_hours=CONFIG["moisture_intrusion_min_hours"],
        detail_fn=lambda sub_df: {
            "moisture_drop_max": float(moisture_delta_24h.loc[sub_df.index].min()) if has_moisture else None,
            "cavity_rise_max": float(cavity_delta_24h.loc[sub_df.index].max()) if has_cavity else None,
            "moisture_range": (float(sub_df["moisture"].min()), float(sub_df["moisture"].max())) if has_moisture else None,
            "cavity_range": (float(sub_df["hum_cavity"].min()), float(sub_df["hum_cavity"].max())) if has_cavity else None,
        },
        severity_fn=lambda sub_df: (
            CRITICAL if (has_moisture and moisture_delta_7d.loc[sub_df.index].min() < -CONFIG["moisture_drop_threshold_7d"] * 2) else
            DANGER if (has_moisture and moisture_delta_24h.loc[sub_df.index].min() < -CONFIG["moisture_drop_threshold_24h"] * 2) else
            WARNING
        ),
        desc_fn=lambda sub_df, details: (
            f"Moisture resistance dropped {abs(details.get('moisture_drop_max') or 0):.1f} pct in 24h"
            + (f", cavity humidity rose {details.get('cavity_rise_max', 0):.1f} pct" if details.get("cavity_rise_max") else "")
        ),
    ))

    return problems


# ---------------------------------------------------------------------------
# Detector 2: CONDENSATION RISK (A1: seasonal-aware, A2: chronic merge)
# ---------------------------------------------------------------------------

def detect_condensation_risk(
    df: pd.DataFrame,
    device_id: str,
    fleet_seasonal: pd.DataFrame | None = None,  # G4: fleet seasonal reference
) -> list[Problem]:
    """
    Detect sustained high cavity humidity indicating condensation/mold risk.

    G3: Hysteresis on thresholds (enter at X%, exit at X-2%) to reduce toggling.
    G4: Fleet seasonal reference catches always-wet-but-stable cavities.
    G7: Absolute humidity as supplementary signal for borderline WARNING cases.
    G10: Chronic WARNING uses combined flag (not just above_absolute).
    """
    problems = []

    if "hum_cavity" not in df.columns:
        return problems

    hum = df["hum_cavity"]
    data_span_months = (df.index[-1] - df.index[0]).total_seconds() / (86400 * 30)

    # A1: Compute seasonal baseline
    seasonal = compute_seasonal_baseline(hum, CONFIG["seasonal_baseline_window_days"])

    # G3: Hysteresis band
    hysteresis_band = CONFIG.get("condensation_hysteresis_band", 2.0)

    for threshold, severity in [
        (CONFIG["condensation_critical_pct"], CRITICAL),
        (CONFIG["condensation_danger_pct"], DANGER),
        (CONFIG["condensation_warning_pct"], WARNING),
    ]:
        # G3: Apply hysteresis instead of simple threshold crossing
        exit_threshold = threshold - hysteresis_band
        above_absolute = _apply_hysteresis(hum, threshold, exit_threshold)

        # A1: For WARNING level, also require deviation above seasonal norm
        if severity == WARNING:
            above_seasonal = seasonal["deviation"] > 1.0
            flag = above_absolute & above_seasonal

            # G4: Fleet seasonal reference - catches "always-wet but stable" cavities
            if fleet_seasonal is not None and not fleet_seasonal.empty:
                months = df.index.month
                fleet_offset = CONFIG.get("fleet_seasonal_offset_pct", 8.0)
                fleet_median_vals = pd.Series(
                    [fleet_seasonal.loc[m, "median"] if m in fleet_seasonal.index else np.nan for m in months],
                    index=df.index,
                )
                above_fleet = hum > (fleet_median_vals + fleet_offset)
                flag = flag | (above_absolute & above_fleet)

            # G7: Absolute humidity as supplementary signal for borderline WARNING
            if CONFIG.get("condensation_use_abs_humidity", False):
                if "temp" in df.columns and "hum_ambient" in df.columns:
                    abs_hum = _compute_absolute_humidity(df["temp"], df["hum_ambient"])
                    high_abs = abs_hum > CONFIG.get("abs_humidity_warning_gkg", 14.0)
                    near_threshold = hum > (threshold - 5.0)
                    flag = flag | (near_threshold & high_abs & (seasonal["deviation"] > 0.5))
        else:
            flag = above_absolute

        # F3: Use different chronic threshold for WARNING vs DANGER/CRITICAL
        chronic_threshold = (CONFIG["condensation_chronic_pct_warning"] if severity == WARNING
                           else CONFIG["condensation_chronic_pct_severe"])

        # G10: Chronic check uses combined flag (not just above_absolute for WARNING)
        if data_span_months >= CONFIG["condensation_chronic_min_months"]:
            pct_above = float(flag.sum()) / max(len(flag), 1) * 100
            if pct_above >= chronic_threshold:
                # Emit single chronic assessment instead of many episodes
                mean_hum = float(hum[above_absolute].mean()) if above_absolute.any() else float(hum.mean())
                max_hum = float(hum.max())
                problems.append(Problem(
                    problem_type="condensation_risk",
                    severity=severity,
                    device_id=device_id,
                    start=df.index[0],
                    end=df.index[-1],
                    description=(
                        f"CHRONIC: Cavity humidity above {threshold}% for {pct_above:.0f}% of time "
                        f"(mean {mean_hum:.1f}%, peak {max_hum:.1f}%) over {data_span_months:.0f} months"
                    ),
                    details={
                        "max_cavity_humidity": max_hum,
                        "mean_cavity_humidity": mean_hum,
                        "threshold": threshold,
                        "pct_time_above": pct_above,
                        "chronic": True,
                        "dew_point_proximity": _dew_point_margin(df),
                    },
                ))
                continue  # skip episode extraction for this severity level

        # F1: Use larger merge gap for WARNING (21 days vs 7 days)
        merge_gap_hours = (CONFIG["condensation_warning_merge_gap_days"] * 24 if severity == WARNING
                          else 24 * 7)

        # Standard episode extraction for non-chronic cases
        episodes = _extract_episodes(
            flag, df, device_id,
            problem_type="condensation_risk",
            min_hours=CONFIG["condensation_min_hours"],
            merge_gap_hours=merge_gap_hours,
            detail_fn=lambda sub_df, thr=threshold: {
                "max_cavity_humidity": float(sub_df["hum_cavity"].max()),
                "mean_cavity_humidity": float(sub_df["hum_cavity"].mean()),
                "threshold": thr,
                "chronic": False,
                "dew_point_proximity": _dew_point_margin(sub_df) if "temp" in sub_df.columns else None,
            },
            severity_fn=lambda sub_df, sev=severity: sev,
            desc_fn=lambda sub_df, details, sev=severity: (
                f"Cavity humidity sustained at {details['mean_cavity_humidity']:.1f}% "
                f"(peak {details['max_cavity_humidity']:.1f}%) for >{CONFIG['condensation_min_hours']}h"
                + (f" - {sev}" if sev != WARNING else "")
            ),
        )

        # F2: Detect recurring pattern for WARNING episodes
        if severity == WARNING and data_span_months >= 12 and len(episodes) >= CONFIG["condensation_recurring_min_episodes"]:
            # Check if episodes span at least 12 months
            if episodes:
                earliest_ep = min(e.start for e in episodes)
                latest_ep = max(e.end for e in episodes)
                span_months = (latest_ep - earliest_ep).total_seconds() / (86400 * 30)

                if span_months >= 12:
                    # Replace fragmented episodes with single recurring assessment
                    mean_hum = float(hum[above_absolute].mean()) if above_absolute.any() else float(hum.mean())
                    max_hum = float(hum.max())
                    total_hours = sum(e.duration_hours for e in episodes)

                    episodes = [Problem(
                        problem_type="condensation_risk",
                        severity=WARNING,
                        device_id=device_id,
                        start=earliest_ep,
                        end=latest_ep,
                        description=(
                            f"RECURRING: {len(episodes)} condensation episodes over {span_months:.0f} months "
                            f"(total {total_hours:.0f}h, mean {mean_hum:.1f}%, peak {max_hum:.1f}%)"
                        ),
                        details={
                            "max_cavity_humidity": max_hum,
                            "mean_cavity_humidity": mean_hum,
                            "threshold": threshold,
                            "episode_count": len(episodes),
                            "recurring": True,
                            "dew_point_proximity": _dew_point_margin(df),
                        },
                    )]

        problems.extend(episodes)

    # Deduplicate: keep highest severity for overlapping periods
    problems = _deduplicate_problems(problems)
    return problems


def _dew_point_margin(df: pd.DataFrame) -> float | None:
    """Compute how close cavity conditions are to dew point (Magnus formula).
    G10: Uses ambient RH (not cavity) for consistent air mass."""
    if "temp" not in df.columns or "hum_ambient" not in df.columns:
        return None
    t = df["temp"].mean()
    rh = df["hum_ambient"].mean() / 100.0
    if rh <= 0:
        return None
    a, b = 17.27, 237.7
    alpha = (a * t) / (b + t) + np.log(rh)
    dew_point = (b * alpha) / (a - alpha)
    return float(t - dew_point)


# ---------------------------------------------------------------------------
# v6 Helper Functions (G1-G9)
# ---------------------------------------------------------------------------

def _time_delta(series: pd.Series, freq: str) -> pd.Series:
    """G2: Time-based delta: series(t) - series(t - freq).
    Unlike diff(N), this uses DatetimeIndex for correct gap handling."""
    shifted = series.shift(freq=freq)
    return series - shifted


def _apply_hysteresis(
    series: pd.Series,
    enter_threshold: float,
    exit_threshold: float,
) -> pd.Series:
    """G3: Vectorized threshold hysteresis.
    State enters True when series > enter_threshold,
    exits True when series < exit_threshold.
    Between exit and enter, state holds previous value."""
    events = pd.Series(np.nan, index=series.index)
    events[series > enter_threshold] = 1.0
    events[series < exit_threshold] = 0.0
    if np.isnan(events.iloc[0]):
        events.iloc[0] = 0.0
    return events.ffill().astype(bool)


def _compute_absolute_humidity(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """G7: Compute absolute humidity (g/kg) from temperature and relative humidity.
    Uses Tetens formula for saturation vapor pressure."""
    e_sat = 6.1078 * 10 ** ((7.5 * temp_c) / (237.3 + temp_c))
    e = e_sat * (rh_pct / 100.0)
    # Mixing ratio: w = 621.97 * e / (P - e), P=1013.25 hPa
    abs_humidity = 621.97 * e / (1013.25 - e)
    return abs_humidity


def _cusum_changepoints(
    series: pd.Series,
    threshold: float,
    drift: float,
    direction: str = "up",
) -> list[tuple[pd.Timestamp, float]]:
    """G5: One-sided CUSUM algorithm for change-point detection.
    Detects sustained mean shifts in the given direction.
    Returns list of (timestamp, cumulative_sum_at_alarm)."""
    values = series.dropna()
    if len(values) < 10:
        return []

    reference = values.iloc[:min(288, len(values))].mean()
    changepoints = []
    cusum = 0.0
    last_alarm_idx = -288  # allow first alarm after 1 day

    for i in range(len(values)):
        if direction == "up":
            cusum = max(0, cusum + (values.iloc[i] - reference - drift))
        else:
            cusum = max(0, cusum + (reference - values.iloc[i] - drift))

        if cusum > threshold and (i - last_alarm_idx) > 288:
            changepoints.append((values.index[i], cusum))
            cusum = 0.0
            last_alarm_idx = i
            reference = values.iloc[max(0, i - 288):i + 1].mean()

    return changepoints


def _fit_drying_curve(
    hum_series: pd.Series,
    initial_wet_threshold: float = 85.0,
    min_days: float = 60.0,
) -> dict | None:
    """G6: Fit exponential drying curve M(t) = a + b * exp(-t / tau).
    Returns dict with fit parameters, or None if fit fails."""
    from scipy.optimize import curve_fit

    hum = hum_series.dropna()
    if len(hum) < 288 * min_days:
        return None

    above_wet = hum > initial_wet_threshold
    if not above_wet.any():
        return None

    peak_idx = hum.idxmax()
    post_peak = hum.loc[peak_idx:]
    if len(post_peak) < 288 * 30:
        return None

    daily = post_peak.resample("D").mean().dropna()
    if len(daily) < 30:
        return None

    t = (daily.index - daily.index[0]).total_seconds() / 86400.0
    y = daily.values

    def exp_decay(t, a, b, tau):
        return a + b * np.exp(-t / tau)

    try:
        a0 = y[-1]
        b0 = max(y[0] - y[-1], 0.1)
        tau0 = 90.0
        popt, _ = curve_fit(
            exp_decay, t, y,
            p0=[a0, b0, tau0],
            bounds=([0, 0, 1], [100, 100, 10000]),
            maxfev=5000,
        )
        a, b, tau = popt

        y_pred = exp_decay(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "plateau_a": float(a), "amplitude_b": float(b), "tau_days": float(tau),
            "r_squared": float(r_squared), "peak_value": float(y[0]),
            "current_value": float(y[-1]), "fit_days": float(t[-1]),
        }
    except (RuntimeError, ValueError):
        return None


def _hampel_filter(series: pd.Series, window: int = 25, threshold: float = 3.0) -> pd.Series:
    """G8: Hampel filter for robust outlier detection.
    Returns boolean Series where True = detected spike/jump."""
    rolling_median = series.rolling(window, center=True, min_periods=window // 3).median()
    abs_deviation = (series - rolling_median).abs()
    mad = abs_deviation.rolling(window, center=True, min_periods=window // 3).median()
    mad = mad.clip(lower=1e-6)
    scaled_mad = mad * 1.4826  # MAD -> std for Gaussian
    return abs_deviation > (threshold * scaled_mad)


def _detect_sensor_drift(
    raw_series: pd.Series,
    ambient_rh: pd.Series,
    ambient_temp: pd.Series,
    window_days: int = 14,
) -> pd.Series:
    """G8: Detect sensor drift by monitoring rolling ratio between cavity and ambient."""
    valid = raw_series.notna() & ambient_rh.notna() & ambient_temp.notna()
    if valid.sum() < 288 * 7:
        return pd.Series(False, index=raw_series.index)

    window_samples = window_days * 288
    raw_rmean = raw_series.rolling(window_samples, min_periods=window_samples // 4).mean()
    amb_rmean = ambient_rh.rolling(window_samples, min_periods=window_samples // 4).mean()

    ratio = raw_rmean / amb_rmean.clip(lower=1.0)
    ratio_std = ratio.rolling(window_samples, min_periods=window_samples // 4).std()
    ratio_delta = ratio.diff(window_samples // 2).abs()

    drift_threshold = CONFIG.get("sensor_drift_threshold_std", 3.0)
    drift_flag = ratio_delta > (drift_threshold * ratio_std.clip(lower=0.01))
    return drift_flag.fillna(False)


def _compute_device_features(df: pd.DataFrame) -> dict:
    """G9: Compute robust features for inter-device comparison."""
    features = {}

    if "hum_cavity" in df.columns:
        hum = df["hum_cavity"].dropna()
        if len(hum) > 0:
            features["cavity_p95"] = float(hum.quantile(0.95))
            features["cavity_mean"] = float(hum.mean())
            features["pct_above_80"] = float((hum > 80).sum() / len(hum) * 100)
            features["pct_above_90"] = float((hum > 90).sum() / len(hum) * 100)
            # Max consecutive hours above 90
            above_90 = (hum > 90).astype(int)
            if above_90.any():
                groups = (above_90 != above_90.shift()).cumsum()
                max_streak = above_90.groupby(groups).sum().max()
                features["max_consec_hours_90"] = float(max_streak) / 12
            else:
                features["max_consec_hours_90"] = 0.0

    if "moisture" in df.columns:
        moist = df["moisture"].dropna()
        if len(moist) > 0:
            features["moisture_p05"] = float(moist.quantile(0.05))
            features["moisture_mean"] = float(moist.mean())
            last_30d = moist.iloc[-min(288 * 30, len(moist)):]
            if len(last_30d) > 288:
                x = np.arange(len(last_30d))
                slope = np.polyfit(x, last_30d.values, 1)[0]
                features["moisture_slope_30d"] = float(slope * 288)  # per day
            else:
                features["moisture_slope_30d"] = 0.0

    return features


def _mad_zscore(values: np.ndarray) -> np.ndarray:
    """G9: MAD (Median Absolute Deviation) z-scores. More robust than mean/std."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-6:
        return np.zeros_like(values)
    return (values - median) / (mad * 1.4826)


# ---------------------------------------------------------------------------
# Detector 3: DRYING FAILURE (G6: exponential curve)
# ---------------------------------------------------------------------------

def detect_drying_failure(df: pd.DataFrame, device_id: str) -> list[Problem]:
    """
    Detect when construction is not drying as expected.
    G6: Tries exponential curve fit first, falls back to linear slope.
    """
    problems = []

    has_moisture = "moisture" in df.columns
    has_cavity = "hum_cavity" in df.columns

    if not has_moisture and not has_cavity:
        return problems

    data_span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    if data_span_days < 90:
        return problems

    # G6: Try exponential drying curve on cavity humidity first
    if has_cavity:
        hum = df["hum_cavity"].dropna()
        fit = _fit_drying_curve(
            hum,
            initial_wet_threshold=CONFIG.get("drying_initial_wet_threshold", 85.0),
            min_days=CONFIG.get("drying_exp_fit_min_days", 60),
        )
        if fit is not None and fit["r_squared"] > 0.5:
            tau = fit["tau_days"]
            plateau = fit["plateau_a"]
            severity = None
            desc_prefix = ""

            if tau > CONFIG.get("drying_tau_danger_days", 730):
                severity = DANGER
                desc_prefix = f"Very slow drying: tau={tau:.0f} days"
            elif tau > CONFIG.get("drying_tau_warning_days", 365):
                severity = WARNING
                desc_prefix = f"Slow drying: tau={tau:.0f} days"

            if plateau > CONFIG.get("drying_plateau_pct", 80.0):
                if severity is None or severity == WARNING:
                    severity = DANGER
                desc_prefix = f"Drying plateau at {plateau:.1f}% (above safe level), tau={tau:.0f}d"

            if severity:
                problems.append(Problem(
                    problem_type="drying_failure",
                    severity=severity,
                    device_id=device_id,
                    start=df.index[0],
                    end=df.index[-1],
                    description=f"{desc_prefix}, R²={fit['r_squared']:.2f}",
                    details={
                        "trend_direction": "EXP_DRYING",
                        "tau_days": tau,
                        "plateau_pct": plateau,
                        "r_squared": fit["r_squared"],
                        "fit_params": fit,
                    },
                ))

    # Fallback: Linear slope on moisture resistance (existing v5 logic)
    if has_moisture:
        moisture = df["moisture"].dropna()
        if len(moisture) > 288 * 30:
            quarterly = moisture.resample("QS").agg(["mean", "std", "min", "max"])
            q_means = quarterly["mean"].dropna()
            if len(q_means) >= 2:
                x = np.arange(len(q_means))
                slope = np.polyfit(x, q_means.values, 1)[0]
                current_avg = float(moisture.iloc[-288*30:].mean())
                overall_avg = float(moisture.mean())

                is_wet = current_avg < 50
                is_worsening = slope < -0.5
                is_stagnant = abs(slope) < 0.3 and is_wet

                if is_wet and is_worsening:
                    problems.append(Problem(
                        problem_type="drying_failure",
                        severity=DANGER,
                        device_id=device_id,
                        start=df.index[0],
                        end=df.index[-1],
                        description=(
                            f"Chronic moisture: resistance WORSENING at {slope:+.1f} pct/quarter "
                            f"(current avg: {current_avg:.1f}%, overall: {overall_avg:.1f}%)"
                        ),
                        details={
                            "trend_direction": "WORSENING",
                            "slope_per_quarter": float(slope),
                            "current_moisture": current_avg,
                            "overall_moisture": overall_avg,
                            "data_span_months": data_span_days / 30,
                        },
                    ))
                elif is_stagnant:
                    problems.append(Problem(
                        problem_type="drying_failure",
                        severity=WARNING,
                        device_id=device_id,
                        start=df.index[0],
                        end=df.index[-1],
                        description=(
                            f"Chronic moisture: resistance STAGNANT at {current_avg:.1f}% "
                            f"(slope: {slope:+.1f} pct/quarter over {data_span_days/30:.0f} months)"
                        ),
                        details={
                            "trend_direction": "STAGNANT",
                            "slope_per_quarter": float(slope),
                            "current_moisture": current_avg,
                            "overall_moisture": overall_avg,
                            "data_span_months": data_span_days / 30,
                        },
                    ))

    if has_cavity:
        hum = df["hum_cavity"].dropna()
        if len(hum) > 288 * 30:
            quarterly_hum = hum.resample("QS").agg(["mean", "std", "min", "max"])
            q_hum_means = quarterly_hum["mean"].dropna()

            if len(q_hum_means) >= 2:
                x = np.arange(len(q_hum_means))
                hum_slope = np.polyfit(x, q_hum_means.values, 1)[0]
                current_hum = float(hum.iloc[-288*30:].mean())

                is_high = current_hum > 75
                is_rising = hum_slope > 0.5

                if is_high and is_rising:
                    problems.append(Problem(
                        problem_type="drying_failure",
                        severity=DANGER if current_hum > 85 else WARNING,
                        device_id=device_id,
                        start=df.index[0],
                        end=df.index[-1],
                        description=(
                            f"Cavity humidity RISING: {hum_slope:+.1f}%/quarter "
                            f"(current avg: {current_hum:.1f}%)"
                        ),
                        details={
                            "trend_direction": "CAVITY_HUM_RISING",
                            "slope_per_quarter": float(hum_slope),
                            "current_humidity": current_hum,
                        },
                    ))

    return problems


# ---------------------------------------------------------------------------
# Detector 4: SENSOR MALFUNCTION (A3: saturation-aware)
# ---------------------------------------------------------------------------

def detect_sensor_malfunction(df: pd.DataFrame, device_id: str) -> list[Problem]:
    """
    Detect sensor hardware issues using RAW (unsmoothed) data.

    A3: Flatlines at sensor boundary values (0% or 100% for humidity) are classified
    as 'sensor_saturation' rather than malfunction - they indicate the physical
    environment exceeded the sensor's range, not that the sensor is broken.
    """
    problems = []

    flatline_window = int(CONFIG["flatline_window_hours"] * 12)
    min_jump_count = CONFIG.get("jump_min_count", 3)
    saturation_map = CONFIG.get("saturation_values", {})

    for col, label, jump_thresh, valid_range in [
        ("temp", "Temperature", CONFIG["jump_threshold_temp"], CONFIG["temp_range"]),
        ("hum_ambient", "Ambient humidity", CONFIG["jump_threshold_humidity"], CONFIG["humidity_range"]),
        ("hum_cavity", "Cavity humidity", CONFIG["jump_threshold_humidity"], CONFIG["humidity_range"]),
        ("moisture", "Moisture resistance", CONFIG["jump_threshold_moisture"], CONFIG["moisture_range"]),
    ]:
        if col not in df.columns:
            continue

        raw_col = f"{col}_raw"
        series = df[raw_col] if raw_col in df.columns else df[col]

        # --- Flatline detection (A3: saturation-aware) ---
        rolling_std = series.rolling(flatline_window, min_periods=flatline_window // 2).std()
        flatline = rolling_std < 1e-6

        # A3: Check if flatline is at a saturation boundary value
        sat_values = saturation_map.get(col, [])
        if sat_values and flatline.any():
            flatline_values = series[flatline]
            is_saturated = pd.Series(False, index=df.index)
            for sv in sat_values:
                is_saturated = is_saturated | ((flatline_values - sv).abs() < 0.5).reindex(df.index, fill_value=False)

            # Real flatline = flatline AND NOT at saturation boundary
            real_flatline = flatline & ~is_saturated
            saturated_flatline = flatline & is_saturated

            # Report saturation as informational (lower severity, different subtype)
            if saturated_flatline.any():
                problems.extend(_extract_episodes(
                    saturated_flatline, df, device_id,
                    problem_type="sensor_malfunction",
                    min_hours=CONFIG["flatline_window_hours"],
                    detail_fn=lambda sub_df, c=col, rc=raw_col: {
                        "subtype": "sensor_saturation",
                        "channel": c,
                        "stuck_value": float(sub_df[rc].iloc[0]) if rc in sub_df.columns else (
                            float(sub_df[c].iloc[0]) if c in sub_df.columns else None
                        ),
                    },
                    severity_fn=lambda sub_df: OK,  # informational, not a malfunction
                    desc_fn=lambda sub_df, details: (
                        f"SATURATION: {label} pegged at {details.get('stuck_value', '?'):.1f} "
                        f"(sensor limit, not malfunction) for {(sub_df.index[-1] - sub_df.index[0]).total_seconds() / 3600:.0f}h"
                    ),
                    merge_gap_hours=48,
                ))
        else:
            real_flatline = flatline

        # Only report non-saturation flatlines as malfunction
        problems.extend(_extract_episodes(
            real_flatline, df, device_id,
            problem_type="sensor_malfunction",
            min_hours=CONFIG["flatline_window_hours"],
            detail_fn=lambda sub_df, c=col, rc=raw_col: {
                "subtype": "flatline",
                "channel": c,
                "stuck_value": float(sub_df[rc].iloc[0]) if rc in sub_df.columns else (
                    float(sub_df[c].iloc[0]) if c in sub_df.columns else None
                ),
            },
            severity_fn=lambda sub_df: WARNING,
            desc_fn=lambda sub_df, details: (
                f"FLATLINE: {label} stuck at {details.get('stuck_value', '?'):.2f} "
                f"for {(sub_df.index[-1] - sub_df.index[0]).total_seconds() / 3600:.0f}h"
            ),
            merge_gap_hours=48,
        ))

        # --- Jump detection (G8: enhanced with Hampel filter) ---
        delta = series.diff().abs()
        jumps = delta > jump_thresh
        jump_density = jumps.astype(int).rolling(12, min_periods=1).sum()
        jump_cluster = jump_density >= min_jump_count

        # G8: Hampel filter enhances jump detection (catches subtle spikes
        # that diff threshold may miss, but only where delta is non-trivial)
        hampel_outliers = _hampel_filter(
            series,
            window=CONFIG.get("hampel_window", 25),
            threshold=CONFIG.get("hampel_threshold", 3.0),
        )
        hampel_density = hampel_outliers.astype(int).rolling(12, min_periods=1).sum()
        # Require delta activity > 50% of threshold to avoid false positives on slow drift
        mild_activity = delta > (jump_thresh * 0.5)
        mild_density = mild_activity.astype(int).rolling(12, min_periods=1).sum()
        hampel_cluster = (hampel_density >= min_jump_count) & (mild_density >= 1)
        jump_cluster = jump_cluster | hampel_cluster

        if jump_cluster.any():
            # G8: Count both threshold jumps and Hampel-detected spikes
            combined_spikes = jumps | hampel_outliers
            problems.extend(_extract_episodes(
                jump_cluster, df, device_id,
                problem_type="sensor_malfunction",
                min_hours=0,
                detail_fn=lambda sub_df, c=col, rc=raw_col: {
                    "subtype": "jump",
                    "channel": c,
                    "max_jump": float(delta.loc[sub_df.index].max()),
                    "jump_count": int(combined_spikes.loc[sub_df.index].sum()),
                },
                severity_fn=lambda sub_df: DANGER,
                desc_fn=lambda sub_df, details: (
                    f"JUMP CLUSTER: {label} had {details.get('jump_count', 0)} spikes "
                    f"(max delta {details.get('max_jump', 0):.1f}) in {(sub_df.index[-1] - sub_df.index[0]).total_seconds() / 3600:.1f}h"
                ),
                merge_gap_hours=24,
            ))

        # --- Out-of-range detection ---
        oor = (series < valid_range[0]) | (series > valid_range[1])
        problems.extend(_extract_episodes(
            oor, df, device_id,
            problem_type="sensor_malfunction",
            min_hours=1,
            detail_fn=lambda sub_df, c=col, rc=raw_col: {
                "subtype": "out_of_range",
                "channel": c,
                "min_value": float(sub_df[rc].min()) if rc in sub_df.columns else (
                    float(sub_df[c].min()) if c in sub_df.columns else None
                ),
                "max_value": float(sub_df[rc].max()) if rc in sub_df.columns else (
                    float(sub_df[c].max()) if c in sub_df.columns else None
                ),
            },
            severity_fn=lambda sub_df: CRITICAL,
            desc_fn=lambda sub_df, details: (
                f"OUT OF RANGE: {label} = [{details.get('min_value', '?'):.1f}, {details.get('max_value', '?'):.1f}] "
                f"(valid: {valid_range})"
            ),
        ))

    # G8: Drift detection (only for cavity humidity - most impactful)
    raw_col_cav = "hum_cavity_raw"
    if raw_col_cav in df.columns:
        ambient_rh = df.get("hum_ambient_raw", df.get("hum_ambient"))
        ambient_temp = df.get("temp_raw", df.get("temp"))
        if ambient_rh is not None and ambient_temp is not None:
            drift_flag = _detect_sensor_drift(
                df[raw_col_cav], ambient_rh, ambient_temp,
                window_days=CONFIG.get("sensor_residual_window_days", 14),
            )
            if drift_flag.any():
                problems.extend(_extract_episodes(
                    drift_flag, df, device_id,
                    problem_type="sensor_malfunction",
                    min_hours=CONFIG.get("sensor_residual_window_days", 14) * 24,
                    detail_fn=lambda sub_df: {"subtype": "drift", "channel": "hum_cavity"},
                    severity_fn=lambda sub_df: WARNING,
                    desc_fn=lambda sub_df, details: (
                        f"DRIFT: Cavity humidity tracking ratio shifted "
                        f"for {(sub_df.index[-1] - sub_df.index[0]).total_seconds() / 3600:.0f}h"
                    ),
                    merge_gap_hours=48,
                ))

    # G8: Stuck-at-mid-value detection (humidity channels only)
    for col_mid in ("hum_cavity", "hum_ambient"):
        if col_mid not in df.columns:
            continue
        raw_col_mid = f"{col_mid}_raw"
        series_mid = df[raw_col_mid] if raw_col_mid in df.columns else df[col_mid]
        rolling_range = series_mid.rolling(flatline_window, min_periods=flatline_window // 2).apply(
            lambda x: x.max() - x.min(), raw=True
        )
        mid_val = 50.0  # midpoint of 0-100 range
        near_mid = (series_mid - mid_val).abs() < 10
        stuck_mid = (rolling_range < 0.5) & near_mid
        # Exclude values already caught by flatline detection
        series_std = series_mid.rolling(flatline_window, min_periods=flatline_window // 2).std()
        already_flatline = series_std < 1e-6
        stuck_mid = stuck_mid & ~already_flatline
        if stuck_mid.any():
            label_mid = "Cavity humidity" if col_mid == "hum_cavity" else "Ambient humidity"
            problems.extend(_extract_episodes(
                stuck_mid, df, device_id,
                problem_type="sensor_malfunction",
                min_hours=CONFIG["flatline_window_hours"],
                detail_fn=lambda sub_df, c=col_mid: {
                    "subtype": "stuck_mid_value",
                    "channel": c,
                    "stuck_value": float(sub_df[c].mean()) if c in sub_df.columns else None,
                },
                severity_fn=lambda sub_df: WARNING,
                desc_fn=lambda sub_df, details: (
                    f"STUCK MID-VALUE: {label_mid} near {details.get('stuck_value', '?'):.1f} "
                    f"with <0.5 pct variation for {(sub_df.index[-1] - sub_df.index[0]).total_seconds() / 3600:.0f}h"
                ),
                merge_gap_hours=48,
            ))

    # Filter out OK-severity (saturation) from problem count, but keep them for reporting
    real_problems = [p for p in problems if p.severity != OK]
    sat_problems = [p for p in problems if p.severity == OK]

    # Cap real problems at 10 per device
    if len(real_problems) > 10:
        total_count = len(real_problems)
        severity_order = {CRITICAL: 3, DANGER: 2, WARNING: 1, OK: 0}
        max_severity = max(real_problems, key=lambda p: severity_order.get(p.severity, 0)).severity
        earliest = min(p.start for p in real_problems)
        latest = max(p.end for p in real_problems)
        total_hours = sum(p.duration_hours for p in real_problems)

        real_problems.sort(key=lambda p: (severity_order.get(p.severity, 0), p.duration_hours), reverse=True)
        kept = real_problems[:5]
        kept.append(Problem(
            problem_type="sensor_malfunction",
            severity=max_severity,
            device_id=device_id,
            start=earliest,
            end=latest,
            description=(
                f"CHRONIC SENSOR ISSUES: {total_count} episodes detected "
                f"({total_hours:.0f}h total malfunction time) - sensor may need replacement"
            ),
            details={"subtype": "chronic_summary", "total_episodes": total_count, "total_hours": total_hours},
        ))
        real_problems = kept

    return real_problems + sat_problems


# ---------------------------------------------------------------------------
# Detector 5: RAPID MOISTURE CHANGE (A5)
# ---------------------------------------------------------------------------

def detect_rapid_moisture_change(df: pd.DataFrame, device_id: str) -> list[Problem]:
    """
    Derivative-based detector for rapid moisture resistance transitions.

    G1: Uses denoised signal for better peak detection.
    G2: Time-based deltas instead of fixed-step diff.
    G5: CUSUM change-point detection for sustained drops.
    G10: Skip 14d logic when insufficient data span.
    """
    problems = []

    if "moisture" not in df.columns:
        return problems

    # G1: Use denoised signal for acute detection
    moisture = df["moisture_denoised"] if "moisture_denoised" in df.columns else df["moisture"]

    data_span = df.index[-1] - df.index[0]
    if data_span < pd.Timedelta("3D"):
        return problems

    # G2: Time-based deltas
    delta_3d = _time_delta(moisture, "3D")
    rate_3d = delta_3d / 3.0

    if data_span >= pd.Timedelta("14D"):
        delta_14d = _time_delta(moisture, "14D")
        rate_14d = delta_14d / 14.0
        rapid_14d = delta_14d < -CONFIG["rapid_moisture_drop_14d"]
    else:
        delta_14d = delta_3d
        rate_14d = rate_3d
        rapid_14d = pd.Series(False, index=df.index)  # G10: skip 14d if insufficient span

    # Flag rapid drops
    rapid_3d = delta_3d < -CONFIG["rapid_moisture_drop_3d"]
    flag = rapid_3d | rapid_14d

    # G5: CUSUM change-point detection for sustained moisture decline
    cusum_down = _cusum_changepoints(
        moisture,
        threshold=CONFIG.get("cusum_threshold", 5.0),
        drift=CONFIG.get("cusum_drift", 0.5),
        direction="down",
    )
    confirm_samples = int(CONFIG.get("cusum_confirmation_window_hours", 72) * 12)
    for cp_time, _ in cusum_down:
        cp_idx = df.index.get_indexer([cp_time], method="nearest")[0]
        start_idx = max(0, cp_idx - confirm_samples // 2)
        end_idx = min(len(df), cp_idx + confirm_samples // 2)
        cusum_flag = pd.Series(False, index=df.index)
        cusum_flag.iloc[start_idx:end_idx] = True
        flag = flag | cusum_flag

    problems.extend(_extract_episodes(
        flag, df, device_id,
        problem_type="rapid_moisture_change",
        min_hours=CONFIG["rapid_change_min_hours"],
        merge_gap_hours=48,
        detail_fn=lambda sub_df: {
            "max_drop_3d": float(delta_3d.loc[sub_df.index].min()),
            "max_rate_per_day": float(rate_3d.loc[sub_df.index].min()),
            "moisture_start": float(moisture.loc[sub_df.index[0]]) if not np.isnan(moisture.loc[sub_df.index[0]]) else None,
            "moisture_end": float(moisture.loc[sub_df.index[-1]]) if not np.isnan(moisture.loc[sub_df.index[-1]]) else None,
            "moisture_min": float(moisture.loc[sub_df.index].min()),
            "moisture_max": float(moisture.loc[sub_df.index].max()),
        },
        severity_fn=lambda sub_df: (
            CRITICAL if delta_3d.loc[sub_df.index].min() < -CONFIG["rapid_moisture_drop_3d"] * 3 else
            DANGER if delta_3d.loc[sub_df.index].min() < -CONFIG["rapid_moisture_drop_3d"] * 2 else
            WARNING
        ),
        desc_fn=lambda sub_df, details: (
            f"Rapid moisture drop: peak derivative -{abs(details['max_drop_3d']):.1f} pct-points in 3 days "
            f"(rate: {abs(details['max_rate_per_day']):.2f} pct/day); "
            + (f"episode range {details['moisture_start']:.1f}% -> {details['moisture_end']:.1f}%"
               if details.get('moisture_start') is not None else "episode range unknown")
        ),
    ))

    return problems


# ---------------------------------------------------------------------------
# C11: Composite health score
# ---------------------------------------------------------------------------

def compute_health_score(problems: list[Problem], data_span_hours: float) -> dict:
    """
    C11: Compute composite health score for a device (0-100, where 100 = healthy).

    Combines problem count, severity, and duration relative to data span.
    """
    if data_span_hours <= 0:
        return {"score": 100, "grade": "A", "penalty_breakdown": {}}

    weights = CONFIG["health_score_weights"]
    total_penalty = 0.0
    breakdown = {}

    for p in problems:
        if p.severity == OK:
            continue
        w = weights.get(p.problem_type, {}).get(p.severity, 5)
        # Duration factor: longer problems penalize more, but cap at 1.0 for chronic
        duration_factor = min(p.duration_hours / data_span_hours, 1.0)
        # Penalty = weight * (0.3 base + 0.7 duration-weighted)
        penalty = w * (0.3 + 0.7 * duration_factor)
        total_penalty += penalty
        key = f"{p.problem_type}:{p.severity}"
        breakdown[key] = breakdown.get(key, 0) + penalty

    score = max(0, 100 - total_penalty)

    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 50:
        grade = "C"
    elif score >= 25:
        grade = "D"
    else:
        grade = "F"

    return {"score": round(score, 1), "grade": grade, "penalty_breakdown": breakdown}


# ---------------------------------------------------------------------------
# C12: Inter-device comparison within installation
# ---------------------------------------------------------------------------

def detect_installation_outliers(
    devices: dict[str, pd.DataFrame],
    all_problems: dict[str, list[Problem]],
    installations: dict[str, dict[str, list[Problem]]],
) -> dict[str, list[Problem]]:
    """
    G9: Flag devices whose behavior deviates significantly from siblings.
    Uses robust features (p95, %time>threshold, consecutive hours) and MAD z-scores.
    """
    extra_problems: dict[str, list[Problem]] = {}
    mad_threshold = CONFIG.get("outlier_mad_zscore_threshold", 3.0)
    min_devices = CONFIG.get("outlier_min_devices", 3)

    for inst_id, inst_device_problems in installations.items():
        inst_device_ids = list(inst_device_problems.keys())
        if len(inst_device_ids) < min_devices:
            continue

        # G9: Compute robust features for all devices in this installation
        all_features = {}
        for did in inst_device_ids:
            if did in devices:
                all_features[did] = _compute_device_features(devices[did])

        if len(all_features) < min_devices:
            continue

        # Collect all feature keys
        feature_keys = set()
        for f in all_features.values():
            feature_keys.update(f.keys())

        for fkey in feature_keys:
            values = []
            dids = []
            for did, f in all_features.items():
                if fkey in f:
                    values.append(f[fkey])
                    dids.append(did)

            if len(values) < min_devices:
                continue

            values_arr = np.array(values)
            z_scores = _mad_zscore(values_arr)

            for i, (did, z) in enumerate(zip(dids, z_scores)):
                # High cavity humidity features (high = bad)
                if fkey in ("cavity_p95", "pct_above_80", "pct_above_90", "max_consec_hours_90", "cavity_mean") and z > mad_threshold:
                    inst_median = float(np.median(values_arr))
                    p = Problem(
                        problem_type="condensation_risk",
                        severity=WARNING,
                        device_id=did,
                        start=devices[did].index[0],
                        end=devices[did].index[-1],
                        description=(
                            f"OUTLIER: {fkey}={values_arr[i]:.1f} vs installation median {inst_median:.1f} "
                            f"(MAD z={z:.1f})"
                        ),
                        details={
                            "subtype": "installation_outlier",
                            "feature": fkey,
                            "device_value": float(values_arr[i]),
                            "installation_median": inst_median,
                            "mad_zscore": float(z),
                        },
                    )
                    extra_problems.setdefault(did, []).append(p)

                # Low moisture features (low = bad)
                elif fkey in ("moisture_p05", "moisture_mean") and z < -mad_threshold:
                    inst_median = float(np.median(values_arr))
                    p = Problem(
                        problem_type="drying_failure",
                        severity=WARNING,
                        device_id=did,
                        start=devices[did].index[0],
                        end=devices[did].index[-1],
                        description=(
                            f"OUTLIER: {fkey}={values_arr[i]:.1f} vs installation median {inst_median:.1f} "
                            f"(MAD z={z:.1f})"
                        ),
                        details={
                            "subtype": "installation_outlier",
                            "feature": fkey,
                            "device_value": float(values_arr[i]),
                            "installation_median": inst_median,
                            "mad_zscore": float(z),
                        },
                    )
                    extra_problems.setdefault(did, []).append(p)

    return extra_problems


# ---------------------------------------------------------------------------
# Helper: extract contiguous episodes from a boolean flag series
# ---------------------------------------------------------------------------

def _extract_episodes(
    flag: pd.Series,
    df: pd.DataFrame,
    device_id: str,
    problem_type: str,
    min_hours: float,
    detail_fn,
    severity_fn,
    desc_fn,
    merge_gap_hours: float | None = None,
) -> list[Problem]:
    """Find contiguous True regions in flag, merge nearby ones, create Problem for each."""
    problems = []
    if flag.sum() == 0:
        return problems

    if merge_gap_hours is None:
        merge_gap_hours = CONFIG["episode_merge_gap_hours"]

    flag = flag.reindex(df.index).fillna(False)
    groups = (flag != flag.shift()).cumsum()
    flagged_groups = groups[flag]

    intervals = []
    for _, grp in flagged_groups.groupby(flagged_groups):
        intervals.append((grp.index[0], grp.index[-1]))

    if not intervals:
        return problems

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        gap_hours = (start - prev_end).total_seconds() / 3600
        if gap_hours <= merge_gap_hours:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    for start, end in merged:
        duration_hours = (end - start).total_seconds() / 3600
        if duration_hours < min_hours:
            continue

        sub_df = df.loc[start:end]
        details = detail_fn(sub_df)
        severity = severity_fn(sub_df)
        description = desc_fn(sub_df, details)

        problems.append(Problem(
            problem_type=problem_type,
            severity=severity,
            device_id=device_id,
            start=start,
            end=end,
            description=description,
            details=details,
        ))

    return problems


def _deduplicate_problems(problems: list[Problem]) -> list[Problem]:
    """Remove lower-severity problems that overlap with higher-severity ones."""
    severity_order = {CRITICAL: 3, DANGER: 2, WARNING: 1, OK: 0}
    problems.sort(key=lambda p: severity_order.get(p.severity, 0), reverse=True)

    kept = []
    for p in problems:
        overlaps_higher = any(
            k.start <= p.end and k.end >= p.start and
            severity_order.get(k.severity, 0) > severity_order.get(p.severity, 0)
            for k in kept
        )
        if not overlaps_higher:
            kept.append(p)
    return kept


# ---------------------------------------------------------------------------
# Visualization (B6-B10 improvements)
# ---------------------------------------------------------------------------

def visualize_device_problems(
    df: pd.DataFrame,
    problems: list[Problem],
    device_id: str,
    output_dir: str = ".",
    fleet_seasonal: pd.DataFrame | None = None,
    health_score: dict | None = None,
) -> str:
    """
    Plot device time series with problem periods highlighted.

    B6: Dual-axis overlay panel for cavity humidity + moisture resistance.
    B7: Chronic problems shown with hatching; acute with solid shading.
    B8: Seasonal reference band on cavity humidity panel.
    B9: Color-coded time series lines by threshold exceedance.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    available = [c for c in COLUMN_NAMES if c in df.columns]

    # B6: Add dual-axis overlay panel if both cavity and moisture available
    has_dual = "hum_cavity" in df.columns and "moisture" in df.columns

    n_panels = len(available) + 1 + (1 if has_dual else 0)  # +1 timeline, +1 dual-axis
    fig, axes = plt.subplots(n_panels, 1, figsize=(20, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    labels = {
        "temp": "Temperature (C)",
        "hum_ambient": "Ambient Humidity (%)",
        "hum_cavity": "Cavity Humidity (%)",
        "moisture": "Moisture Resistance (%)",
    }

    severity_colors = {
        OK: "#90CAF9",       # light blue (informational)
        WARNING: "#FFD700",   # gold
        DANGER: "#FF8C00",    # dark orange
        CRITICAL: "#DC143C",  # crimson
    }

    # B7+F6: Separate acute, chronic, and recurring problems
    acute_problems = [p for p in problems if not p.is_chronic and not p.details.get("recurring")]
    chronic_problems = [p for p in problems if p.is_chronic and p.severity != OK and not p.details.get("recurring")]
    recurring_problems = [p for p in problems if p.details.get("recurring") and p.severity != OK]

    ax_idx = 0

    # Plot each individual channel
    for col in available:
        ax = axes[ax_idx]
        ax_idx += 1

        # B9: Color-code the time series line
        if col == "hum_cavity":
            _plot_colored_line(ax, df.index, df[col],
                               thresholds=[
                                   (CONFIG["condensation_critical_pct"], severity_colors[CRITICAL]),
                                   (CONFIG["condensation_danger_pct"], severity_colors[DANGER]),
                                   (CONFIG["condensation_warning_pct"], severity_colors[WARNING]),
                               ],
                               default_color="#333")
            # Threshold lines
            ax.axhline(80, color=severity_colors[WARNING], linestyle="--", linewidth=0.8, alpha=0.5, label="80% (mold)")
            ax.axhline(90, color=severity_colors[DANGER], linestyle="--", linewidth=0.8, alpha=0.5, label="90% (condensation)")
            ax.axhline(95, color=severity_colors[CRITICAL], linestyle="--", linewidth=0.8, alpha=0.5, label="95% (active)")

            # B8: Seasonal reference band
            if fleet_seasonal is not None:
                _plot_seasonal_band(ax, df.index, fleet_seasonal)

        else:
            ax.plot(df.index, df[col], linewidth=0.4, color="#333", alpha=0.8)

        ax.set_ylabel(labels.get(col, col), fontsize=9)
        ax.grid(True, alpha=0.2)

        # Shade acute problems
        for p in acute_problems:
            if p.severity == OK:
                continue
            color = severity_colors.get(p.severity, "#999")
            ax.axvspan(p.start, p.end, alpha=0.15, color=color, zorder=0)

        # B7: Hatch chronic problems (subtle diagonal lines instead of solid fill)
        for p in chronic_problems:
            color = severity_colors.get(p.severity, "#999")
            ax.axvspan(p.start, p.end, alpha=0.05, color=color, zorder=0,
                       hatch='///', linewidth=0)

        # F6: Recurring problems shown as semi-transparent band
        for p in recurring_problems:
            color = severity_colors.get(p.severity, "#999")
            ax.axvspan(p.start, p.end, alpha=0.08, color=color, zorder=0)

    # B6: Dual-axis overlay panel
    if has_dual:
        ax_dual = axes[ax_idx]
        ax_idx += 1

        color_hum = "#1565C0"
        color_moist = "#E65100"

        ax_dual.set_ylabel("Cavity Humidity (%)", fontsize=9, color=color_hum)
        _plot_colored_line(ax_dual, df.index, df["hum_cavity"],
                          thresholds=[
                              (CONFIG["condensation_critical_pct"], severity_colors[CRITICAL]),
                              (CONFIG["condensation_danger_pct"], severity_colors[DANGER]),
                              (CONFIG["condensation_warning_pct"], severity_colors[WARNING]),
                          ],
                          default_color=color_hum, linewidth=0.6)
        ax_dual.tick_params(axis="y", labelcolor=color_hum)

        ax_dual2 = ax_dual.twinx()
        ax_dual2.plot(df.index, df["moisture"], linewidth=0.6, color=color_moist, alpha=0.7)
        ax_dual2.set_ylabel("Moisture Resistance (%)", fontsize=9, color=color_moist)
        ax_dual2.tick_params(axis="y", labelcolor=color_moist)
        ax_dual2.invert_yaxis()  # invert so that low moisture = top (aligned with high humidity)

        ax_dual.set_title("Cavity Humidity vs Moisture Resistance (inverted)", fontsize=9, fontstyle="italic")
        ax_dual.grid(True, alpha=0.2)

        for p in acute_problems:
            if p.severity == OK:
                continue
            color = severity_colors.get(p.severity, "#999")
            ax_dual.axvspan(p.start, p.end, alpha=0.15, color=color, zorder=0)

    # B7: Problem timeline panel with chronic/acute distinction
    ax_timeline = axes[ax_idx]
    type_y = {
        "moisture_intrusion": 4,
        "rapid_moisture_change": 3,
        "condensation_risk": 2,
        "drying_failure": 1,
        "sensor_malfunction": 0,
    }
    type_labels = {
        "moisture_intrusion": "Moisture Intrusion",
        "rapid_moisture_change": "Rapid Moisture Change",
        "condensation_risk": "Condensation",
        "drying_failure": "Drying Failure",
        "sensor_malfunction": "Sensor Malfunction",
    }

    for p in problems:
        if p.severity == OK:
            continue
        y = type_y.get(p.problem_type, 0)
        color = severity_colors.get(p.severity, "#999")

        if p.details.get("recurring"):
            # F6: Recurring = semi-transparent continuous band with "RECURRING" label
            ax_timeline.barh(y, (p.end - p.start).total_seconds() / 86400,
                             left=p.start, height=0.6, color=color, alpha=0.2,
                             edgecolor=color, linewidth=1.0, linestyle="--")
            # Add "RECURRING" label at center
            mid_time = p.start + (p.end - p.start) / 2
            ax_timeline.text(mid_time, y, f"RECURRING ({p.details.get('episode_count', '?')} episodes)",
                             ha="center", va="center",
                             fontsize=6, fontweight="bold", color="#333", alpha=0.7)
        elif p.is_chronic:
            # B7: Chronic = hatched bar with text label
            ax_timeline.barh(y, (p.end - p.start).total_seconds() / 86400,
                             left=p.start, height=0.6, color=color, alpha=0.3,
                             edgecolor=color, linewidth=1.0, hatch='///')
            # Add "CHRONIC" label at center
            mid_time = p.start + (p.end - p.start) / 2
            ax_timeline.text(mid_time, y, "CHRONIC", ha="center", va="center",
                             fontsize=6, fontweight="bold", color="#333", alpha=0.7)
        else:
            ax_timeline.barh(y, (p.end - p.start).total_seconds() / 86400,
                             left=p.start, height=0.6, color=color, alpha=0.7,
                             edgecolor="black", linewidth=0.3)

    ax_timeline.set_yticks(list(type_y.values()))
    ax_timeline.set_yticklabels(list(type_labels.values()), fontsize=8)
    ax_timeline.set_ylim(-0.5, max(type_y.values()) + 0.5)
    ax_timeline.grid(True, axis="x", alpha=0.2)
    ax_timeline.set_ylabel("Problems", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=c, label=s.upper(), alpha=0.7)
        for s, c in severity_colors.items() if s != OK
    ]
    ax_timeline.legend(handles=legend_patches, loc="upper right", fontsize=7)

    # Title with health score
    score_str = ""
    if health_score:
        score_str = f"  |  Health: {health_score['score']}/100 (Grade {health_score['grade']})"
    fig.suptitle(
        f"Building Health - Device {device_id[:12]}...{score_str}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"problems_{device_id[:12]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _plot_colored_line(ax, x, y, thresholds, default_color="#333", linewidth=0.4):
    """
    B9: Plot a time series line with color changing based on threshold exceedance.

    thresholds: list of (value, color) sorted highest-first.
    """
    y_arr = np.array(y, dtype=float)
    x_arr = np.array(x)

    # Plot base line in default color
    ax.plot(x_arr, y_arr, linewidth=linewidth, color=default_color, alpha=0.4, zorder=1)

    # Overlay colored segments for each threshold (highest first)
    for thresh, color in thresholds:
        mask = y_arr > thresh
        if not mask.any():
            continue
        y_colored = y_arr.copy()
        y_colored[~mask] = np.nan
        ax.plot(x_arr, y_colored, linewidth=linewidth * 2, color=color, alpha=0.8, zorder=2)


def _plot_seasonal_band(ax, index, fleet_seasonal):
    """B8: Plot seasonal reference band (fleet median +/- IQR by month)."""
    if fleet_seasonal is None or fleet_seasonal.empty:
        return

    months = index.month
    median_vals = np.array([fleet_seasonal.loc[m, "median"] if m in fleet_seasonal.index else np.nan for m in months])
    q25_vals = np.array([fleet_seasonal.loc[m, "q25"] if m in fleet_seasonal.index else np.nan for m in months])
    q75_vals = np.array([fleet_seasonal.loc[m, "q75"] if m in fleet_seasonal.index else np.nan for m in months])

    ax.fill_between(index, q25_vals, q75_vals, alpha=0.08, color="#4CAF50", label="Fleet seasonal IQR")
    ax.plot(index, median_vals, linewidth=0.8, color="#4CAF50", alpha=0.4, linestyle=":", label="Fleet median")


def visualize_installation_summary(
    all_problems: dict[str, list[Problem]],
    installation_id: str,
    output_dir: str = ".",
    health_scores: dict[str, dict] | None = None,
) -> str:
    """Heatmap showing problem severity per device over time, with health scores."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    severity_val = {OK: 0, WARNING: 1, DANGER: 2, CRITICAL: 3}
    severity_colors_map = {0: "#E8F5E9", 1: "#FFD700", 2: "#FF8C00", 3: "#DC143C"}

    device_ids = sorted(all_problems.keys())
    if not device_ids:
        return ""

    all_p = [p for probs in all_problems.values() for p in probs if p.severity != OK]
    if not all_p:
        return ""

    min_date = min(p.start for p in all_p)
    max_date = max(p.end for p in all_p)
    months = pd.date_range(min_date.to_period("M").to_timestamp(), max_date, freq="MS")

    matrix = np.zeros((len(device_ids), len(months)))
    for i, did in enumerate(device_ids):
        for p in all_problems.get(did, []):
            sv = severity_val.get(p.severity, 0)
            for j, m in enumerate(months):
                m_end = m + pd.offsets.MonthEnd(1)
                if p.start <= m_end and p.end >= m:
                    matrix[i, j] = max(matrix[i, j], sv)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([severity_colors_map[k] for k in sorted(severity_colors_map)])

    fig, ax = plt.subplots(figsize=(20, max(4, len(device_ids) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")

    # Y-axis labels: device ID + health score
    ylabels = []
    for d in device_ids:
        label = d[:12]
        if health_scores and d in health_scores:
            hs = health_scores[d]
            label += f" [{hs['grade']}:{hs['score']}]"
        ylabels.append(label)

    ax.set_yticks(range(len(device_ids)))
    ax.set_yticklabels(ylabels, fontsize=7)
    month_labels = [m.strftime("%Y-%m") for m in months]
    step = max(1, len(month_labels) // 20)
    ax.set_xticks(range(0, len(month_labels), step))
    ax.set_xticklabels(month_labels[::step], fontsize=7, rotation=45, ha="right")

    legend_patches = [
        mpatches.Patch(color=severity_colors_map[0], label="OK"),
        mpatches.Patch(color=severity_colors_map[1], label="WARNING"),
        mpatches.Patch(color=severity_colors_map[2], label="DANGER"),
        mpatches.Patch(color=severity_colors_map[3], label="CRITICAL"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7)
    ax.set_title(f"Building Health Summary - Installation {installation_id[:16]}...", fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"summary_{installation_id[:12]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# B10: Per-installation comparison panel (box plots)
# ---------------------------------------------------------------------------

def visualize_installation_comparison(
    devices: dict[str, pd.DataFrame],
    inst_device_ids: list[str],
    installation_id: str,
    output_dir: str = ".",
    health_scores: dict[str, dict] | None = None,
) -> str:
    """
    B10: Box/violin plot comparing cavity humidity and moisture resistance
    distributions across devices in the same installation.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, max(4, len(inst_device_ids) * 0.4)))

    # Prepare data
    cavity_data = []
    moisture_data = []
    device_labels = []

    for did in sorted(inst_device_ids):
        if did not in devices:
            continue
        df = devices[did]
        label = did[:12]
        if health_scores and did in health_scores:
            label += f" [{health_scores[did]['grade']}]"
        device_labels.append(label)

        if "hum_cavity" in df.columns:
            cavity_data.append(df["hum_cavity"].dropna().values)
        else:
            cavity_data.append(np.array([]))

        if "moisture" in df.columns:
            moisture_data.append(df["moisture"].dropna().values)
        else:
            moisture_data.append(np.array([]))

    if not device_labels:
        plt.close()
        return ""

    # Cavity humidity box plot
    ax1 = axes[0]
    bp1 = ax1.boxplot(
        [d for d in cavity_data],
        vert=False, patch_artist=True,
        widths=0.6, showfliers=False,
    )
    for i, box in enumerate(bp1["boxes"]):
        median_val = np.median(cavity_data[i]) if len(cavity_data[i]) > 0 else 0
        if median_val > 95:
            box.set_facecolor("#DC143C")
        elif median_val > 90:
            box.set_facecolor("#FF8C00")
        elif median_val > 80:
            box.set_facecolor("#FFD700")
        else:
            box.set_facecolor("#E8F5E9")
        box.set_alpha(0.7)

    ax1.set_yticklabels(device_labels, fontsize=7)
    ax1.set_xlabel("Cavity Humidity (%)", fontsize=9)
    ax1.axvline(80, color="#FFD700", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axvline(90, color="#FF8C00", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axvline(95, color="#DC143C", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_title("Cavity Humidity Distribution", fontsize=10)
    ax1.grid(True, axis="x", alpha=0.2)

    # Moisture resistance box plot
    ax2 = axes[1]
    bp2 = ax2.boxplot(
        [d for d in moisture_data],
        vert=False, patch_artist=True,
        widths=0.6, showfliers=False,
    )
    for i, box in enumerate(bp2["boxes"]):
        median_val = np.median(moisture_data[i]) if len(moisture_data[i]) > 0 else 50
        if median_val < 15:
            box.set_facecolor("#DC143C")
        elif median_val < 30:
            box.set_facecolor("#FF8C00")
        elif median_val < 50:
            box.set_facecolor("#FFD700")
        else:
            box.set_facecolor("#E8F5E9")
        box.set_alpha(0.7)

    ax2.set_yticklabels(device_labels, fontsize=7)
    ax2.set_xlabel("Moisture Resistance (%)", fontsize=9)
    ax2.set_title("Moisture Resistance Distribution", fontsize=10)
    ax2.grid(True, axis="x", alpha=0.2)

    fig.suptitle(f"Device Comparison - Installation {installation_id[:16]}...", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"comparison_{installation_id[:12]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: str,
    output_dir: str = "results",
    max_devices: int | None = None,
    report_quarter: str | None = None,
):
    """Run building problem detection pipeline (v6)."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SENZOMATIC BUILDING PROBLEM DETECTION v6")
    print("=" * 70)

    # Load data
    print("\nLoading sensor data...")
    devices = load_device_data(data_dir)

    if max_devices:
        device_ids = list(devices.keys())[:max_devices]
        devices = {k: devices[k] for k in device_ids}
        print(f"  Limited to {max_devices} devices for testing")

    if report_quarter:
        year, q = report_quarter.split("-Q")
        q = int(q)
        q_start = pd.Timestamp(f"{year}-{(q-1)*3+1:02d}-01")
        q_end = q_start + pd.DateOffset(months=3)
        print(f"  Filtering to quarter {report_quarter}: {q_start:%Y-%m-%d} -> {q_end:%Y-%m-%d}")
        for did in list(devices.keys()):
            devices[did] = devices[did].loc[q_start:q_end]
            if len(devices[did]) == 0:
                del devices[did]

    # Preprocess all devices
    print("\nPreprocessing: median filter + smoothing...")
    for device_id in list(devices.keys()):
        df = devices[device_id]
        if len(df) < CONFIG["min_data_hours"] * 12:
            print(f"  Skipping {device_id[:12]}... (only {len(df)} samples, <{CONFIG['min_data_hours']}h)")
            del devices[device_id]
            continue
        devices[device_id] = preprocess_device_data(df)
    print(f"  {len(devices)} devices after preprocessing")

    # B8: Compute fleet seasonal profile
    print("\nComputing fleet seasonal profile...")
    fleet_seasonal = compute_fleet_seasonal_profile(devices, "hum_cavity")
    if fleet_seasonal is not None:
        print(f"  Seasonal profile computed from {len(devices)} devices")

    # Run all detectors on each device
    all_problems: dict[str, list[Problem]] = {}
    health_scores: dict[str, dict] = {}
    summary_rows = []

    for i, (device_id, df) in enumerate(devices.items()):
        print(f"\n[{i+1}/{len(devices)}] Device {device_id[:12]}... ({len(df)} samples, channels: {[c for c in df.columns if not c.endswith('_raw')]})")

        device_problems = []

        # Detector 1: Moisture intrusion (A4: AND logic)
        probs = detect_moisture_intrusion(df, device_id)
        device_problems.extend(probs)
        if probs:
            print(f"  MOISTURE INTRUSION:      {len(probs)} episodes")

        # Detector 2: Condensation risk (A1+A2: seasonal-aware, chronic merge)
        probs = detect_condensation_risk(df, device_id, fleet_seasonal=fleet_seasonal)
        device_problems.extend(probs)
        if probs:
            chronic_count = sum(1 for p in probs if p.details.get("chronic"))
            print(f"  CONDENSATION RISK:       {len(probs)} episodes ({chronic_count} chronic)")

        # Detector 3: Drying failure
        probs = detect_drying_failure(df, device_id)
        device_problems.extend(probs)
        if probs:
            print(f"  DRYING FAILURE:          {len(probs)} episodes")

        # Detector 4: Sensor malfunction (A3: saturation-aware)
        probs = detect_sensor_malfunction(df, device_id)
        device_problems.extend(probs)
        real_malfunctions = [p for p in probs if p.severity != OK]
        sat_count = sum(1 for p in probs if p.details.get("subtype") == "sensor_saturation")
        if real_malfunctions:
            print(f"  SENSOR MALFUNCTION:      {len(real_malfunctions)} episodes" +
                  (f" (+{sat_count} saturation)" if sat_count else ""))
        elif sat_count:
            print(f"  SENSOR SATURATION:       {sat_count} (informational)")

        # Detector 5: Rapid moisture change (A5)
        probs = detect_rapid_moisture_change(df, device_id)
        device_problems.extend(probs)
        if probs:
            print(f"  RAPID MOISTURE CHANGE:   {len(probs)} episodes")

        if not device_problems:
            print("  OK - no problems detected")

        all_problems[device_id] = device_problems

        # C11: Health score
        data_span_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
        hs = compute_health_score(device_problems, data_span_hours)
        health_scores[device_id] = hs
        print(f"  Health Score: {hs['score']}/100 (Grade {hs['grade']})")

        # Visualize
        real_problems = [p for p in device_problems if p.severity != OK]
        if real_problems:
            path = visualize_device_problems(
                df, device_problems, device_id, output_dir,
                fleet_seasonal=fleet_seasonal,
                health_score=hs,
            )
            print(f"  -> {path}")

        # Summary row
        for p in device_problems:
            if p.severity == OK:
                continue
            summary_rows.append({
                "device_id": device_id,
                "installation_id": df.attrs.get("installation_id", "unknown"),
                "problem_type": p.problem_type,
                "severity": p.severity,
                "start": p.start,
                "end": p.end,
                "duration_hours": p.duration_hours,
                "is_chronic": p.is_chronic,
                "description": p.description,
                "health_score": hs["score"],
                "health_grade": hs["grade"],
            })

    # C12: Inter-device comparison
    print("\n" + "=" * 70)
    print("INTER-DEVICE ANALYSIS (C12)")
    print("=" * 70)

    installations: dict[str, dict[str, list[Problem]]] = {}
    for did, probs in all_problems.items():
        inst_id = devices[did].attrs.get("installation_id", "unknown")
        installations.setdefault(inst_id, {})[did] = probs

    outlier_problems = detect_installation_outliers(devices, all_problems, installations)
    for did, extra_probs in outlier_problems.items():
        all_problems[did].extend(extra_probs)
        for p in extra_probs:
            print(f"  {p}")
            summary_rows.append({
                "device_id": did,
                "installation_id": devices[did].attrs.get("installation_id", "unknown"),
                "problem_type": p.problem_type,
                "severity": p.severity,
                "start": p.start,
                "end": p.end,
                "duration_hours": p.duration_hours,
                "is_chronic": p.is_chronic,
                "description": p.description,
                "health_score": health_scores.get(did, {}).get("score", ""),
                "health_grade": health_scores.get(did, {}).get("grade", ""),
            })

    # Export CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "problems_report.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"\nProblems report saved: {csv_path}")
    else:
        summary_df = pd.DataFrame()

    # Health scores CSV
    if health_scores:
        hs_rows = []
        for did, hs in health_scores.items():
            hs_rows.append({
                "device_id": did,
                "installation_id": devices[did].attrs.get("installation_id", "unknown"),
                "health_score": hs["score"],
                "health_grade": hs["grade"],
            })
        hs_df = pd.DataFrame(hs_rows).sort_values("health_score")
        hs_csv = os.path.join(output_dir, "health_scores.csv")
        hs_df.to_csv(hs_csv, index=False)
        print(f"Health scores saved: {hs_csv}")

    # Installation summaries
    print("\n" + "=" * 70)
    print("INSTALLATION SUMMARIES")
    print("=" * 70)

    for inst_id, inst_problems in installations.items():
        total = sum(len([p for p in probs if p.severity != OK]) for probs in inst_problems.values())
        critical = sum(1 for probs in inst_problems.values() for p in probs if p.severity == CRITICAL)
        danger = sum(1 for probs in inst_problems.values() for p in probs if p.severity == DANGER)
        warning = sum(1 for probs in inst_problems.values() for p in probs if p.severity == WARNING)

        # Installation average health score
        inst_scores = [health_scores[d]["score"] for d in inst_problems if d in health_scores]
        avg_score = np.mean(inst_scores) if inst_scores else 100

        print(f"\n  Installation {inst_id[:16]}... ({len(inst_problems)} devices, avg health: {avg_score:.0f}/100)")
        print(f"    Total problems: {total} (CRITICAL: {critical}, DANGER: {danger}, WARNING: {warning})")

        if total > 0:
            path = visualize_installation_summary(
                inst_problems, inst_id, output_dir,
                health_scores=health_scores,
            )
            if path:
                print(f"    -> {path}")

        # B10: Installation comparison plot
        inst_dids = list(inst_problems.keys())
        if len(inst_dids) >= 2:
            path = visualize_installation_comparison(
                devices, inst_dids, inst_id, output_dir,
                health_scores=health_scores,
            )
            if path:
                print(f"    -> {path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (v6)")
    print("=" * 70)
    all_p = [p for probs in all_problems.values() for p in probs if p.severity != OK]
    print(f"  Devices analyzed:    {len(devices)}")
    print(f"  Total problems:      {len(all_p)}")
    print(f"    CRITICAL:          {sum(1 for p in all_p if p.severity == CRITICAL)}")
    print(f"    DANGER:            {sum(1 for p in all_p if p.severity == DANGER)}")
    print(f"    WARNING:           {sum(1 for p in all_p if p.severity == WARNING)}")
    print(f"  Devices with issues: {sum(1 for probs in all_problems.values() if any(p.severity != OK for p in probs))}")
    print(f"  Clean devices:       {sum(1 for probs in all_problems.values() if all(p.severity == OK for p in probs) or not probs)}")
    print(f"  Results in:          {output_dir}/")

    # Health score distribution
    scores = [hs["score"] for hs in health_scores.values()]
    if scores:
        print(f"\n  HEALTH SCORE DISTRIBUTION:")
        print(f"    Mean:    {np.mean(scores):.0f}/100")
        print(f"    Median:  {np.median(scores):.0f}/100")
        print(f"    Min:     {np.min(scores):.0f}/100")
        print(f"    Grade A: {sum(1 for s in scores if s >= 90)}")
        print(f"    Grade B: {sum(1 for s in scores if 75 <= s < 90)}")
        print(f"    Grade C: {sum(1 for s in scores if 50 <= s < 75)}")
        print(f"    Grade D: {sum(1 for s in scores if 25 <= s < 50)}")
        print(f"    Grade F: {sum(1 for s in scores if s < 25)}")

    # Print worst problems
    worst = sorted(all_p, key=lambda p: {CRITICAL: 3, DANGER: 2, WARNING: 1}.get(p.severity, 0), reverse=True)[:20]
    if worst:
        print(f"\n  TOP PROBLEMS:")
        for p in worst:
            print(f"    {p}")

    # Print worst health scores
    worst_devices = sorted(health_scores.items(), key=lambda x: x[1]["score"])[:10]
    if worst_devices:
        print(f"\n  WORST DEVICES (by health score):")
        for did, hs in worst_devices:
            inst_id = devices[did].attrs.get("installation_id", "unknown")[:12]
            print(f"    {did[:12]} | {inst_id} | Score: {hs['score']}/100 (Grade {hs['grade']})")

    return all_problems


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Senzomatic Building Problem Detection v6")
    parser.add_argument(
        "--data-dir",
        default=r"G:\My Drive\Rozinet\RMind\Clients\Senzomatic\Data_sensors\exported_data_2026-01-22\exported_data",
        help="Path to exported_data directory",
    )
    parser.add_argument("--output-dir", default="results_v6", help="Output directory")
    parser.add_argument("--max-devices", type=int, default=None, help="Limit devices (for testing)")
    parser.add_argument("--quarter", type=str, default=None, help="Analyze specific quarter, e.g. 2025-Q3")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_devices=args.max_devices,
        report_quarter=args.quarter,
    )
