"""
Building Problem Detection for Senzomatic Sensor Data.

Detects specific building pathologies from HT sensor readings:
  1. MOISTURE INTRUSION  - water entering construction (moisture_resistance drop, cavity humidity spike)
  2. CONDENSATION RISK   - cavity humidity sustained >80%, dew point proximity
  3. DRYING FAILURE      - post-construction moisture not decreasing as expected
  4. SENSOR MALFUNCTION  - flatlines, impossible jumps, out-of-range values

Sensors per device (HT02):
  - temperature_ambient_celsius   (ambient temperature at sensor)
  - rel_humidity_ambient_pct      (ambient relative humidity)
  - rel_humidity_cavity_pct       (relative humidity inside wall cavity)
  - moisture_resistance_pct       (material moisture resistance; higher = drier)

Aligned with Senzomatic quarterly report types:
  - "Pravidelny ctvrtletni report"  (routine monitoring)
  - "Narust vlhkosti v konstrukci"  (acute moisture alert)
  - "Report po aktivaci systemu"    (post-activation baseline)
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

    # --- Preprocessing ---
    "median_filter_window": 7,              # median filter kernel size (samples); removes short spikes
    "smoothing_window": 6,                  # rolling mean window (samples, ~30min); smooths remaining noise
    "min_data_hours": 48,                   # skip devices with less data than this

    # --- Moisture intrusion ---
    # A drop in moisture_resistance over 24h that exceeds this pct-points = alert
    "moisture_drop_threshold_24h": 3.0,     # pct-points drop in 24h (was 2.0)
    "moisture_drop_threshold_7d": 5.0,      # pct-points drop in 7 days
    # Cavity humidity rise that co-occurs with moisture drop
    "cavity_rise_threshold_24h": 8.0,       # pct-points rise in 24h (was 5.0)
    # Minimum duration (hours) before flagging as intrusion (not transient)
    "moisture_intrusion_min_hours": 24,     # at least 1 day (was 6)

    # --- Condensation risk ---
    "condensation_warning_pct": 80.0,       # cavity hum % -> mold risk
    "condensation_danger_pct": 90.0,        # cavity hum % -> condensation likely
    "condensation_critical_pct": 95.0,      # cavity hum % -> active condensation
    # Minimum sustained duration (hours) above threshold to flag
    "condensation_min_hours": 48,           # at least 2 days (was 12)

    # --- Drying failure ---
    # Weekly rolling average; if moisture trend reverses or plateaus for this many weeks
    "drying_eval_window_weeks": 4,          # evaluate trend over this window
    "drying_plateau_tolerance": 0.5,        # pct-points change below this = plateau
    "drying_reversal_threshold": 1.0,       # moisture_resistance drops by this = reversal

    # --- Sensor malfunction ---
    "flatline_window_hours": 24,            # zero variance for this long = flatline (was 6)
    "jump_threshold_temp": 10.0,            # deg C jump in 5 min (was 5.0)
    "jump_threshold_humidity": 25.0,        # pct-points jump in 5 min (was 15.0)
    "jump_threshold_moisture": 20.0,        # pct-points jump in 5 min (was 10.0)
    "jump_min_count": 3,                    # require at least this many jumps in a cluster
    "temp_range": (-40.0, 60.0),            # physically possible range
    "humidity_range": (0.0, 100.0),
    "moisture_range": (0.0, 100.0),

    # --- Episode merging ---
    "episode_merge_gap_hours": 6,           # merge episodes separated by less than this
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
    problem_type: str       # moisture_intrusion | condensation | drying_failure | sensor_malfunction
    severity: str           # ok | warning | danger | critical
    device_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    description: str
    details: dict = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600

    def __str__(self):
        dur = f"{self.duration_hours:.0f}h"
        return f"[{self.severity.upper():8s}] {self.problem_type:20s} | {self.device_id[:12]} | {self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d} ({dur}) | {self.description}"


# ---------------------------------------------------------------------------
# Step 1: Data Loading (reused from previous version)
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
    Preprocess sensor data to remove noise and short spikes.

    1. Median filter: removes isolated spikes (salt-and-pepper noise from sensor glitches)
       without blurring real step changes. Window of 7 samples = 35 min.
    2. Light smoothing: rolling mean to reduce remaining high-frequency noise.
       Window of 6 samples = 30 min.
    3. Forward-fill small gaps (up to 30 min = 6 samples).

    The original raw data is preserved in columns with '_raw' suffix for
    sensor malfunction detection (which needs to see the actual spikes).
    """
    from scipy.ndimage import median_filter

    median_k = CONFIG["median_filter_window"]
    smooth_k = CONFIG["smoothing_window"]

    processed = df.copy()

    for col in COLUMN_NAMES:
        if col not in processed.columns:
            continue
        raw = processed[col].values.copy()

        # Save raw for sensor malfunction detector
        processed[f"{col}_raw"] = raw

        # Step 1: Median filter (ignoring NaNs by filling temporarily)
        mask = np.isnan(raw)
        if mask.all():
            continue
        filled = pd.Series(raw).ffill().bfill().values
        filtered = median_filter(filled, size=median_k)
        filtered[mask] = np.nan  # restore NaN positions

        # Step 2: Light rolling mean smoothing
        smoothed = pd.Series(filtered, index=df.index).rolling(
            smooth_k, min_periods=1, center=True
        ).mean()

        processed[col] = smoothed

    return processed


# ---------------------------------------------------------------------------
# Detector 1: MOISTURE INTRUSION
# ---------------------------------------------------------------------------

def detect_moisture_intrusion(df: pd.DataFrame, device_id: str) -> list[Problem]:
    """
    Detect water entering construction.

    Physics: when water infiltrates, moisture_resistance DROPS (lower resistance = wetter)
    and cavity humidity RISES. We look for:
      - Rapid moisture_resistance decrease (24h and 7d windows)
      - Concurrent cavity humidity increase
      - Sustained for >24h (not transient rain splash)
    """
    problems = []

    has_moisture = "moisture" in df.columns
    has_cavity = "hum_cavity" in df.columns

    if not has_moisture and not has_cavity:
        return problems

    # 24h rolling change (288 samples at 5-min)
    w24h = 288
    w7d = 288 * 7

    if has_moisture:
        moisture_delta_24h = df["moisture"].diff(w24h)  # negative = dropping
        moisture_delta_7d = df["moisture"].diff(min(w7d, len(df) - 1)) if len(df) > w7d else moisture_delta_24h
    else:
        moisture_delta_24h = pd.Series(0, index=df.index)
        moisture_delta_7d = pd.Series(0, index=df.index)

    if has_cavity:
        cavity_delta_24h = df["hum_cavity"].diff(w24h)  # positive = rising
    else:
        cavity_delta_24h = pd.Series(0, index=df.index)

    # Flag: moisture dropping OR cavity rising significantly
    moisture_dropping = moisture_delta_24h < -CONFIG["moisture_drop_threshold_24h"]
    moisture_dropping_7d = moisture_delta_7d < -CONFIG["moisture_drop_threshold_7d"]
    cavity_rising = cavity_delta_24h > CONFIG["cavity_rise_threshold_24h"]

    # Combined signal: moisture drop (short or long term) AND/OR cavity rise
    if has_moisture and has_cavity:
        flag = (moisture_dropping & cavity_rising) | moisture_dropping_7d
    elif has_moisture:
        flag = moisture_dropping | moisture_dropping_7d
    else:
        flag = cavity_rising

    # Find contiguous flagged regions and filter by minimum duration
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
# Detector 2: CONDENSATION RISK
# ---------------------------------------------------------------------------

def detect_condensation_risk(df: pd.DataFrame, device_id: str) -> list[Problem]:
    """
    Detect sustained high cavity humidity indicating condensation/mold risk.

    Produces max ~1 episode per quarter per severity level by using a large
    merge gap (7 days). Chronic devices get a few long episodes instead of many short ones.

    Thresholds (building science):
      - >80% RH sustained -> mold growth conditions (WARNING)
      - >90% RH sustained -> condensation likely (DANGER)
      - >95% RH sustained -> active condensation (CRITICAL)
    """
    problems = []

    if "hum_cavity" not in df.columns:
        return problems

    hum = df["hum_cavity"]

    for threshold, severity in [
        (CONFIG["condensation_critical_pct"], CRITICAL),
        (CONFIG["condensation_danger_pct"], DANGER),
        (CONFIG["condensation_warning_pct"], WARNING),
    ]:
        flag = hum > threshold

        episodes = _extract_episodes(
            flag, df, device_id,
            problem_type="condensation_risk",
            min_hours=CONFIG["condensation_min_hours"],
            merge_gap_hours=24 * 7,  # 7-day merge gap: bridges weekly oscillations
            detail_fn=lambda sub_df: {
                "max_cavity_humidity": float(sub_df["hum_cavity"].max()),
                "mean_cavity_humidity": float(sub_df["hum_cavity"].mean()),
                "threshold": threshold,
                "dew_point_proximity": _dew_point_margin(sub_df) if "temp" in sub_df.columns else None,
            },
            severity_fn=lambda sub_df: severity,
            desc_fn=lambda sub_df, details: (
                f"Cavity humidity sustained at {details['mean_cavity_humidity']:.1f}% "
                f"(peak {details['max_cavity_humidity']:.1f}%) for >{CONFIG['condensation_min_hours']}h"
                + (f" - {severity}" if severity != WARNING else "")
            ),
        )
        problems.extend(episodes)

    # Deduplicate: keep highest severity for overlapping periods
    problems = _deduplicate_problems(problems)
    return problems


def _dew_point_margin(df: pd.DataFrame) -> float | None:
    """Compute how close cavity conditions are to dew point (Magnus formula)."""
    if "temp" not in df.columns or "hum_cavity" not in df.columns:
        return None
    t = df["temp"].mean()
    rh = df["hum_cavity"].mean() / 100.0
    if rh <= 0:
        return None
    # Magnus formula for dew point
    a, b = 17.27, 237.7
    alpha = (a * t) / (b + t) + np.log(rh)
    dew_point = (b * alpha) / (a - alpha)
    return float(t - dew_point)  # margin above dew point; <0 means condensing


# ---------------------------------------------------------------------------
# Detector 3: DRYING FAILURE
# ---------------------------------------------------------------------------

def detect_drying_failure(df: pd.DataFrame, device_id: str) -> list[Problem]:
    """
    Detect when construction is not drying as expected.

    Produces a SINGLE chronic assessment per device by analyzing the full data
    period rather than many fragmented episodes. Compares year-over-year trend
    (aligned with Senzomatic quarterly reports).

    Output: max 1-2 problems per device (one for moisture, one for cavity humidity).
    """
    problems = []

    has_moisture = "moisture" in df.columns
    has_cavity = "hum_cavity" in df.columns

    if not has_moisture and not has_cavity:
        return problems

    # Need at least 3 months of data for meaningful trend
    data_span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    if data_span_days < 90:
        return problems

    # --- Moisture resistance chronic assessment ---
    if has_moisture:
        moisture = df["moisture"].dropna()
        if len(moisture) > 288 * 30:  # at least 1 month
            # Compute quarterly averages
            quarterly = moisture.resample("QS").agg(["mean", "std", "min", "max"])

            # Overall trend: linear regression over quarterly means
            q_means = quarterly["mean"].dropna()
            if len(q_means) >= 2:
                x = np.arange(len(q_means))
                slope = np.polyfit(x, q_means.values, 1)[0]  # pct-points per quarter
                current_avg = float(moisture.iloc[-288*30:].mean())  # last month avg
                overall_avg = float(moisture.mean())

                # Classify: is the device drying (improving) or not?
                is_wet = current_avg < 50  # moisture resistance below 50% = wet
                is_worsening = slope < -0.5  # losing >0.5 pct per quarter
                is_stagnant = abs(slope) < 0.3 and is_wet
                is_improving = slope > 0.5

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

    # --- Cavity humidity chronic assessment ---
    if has_cavity:
        hum = df["hum_cavity"].dropna()
        if len(hum) > 288 * 30:
            quarterly_hum = hum.resample("QS").agg(["mean", "std", "min", "max"])
            q_hum_means = quarterly_hum["mean"].dropna()

            if len(q_hum_means) >= 2:
                x = np.arange(len(q_hum_means))
                hum_slope = np.polyfit(x, q_hum_means.values, 1)[0]
                current_hum = float(hum.iloc[-288*30:].mean())

                # Flag if humidity is high AND rising
                is_high = current_hum > 75
                is_rising = hum_slope > 0.5  # rising >0.5 pct per quarter

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
# Detector 4: SENSOR MALFUNCTION
# ---------------------------------------------------------------------------

def detect_sensor_malfunction(df: pd.DataFrame, device_id: str) -> list[Problem]:
    """
    Detect sensor hardware issues using RAW (unsmoothed) data:
      - Flatline: zero variance for extended period (sensor stuck)
      - Jumps: physically impossible rate of change, requires cluster of jumps
      - Out-of-range: values outside physical bounds, sustained

    Caps total episodes at 10 per device. If a sensor has chronic issues,
    reports a summary instead of hundreds of episodes.
    """
    problems = []

    flatline_window = int(CONFIG["flatline_window_hours"] * 12)  # samples
    min_jump_count = CONFIG.get("jump_min_count", 3)

    for col, label, jump_thresh, valid_range in [
        ("temp", "Temperature", CONFIG["jump_threshold_temp"], CONFIG["temp_range"]),
        ("hum_ambient", "Ambient humidity", CONFIG["jump_threshold_humidity"], CONFIG["humidity_range"]),
        ("hum_cavity", "Cavity humidity", CONFIG["jump_threshold_humidity"], CONFIG["humidity_range"]),
        ("moisture", "Moisture resistance", CONFIG["jump_threshold_moisture"], CONFIG["moisture_range"]),
    ]:
        if col not in df.columns:
            continue

        # Use raw data for malfunction detection (not smoothed)
        raw_col = f"{col}_raw"
        series = df[raw_col] if raw_col in df.columns else df[col]

        # --- Flatline detection (use raw data to catch stuck sensors) ---
        rolling_std = series.rolling(flatline_window, min_periods=flatline_window // 2).std()
        flatline = rolling_std < 1e-6  # effectively zero variance

        problems.extend(_extract_episodes(
            flatline, df, device_id,
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
            merge_gap_hours=48,  # merge flatlines within 2 days (was 1h)
        ))

        # --- Jump detection (require cluster of jumps, not isolated) ---
        delta = series.diff().abs()
        jumps = delta > jump_thresh

        # Only flag if there are enough jumps in a 1-hour window (12 samples)
        jump_density = jumps.astype(int).rolling(12, min_periods=1).sum()
        jump_cluster = jump_density >= min_jump_count

        if jump_cluster.any():
            problems.extend(_extract_episodes(
                jump_cluster, df, device_id,
                problem_type="sensor_malfunction",
                min_hours=0,  # clusters are already filtered by count
                detail_fn=lambda sub_df, c=col, rc=raw_col: {
                    "subtype": "jump",
                    "channel": c,
                    "max_jump": float(delta.loc[sub_df.index].max()),
                    "jump_count": int(jumps.loc[sub_df.index].sum()),
                },
                severity_fn=lambda sub_df: DANGER,
                desc_fn=lambda sub_df, details: (
                    f"JUMP CLUSTER: {label} had {details.get('jump_count', 0)} jumps "
                    f"(max {details.get('max_jump', 0):.1f}) in {(sub_df.index[-1] - sub_df.index[0]).total_seconds() / 3600:.1f}h"
                ),
                merge_gap_hours=24,  # merge jump clusters within 1 day (was 2h)
            ))

        # --- Out-of-range detection (require sustained, not single sample) ---
        oor = (series < valid_range[0]) | (series > valid_range[1])

        problems.extend(_extract_episodes(
            oor, df, device_id,
            problem_type="sensor_malfunction",
            min_hours=1,  # at least 1 hour of out-of-range
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

    # Cap at 10 episodes per device. If more, consolidate into a chronic summary.
    if len(problems) > 10:
        total_count = len(problems)
        severity_order = {CRITICAL: 3, DANGER: 2, WARNING: 1, OK: 0}
        max_severity = max(problems, key=lambda p: severity_order.get(p.severity, 0)).severity
        earliest = min(p.start for p in problems)
        latest = max(p.end for p in problems)
        total_hours = sum(p.duration_hours for p in problems)

        # Keep top 5 most severe, replace rest with a summary
        problems.sort(key=lambda p: (severity_order.get(p.severity, 0), p.duration_hours), reverse=True)
        kept = problems[:5]
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
        problems = kept

    return problems


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
    """
    Find contiguous True regions in flag, merge nearby ones, create Problem for each.

    Args:
        merge_gap_hours: If two True regions are separated by less than this many hours,
            merge them into one episode. Defaults to CONFIG["episode_merge_gap_hours"].
    """
    problems = []
    if flag.sum() == 0:
        return problems

    if merge_gap_hours is None:
        merge_gap_hours = CONFIG["episode_merge_gap_hours"]

    # Find contiguous groups of True
    flag = flag.reindex(df.index).fillna(False)
    groups = (flag != flag.shift()).cumsum()
    flagged_groups = groups[flag]

    # Collect raw intervals first
    intervals = []
    for _, grp in flagged_groups.groupby(flagged_groups):
        intervals.append((grp.index[0], grp.index[-1]))

    if not intervals:
        return problems

    # Merge nearby intervals (bridge small gaps)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        gap_hours = (start - prev_end).total_seconds() / 3600
        if gap_hours <= merge_gap_hours:
            # Merge with previous interval
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Create problems from merged intervals
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
# Visualization
# ---------------------------------------------------------------------------

def visualize_device_problems(
    df: pd.DataFrame,
    problems: list[Problem],
    device_id: str,
    output_dir: str = ".",
) -> str:
    """Plot device time series with problem periods highlighted."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    available = [c for c in COLUMN_NAMES if c in df.columns]
    n_panels = len(available) + 1  # +1 for problem timeline
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
        WARNING: "#FFD700",   # gold
        DANGER: "#FF8C00",    # dark orange
        CRITICAL: "#DC143C",  # crimson
    }

    # Separate acute problems (short) from chronic (>90 days) for shading
    acute_problems = [p for p in problems if p.duration_hours < 90 * 24]

    # Plot each channel
    for ax, col in zip(axes, available):
        ax.plot(df.index, df[col], linewidth=0.4, color="#333", alpha=0.8)
        ax.set_ylabel(labels.get(col, col), fontsize=9)
        ax.grid(True, alpha=0.2)

        # Add threshold lines for cavity humidity
        if col == "hum_cavity":
            ax.axhline(80, color=severity_colors[WARNING], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(90, color=severity_colors[DANGER], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(95, color=severity_colors[CRITICAL], linestyle="--", linewidth=0.8, alpha=0.7)

        # Only shade ACUTE problem periods on data panels (not chronic ones that cover everything)
        for p in acute_problems:
            color = severity_colors.get(p.severity, "#999")
            ax.axvspan(p.start, p.end, alpha=0.15, color=color, zorder=0)

    # Problem timeline panel
    ax_timeline = axes[-1]
    type_y = {"moisture_intrusion": 3, "condensation_risk": 2, "drying_failure": 1, "sensor_malfunction": 0}
    type_labels = {"moisture_intrusion": "Moisture Intrusion", "condensation_risk": "Condensation",
                   "drying_failure": "Drying Failure", "sensor_malfunction": "Sensor Malfunction"}

    for p in problems:
        y = type_y.get(p.problem_type, 0)
        color = severity_colors.get(p.severity, "#999")
        ax_timeline.barh(y, (p.end - p.start).total_seconds() / 86400,
                         left=p.start, height=0.6, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)

    ax_timeline.set_yticks(list(type_y.values()))
    ax_timeline.set_yticklabels(list(type_labels.values()), fontsize=8)
    ax_timeline.set_ylim(-0.5, 3.5)
    ax_timeline.grid(True, axis="x", alpha=0.2)
    ax_timeline.set_ylabel("Problems", fontsize=9)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=s.upper(), alpha=0.7) for s, c in severity_colors.items()]
    ax_timeline.legend(handles=legend_patches, loc="upper right", fontsize=7)

    fig.suptitle(f"Building Health - Device {device_id[:12]}...", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"problems_{device_id[:12]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def visualize_installation_summary(
    all_problems: dict[str, list[Problem]],
    installation_id: str,
    output_dir: str = ".",
) -> str:
    """Heatmap showing problem severity per device over time."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    severity_val = {OK: 0, WARNING: 1, DANGER: 2, CRITICAL: 3}
    severity_colors_map = {0: "#E8F5E9", 1: "#FFD700", 2: "#FF8C00", 3: "#DC143C"}

    device_ids = sorted(all_problems.keys())
    if not device_ids:
        return ""

    # Collect all problems into monthly severity per device
    all_p = []
    for did, probs in all_problems.items():
        for p in probs:
            all_p.append(p)

    if not all_p:
        return ""

    # Determine time range
    min_date = min(p.start for p in all_p)
    max_date = max(p.end for p in all_p)
    months = pd.date_range(min_date.to_period("M").to_timestamp(), max_date, freq="MS")

    # Build matrix: max severity per device per month
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

    ax.set_yticks(range(len(device_ids)))
    ax.set_yticklabels([d[:12] for d in device_ids], fontsize=7)
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
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: str,
    output_dir: str = "results",
    max_devices: int | None = None,
    report_quarter: str | None = None,
):
    """
    Run building problem detection pipeline.

    Args:
        data_dir: Path to exported_data directory
        output_dir: Directory for output CSVs and plots
        max_devices: Limit number of devices (for testing)
        report_quarter: If set (e.g. "2025-Q3"), only analyze the 3 months of that quarter
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SENZOMATIC BUILDING PROBLEM DETECTION")
    print("=" * 70)

    # Load data
    print("\nLoading sensor data...")
    devices = load_device_data(data_dir)

    if max_devices:
        device_ids = list(devices.keys())[:max_devices]
        devices = {k: devices[k] for k in device_ids}
        print(f"  Limited to {max_devices} devices for testing")

    # Optionally filter to a quarter
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

    # Run all detectors on each device
    all_problems: dict[str, list[Problem]] = {}
    summary_rows = []

    # Preprocess all devices (noise removal)
    print("\nPreprocessing: median filter + smoothing...")
    for device_id in list(devices.keys()):
        df = devices[device_id]
        # Skip devices with too little data
        if len(df) < CONFIG["min_data_hours"] * 12:
            print(f"  Skipping {device_id[:12]}... (only {len(df)} samples, <{CONFIG['min_data_hours']}h)")
            del devices[device_id]
            continue
        devices[device_id] = preprocess_device_data(df)
    print(f"  {len(devices)} devices after preprocessing")

    for i, (device_id, df) in enumerate(devices.items()):
        print(f"\n[{i+1}/{len(devices)}] Device {device_id[:12]}... ({len(df)} samples, channels: {[c for c in df.columns if not c.endswith('_raw')]})")

        device_problems = []

        # Detector 1: Moisture intrusion
        probs = detect_moisture_intrusion(df, device_id)
        device_problems.extend(probs)
        if probs:
            print(f"  MOISTURE INTRUSION: {len(probs)} episodes")

        # Detector 2: Condensation risk
        probs = detect_condensation_risk(df, device_id)
        device_problems.extend(probs)
        if probs:
            print(f"  CONDENSATION RISK:  {len(probs)} episodes")

        # Detector 3: Drying failure
        probs = detect_drying_failure(df, device_id)
        device_problems.extend(probs)
        if probs:
            print(f"  DRYING FAILURE:     {len(probs)} episodes")

        # Detector 4: Sensor malfunction
        probs = detect_sensor_malfunction(df, device_id)
        device_problems.extend(probs)
        if probs:
            print(f"  SENSOR MALFUNCTION: {len(probs)} episodes")

        if not device_problems:
            print("  OK - no problems detected")

        all_problems[device_id] = device_problems

        # Visualize
        if device_problems:
            path = visualize_device_problems(df, device_problems, device_id, output_dir)
            print(f"  -> {path}")

        # Summary row
        for p in device_problems:
            summary_rows.append({
                "device_id": device_id,
                "installation_id": df.attrs.get("installation_id", "unknown"),
                "problem_type": p.problem_type,
                "severity": p.severity,
                "start": p.start,
                "end": p.end,
                "duration_hours": p.duration_hours,
                "description": p.description,
            })

    # Export CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "problems_report.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"\nProblems report saved: {csv_path}")
    else:
        summary_df = pd.DataFrame()

    # Installation summaries
    print("\n" + "=" * 70)
    print("INSTALLATION SUMMARIES")
    print("=" * 70)

    installations: dict[str, dict[str, list[Problem]]] = {}
    for did, probs in all_problems.items():
        inst_id = devices[did].attrs.get("installation_id", "unknown")
        installations.setdefault(inst_id, {})[did] = probs

    for inst_id, inst_problems in installations.items():
        total = sum(len(p) for p in inst_problems.values())
        critical = sum(1 for probs in inst_problems.values() for p in probs if p.severity == CRITICAL)
        danger = sum(1 for probs in inst_problems.values() for p in probs if p.severity == DANGER)
        warning = sum(1 for probs in inst_problems.values() for p in probs if p.severity == WARNING)

        print(f"\n  Installation {inst_id[:16]}... ({len(inst_problems)} devices)")
        print(f"    Total problems: {total} (CRITICAL: {critical}, DANGER: {danger}, WARNING: {warning})")

        if total > 0:
            path = visualize_installation_summary(inst_problems, inst_id, output_dir)
            if path:
                print(f"    -> {path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    all_p = [p for probs in all_problems.values() for p in probs]
    print(f"  Devices analyzed:    {len(devices)}")
    print(f"  Total problems:      {len(all_p)}")
    print(f"    CRITICAL:          {sum(1 for p in all_p if p.severity == CRITICAL)}")
    print(f"    DANGER:            {sum(1 for p in all_p if p.severity == DANGER)}")
    print(f"    WARNING:           {sum(1 for p in all_p if p.severity == WARNING)}")
    print(f"  Devices with issues: {sum(1 for probs in all_problems.values() if probs)}")
    print(f"  Clean devices:       {sum(1 for probs in all_problems.values() if not probs)}")
    print(f"  Results in:          {output_dir}/")

    # Print worst problems
    worst = sorted(all_p, key=lambda p: {CRITICAL: 3, DANGER: 2, WARNING: 1}.get(p.severity, 0), reverse=True)[:20]
    if worst:
        print(f"\n  TOP PROBLEMS:")
        for p in worst:
            print(f"    {p}")

    return all_problems


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Senzomatic Building Problem Detection")
    parser.add_argument(
        "--data-dir",
        default=r"G:\My Drive\Rozinet\RMind\Clients\Senzomatic\Data_sensors\exported_data_2026-01-22\exported_data",
        help="Path to exported_data directory",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-devices", type=int, default=None, help="Limit devices (for testing)")
    parser.add_argument("--quarter", type=str, default=None, help="Analyze specific quarter, e.g. 2025-Q3")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_devices=args.max_devices,
        report_quarter=args.quarter,
    )
