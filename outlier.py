"""
Multidimensional Local Outlier Detection for Senzomatic Sensor Data.

3-Layer Pipeline:
  1. Windowed Multivariate LOF (Local Outlier Factor)
  2. Rolling Mahalanobis Distance
  3. Cross-Device Consensus

Each device has 4 channels: temperature, ambient humidity, cavity humidity, moisture resistance.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ─── Configuration ───────────────────────────────────────────────────────────

CONFIG = {
    "rolling_windows": [12, 72, 288],       # 1h, 6h, 24h at 5-min sampling
    "lof_window_size": 288,                  # 24h sliding window for LOF
    "lof_n_neighbors": 20,
    "lof_stride": 12,                        # 1h stride
    "mahal_window_size": 288,                # 24h window for covariance
    "cross_device_mad_threshold": 3.0,
    "ensemble_weights": [0.5, 0.3, 0.2],    # LOF, Mahalanobis, CrossDevice
    "outlier_threshold": 0.7,
    "resample_interval": "5min",             # align all channels to 5-min grid
}

SENSOR_TYPES = [
    "temperature_ambient_celsius",
    "rel_humidity_ambient_pct",
    "rel_humidity_cavity_pct",
    "moisture_resistance_pct",
]

COLUMN_NAMES = ["temp", "hum_ambient", "hum_cavity", "moisture"]


# ─── Step 1: Data Loading & Alignment ────────────────────────────────────────

def load_single_channel(filepath: str) -> pd.Series | None:
    """Load a single sensor channel from a .json_line file. Returns None on failure."""
    try:
        with open(filepath, "r") as f:
            line = f.readline().strip()
            if not line:
                return None
            data = json.loads(line)

        timestamps = pd.to_datetime(data["timestamps"], unit="ms")
        values = np.array(data["values"], dtype=np.float64)

        series = pd.Series(values, index=timestamps, name=data["metric"]["__name__"])
        series = series[~series.index.duplicated(keep="first")].sort_index()
        return series
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  Warning: failed to load {filepath}: {e}")
        return None


def load_device_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Walk the export directory tree and load all devices.

    Returns:
        dict mapping device_id -> DataFrame with columns
        [temp, hum_ambient, hum_cavity, moisture] indexed by timestamp.
    """
    # Discover all .json_line files grouped by device
    device_files: dict[str, dict[str, str]] = {}
    installation_map: dict[str, str] = {}  # device_id -> installation_id

    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".json_line"):
                continue
            # Parse: {device_id}_{sensor_type}.json_line
            # device_id is a UUID (36 chars)
            device_id = fname[:36]
            sensor_type = fname[37:].replace(".json_line", "")

            if sensor_type not in SENSOR_TYPES:
                continue

            if device_id not in device_files:
                device_files[device_id] = {}
            device_files[device_id][sensor_type] = os.path.join(root, fname)

            # Extract installation_id from first file's metadata
            if device_id not in installation_map:
                try:
                    with open(os.path.join(root, fname), "r") as f:
                        meta = json.loads(f.readline())["metric"]
                        installation_map[device_id] = meta.get("installation_id", "unknown")
                except Exception:
                    installation_map[device_id] = "unknown"

    print(f"Found {len(device_files)} devices")

    devices = {}
    for device_id, files_map in device_files.items():
        if len(files_map) < 2:
            print(f"  Skipping {device_id[:12]}: only {len(files_map)}/4 channel files found")
            continue

        channels = {}
        for sensor_type, col_name in zip(SENSOR_TYPES, COLUMN_NAMES):
            if sensor_type in files_map:
                series = load_single_channel(files_map[sensor_type])
                if series is not None and len(series) > 0:
                    channels[col_name] = series

        if len(channels) < 2:
            print(f"  Skipping {device_id[:12]}: only {len(channels)}/4 channels loaded")
            continue

        # Align all channels on a common 5-min grid
        df = pd.DataFrame(channels)
        df = df.resample(CONFIG["resample_interval"]).mean()
        df = df.dropna(how="all")  # drop rows where all channels are NaN

        # Forward-fill small gaps (up to 30 min = 6 samples)
        df = df.ffill(limit=6)

        # Store installation info as attribute
        df.attrs["device_id"] = device_id
        df.attrs["installation_id"] = installation_map.get(device_id, "unknown")

        devices[device_id] = df

    print(f"Loaded {len(devices)} devices (2+ channels each)")
    return devices


# ─── Step 2: Feature Engineering ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a device DataFrame with rolling statistics and cross-channel features.

    Input columns: [temp, hum_ambient, hum_cavity, moisture]
    Output: original columns + delta + rolling stats + cross-channel ratios
    """
    result = df.copy()

    for col in COLUMN_NAMES:
        if col not in df.columns:
            continue
        # Rate of change (first difference)
        result[f"{col}_delta"] = df[col].diff()

        for win in CONFIG["rolling_windows"]:
            result[f"{col}_rmean_{win}"] = df[col].rolling(win, min_periods=1).mean()
            result[f"{col}_rstd_{win}"] = df[col].rolling(win, min_periods=1).std()

    # Cross-channel: deviation of cavity humidity from ambient humidity
    if "hum_cavity" in df.columns and "hum_ambient" in df.columns:
        result["hum_cavity_minus_ambient"] = df["hum_cavity"] - df["hum_ambient"]

    # Cross-channel: temp-humidity interaction (dew point proxy)
    if "temp" in df.columns and "hum_ambient" in df.columns:
        result["temp_hum_ratio"] = df["temp"] / df["hum_ambient"].replace(0, np.nan)

    result = result.dropna()
    return result


# ─── Step 3: Windowed LOF Scoring ────────────────────────────────────────────

def detect_lof_outliers(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.Series:
    """
    Run LOF with a sliding window over the multivariate feature space.

    Returns a Series of LOF scores (higher = more anomalous) aligned with df.index.
    """
    if feature_cols is None:
        feature_cols = COLUMN_NAMES  # use raw 4D channels

    data = df[feature_cols].values
    n = len(data)
    window = CONFIG["lof_window_size"]
    stride = CONFIG["lof_stride"]
    n_neighbors = min(CONFIG["lof_n_neighbors"], window // 2 - 1)

    # Accumulate scores (each point may appear in multiple windows)
    score_sum = np.zeros(n)
    score_count = np.zeros(n)

    for start in range(0, max(1, n - window + 1), stride):
        end = min(start + window, n)
        chunk = data[start:end]

        if len(chunk) < n_neighbors + 1:
            continue

        # Standardize within the window for fair distance computation
        scaler = StandardScaler()
        chunk_scaled = scaler.fit_transform(chunk)

        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(chunk) - 1),
            contamination="auto",
            novelty=False,
        )
        lof.fit(chunk_scaled)

        # negative_outlier_factor_ is negative; more negative = more outlier
        scores = -lof.negative_outlier_factor_

        score_sum[start:end] += scores
        score_count[start:end] += 1

    # Average across overlapping windows
    score_count[score_count == 0] = 1
    avg_scores = score_sum / score_count

    return pd.Series(avg_scores, index=df.index, name="lof_score")


# ─── Step 4: Rolling Mahalanobis Distance ────────────────────────────────────

def detect_mahalanobis_outliers(
    df: pd.DataFrame, feature_cols: list[str] | None = None
) -> pd.Series:
    """
    Compute rolling Mahalanobis distance using a shrunk covariance estimator.

    Returns a Series of Mahalanobis distances aligned with df.index.
    """
    if feature_cols is None:
        feature_cols = COLUMN_NAMES

    data = df[feature_cols].values
    n = len(data)
    window = CONFIG["mahal_window_size"]
    half_w = window // 2
    distances = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w)

        if end - start < len(feature_cols) + 2:
            continue

        local_data = data[start:end]
        point = data[i]

        try:
            # Ledoit-Wolf shrinkage for robust covariance estimation
            cov_estimator = LedoitWolf().fit(local_data)
            cov_matrix = cov_estimator.covariance_
            mean = local_data.mean(axis=0)

            cov_inv = np.linalg.pinv(cov_matrix)
            distances[i] = mahalanobis(point, mean, cov_inv)
        except Exception:
            distances[i] = np.nan

    return pd.Series(distances, index=df.index, name="mahal_distance")


def detect_mahalanobis_outliers_fast(
    df: pd.DataFrame, feature_cols: list[str] | None = None
) -> pd.Series:
    """
    Fast approximation: compute Mahalanobis in non-overlapping blocks,
    then interpolate for stride positions.
    """
    if feature_cols is None:
        feature_cols = COLUMN_NAMES

    data = df[feature_cols].values
    n = len(data)
    window = CONFIG["mahal_window_size"]
    stride = CONFIG["lof_stride"]
    distances = np.full(n, np.nan)

    for block_start in range(0, max(1, n - window + 1), stride):
        block_end = min(block_start + window, n)
        block = data[block_start:block_end]

        if len(block) < len(feature_cols) + 2:
            continue

        try:
            cov_estimator = LedoitWolf().fit(block)
            cov_matrix = cov_estimator.covariance_
            mean = block.mean(axis=0)
            cov_inv = np.linalg.pinv(cov_matrix)

            for i in range(block_start, block_end):
                d = mahalanobis(data[i], mean, cov_inv)
                if np.isnan(distances[i]):
                    distances[i] = d
                else:
                    distances[i] = (distances[i] + d) / 2  # average overlapping
        except Exception:
            continue

    return pd.Series(distances, index=df.index, name="mahal_distance")


# ─── Step 5: Cross-Device Consensus ──────────────────────────────────────────

def detect_cross_device_outliers(
    devices: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """
    Compare each device against its installation peers.

    For each timestamp, compute the installation median per channel.
    Flag devices that deviate more than `cross_device_mad_threshold` MADs
    from the peer median.

    Returns dict mapping device_id -> Series of cross-device anomaly scores.
    """
    threshold = CONFIG["cross_device_mad_threshold"]

    # Group devices by installation
    installations: dict[str, list[str]] = {}
    for device_id, df in devices.items():
        inst_id = df.attrs.get("installation_id", "unknown")
        installations.setdefault(inst_id, []).append(device_id)

    results = {}

    for inst_id, device_ids in installations.items():
        if len(device_ids) < 3:
            # Not enough peers for meaningful comparison
            for did in device_ids:
                results[did] = pd.Series(
                    0.0, index=devices[did].index, name="cross_device_score"
                )
            continue

        # Build a common time index for this installation
        common_idx = devices[device_ids[0]].index
        for did in device_ids[1:]:
            common_idx = common_idx.union(devices[did].index)

        # For each channel, compute median and MAD across devices
        for did in device_ids:
            df = devices[did].reindex(common_idx)
            scores = np.zeros(len(df))
            available_cols = [c for c in COLUMN_NAMES if c in devices[did].columns]

            for col in available_cols:
                if col not in df.columns:
                    continue

                # Stack all peer values at each timestamp
                peer_values = []
                for peer_id in device_ids:
                    if peer_id == did:
                        continue
                    peer_df = devices[peer_id].reindex(common_idx)
                    if col in peer_df.columns:
                        peer_values.append(peer_df[col].values)

                if not peer_values:
                    continue

                peer_matrix = np.array(peer_values)  # shape: (n_peers, n_timestamps)
                peer_median = np.nanmedian(peer_matrix, axis=0)
                peer_mad = np.nanmedian(np.abs(peer_matrix - peer_median), axis=0)
                peer_mad[peer_mad == 0] = 1e-6  # avoid division by zero

                device_values = df[col].values
                deviation = np.abs(device_values - peer_median) / peer_mad
                # Normalize: score = max(0, deviation - threshold) / threshold
                channel_score = np.clip((deviation - threshold) / threshold, 0, None)
                channel_score = np.nan_to_num(channel_score, nan=0.0)
                scores += channel_score

            # Average across channels
            scores /= max(len(available_cols), 1)
            results[did] = pd.Series(
                scores, index=common_idx, name="cross_device_score"
            ).reindex(devices[did].index, method="nearest")

    return results


# ─── Step 6: Ensemble Scoring ────────────────────────────────────────────────

def normalize_scores(s: pd.Series) -> pd.Series:
    """Normalize a score series to [0, 1] using robust min-max (1st-99th percentile)."""
    low = s.quantile(0.01)
    high = s.quantile(0.99)
    if high == low:
        return pd.Series(0.0, index=s.index, name=s.name)
    return ((s - low) / (high - low)).clip(0, 1)


def compute_ensemble_score(
    lof_scores: pd.Series,
    mahal_scores: pd.Series,
    cross_scores: pd.Series,
) -> pd.DataFrame:
    """
    Combine the three outlier scores into a final ensemble score.

    Returns DataFrame with individual normalized scores, ensemble score, and binary flag.
    """
    w = CONFIG["ensemble_weights"]

    lof_norm = normalize_scores(lof_scores.fillna(0))
    mahal_norm = normalize_scores(mahal_scores.fillna(0))
    cross_norm = normalize_scores(cross_scores.fillna(0))

    ensemble = w[0] * lof_norm + w[1] * mahal_norm + w[2] * cross_norm

    result = pd.DataFrame({
        "lof_score": lof_norm,
        "mahal_score": mahal_norm,
        "cross_device_score": cross_norm,
        "ensemble_score": ensemble,
        "is_outlier": ensemble > CONFIG["outlier_threshold"],
    })

    return result


# ─── Step 7: Visualization ──────────────────────────────────────────────────

def visualize_device_outliers(
    df: pd.DataFrame, scores_df: pd.DataFrame, device_id: str, output_dir: str = "."
):
    """Plot time series with outlier points highlighted."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    outlier_mask = scores_df["is_outlier"].values

    for ax, col, label in zip(
        axes[:4],
        COLUMN_NAMES,
        ["Temperature (C)", "Humidity Ambient (%)", "Humidity Cavity (%)", "Moisture Resistance (%)"],
    ):
        if col in df.columns:
            ax.plot(df.index, df[col], linewidth=0.3, alpha=0.7, label=label)
            # Highlight outliers
            outlier_idx = df.index[outlier_mask[: len(df)]]
            if col in df.columns:
                ax.scatter(
                    outlier_idx,
                    df.loc[outlier_idx, col],
                    c="red", s=4, alpha=0.6, label="Outlier", zorder=5,
                )
        ax.set_ylabel(label, fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    # Ensemble score
    axes[4].plot(scores_df.index, scores_df["ensemble_score"], linewidth=0.3, color="purple")
    axes[4].axhline(y=CONFIG["outlier_threshold"], color="red", linestyle="--", linewidth=0.8, label="Threshold")
    axes[4].fill_between(
        scores_df.index, 0, scores_df["ensemble_score"],
        where=scores_df["is_outlier"], color="red", alpha=0.3,
    )
    axes[4].set_ylabel("Ensemble Score", fontsize=8)
    axes[4].legend(loc="upper right", fontsize=7)
    axes[4].grid(True, alpha=0.3)

    fig.suptitle(f"Outlier Detection - Device {device_id[:12]}...", fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"outliers_{device_id[:12]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out_path}")
    return out_path


def visualize_installation_heatmap(
    all_results: dict[str, pd.DataFrame], installation_id: str, output_dir: str = "."
):
    """Heatmap of outlier density across devices and time (monthly buckets)."""
    import matplotlib.pyplot as plt

    device_ids = sorted(all_results.keys())
    if not device_ids:
        return

    # Build monthly outlier rate per device
    records = []
    for did in device_ids:
        res = all_results[did]
        monthly = res["is_outlier"].resample("ME").mean()
        for month, rate in monthly.items():
            records.append({"device": did[:12], "month": month, "outlier_rate": rate})

    heatmap_df = pd.DataFrame(records).pivot(index="device", columns="month", values="outlier_rate")
    heatmap_df = heatmap_df.fillna(0)

    fig, ax = plt.subplots(figsize=(18, max(4, len(device_ids) * 0.4)))
    im = ax.imshow(heatmap_df.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.3)

    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index, fontsize=7)
    # Show every 3rd month label
    month_labels = [d.strftime("%Y-%m") for d in heatmap_df.columns]
    ax.set_xticks(range(0, len(month_labels), 3))
    ax.set_xticklabels(month_labels[::3], fontsize=7, rotation=45, ha="right")

    plt.colorbar(im, ax=ax, label="Outlier Rate")
    ax.set_title(f"Monthly Outlier Rate - Installation {installation_id[:12]}...")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"heatmap_{installation_id[:12]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {out_path}")
    return out_path


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(
    data_dir: str,
    output_dir: str = "results",
    max_devices: int | None = None,
    use_fast_mahal: bool = True,
):
    """
    Run the full outlier detection pipeline.

    Args:
        data_dir: Path to exported_data directory
        output_dir: Directory for output CSVs and plots
        max_devices: Limit number of devices to process (for testing)
        use_fast_mahal: Use fast block-based Mahalanobis (recommended for large data)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    print("=" * 60)
    print("STEP 1: Loading sensor data...")
    print("=" * 60)
    devices = load_device_data(data_dir)

    if max_devices:
        device_ids = list(devices.keys())[:max_devices]
        devices = {k: devices[k] for k in device_ids}
        print(f"  Limited to {max_devices} devices for testing")

    # Step 5 (pre-compute): Cross-device consensus
    print("\n" + "=" * 60)
    print("STEP 5: Computing cross-device consensus scores...")
    print("=" * 60)
    cross_scores = detect_cross_device_outliers(devices)

    # Process each device
    all_results: dict[str, pd.DataFrame] = {}

    for i, (device_id, df) in enumerate(devices.items()):
        print(f"\n{'-' * 60}")
        print(f"Processing device {i + 1}/{len(devices)}: {device_id[:12]}...")
        print(f"  Samples: {len(df)}, Range: {df.index[0]} -> {df.index[-1]}")
        print(f"  Channels: {list(df.columns)}")
        print(f"  NaN ratio: {df.isna().mean().to_dict()}")

        # Determine available feature columns for this device
        available_cols = [c for c in COLUMN_NAMES if c in df.columns]

        # Step 2: Feature engineering
        df_feat = engineer_features(df)
        print(f"  Features: {len(df_feat.columns)} cols, {len(df_feat)} rows after dropna")

        if len(df_feat) < CONFIG["lof_window_size"]:
            print(f"  SKIP: too few samples ({len(df_feat)} < {CONFIG['lof_window_size']})")
            continue

        # Step 3: LOF
        print("  Running LOF...")
        lof_scores = detect_lof_outliers(df_feat, feature_cols=available_cols)

        # Step 4: Mahalanobis
        print("  Running Mahalanobis...")
        if use_fast_mahal:
            mahal_scores = detect_mahalanobis_outliers_fast(df_feat, feature_cols=available_cols)
        else:
            mahal_scores = detect_mahalanobis_outliers(df_feat, feature_cols=available_cols)

        # Step 5: Cross-device (already computed)
        device_cross = cross_scores.get(
            device_id,
            pd.Series(0.0, index=df_feat.index, name="cross_device_score"),
        )
        device_cross = device_cross.reindex(df_feat.index, method="nearest").fillna(0)

        # Step 6: Ensemble
        print("  Computing ensemble score...")
        scores_df = compute_ensemble_score(lof_scores, mahal_scores, device_cross)

        n_outliers = scores_df["is_outlier"].sum()
        pct = 100 * n_outliers / len(scores_df) if len(scores_df) > 0 else 0
        print(f"  Outliers: {n_outliers} / {len(scores_df)} ({pct:.2f}%)")

        # Save CSV
        export_df = scores_df.copy()
        export_df["device_id"] = device_id
        export_df["installation_id"] = df.attrs.get("installation_id", "unknown")
        export_df.to_csv(os.path.join(output_dir, f"scores_{device_id[:12]}.csv"))

        # Step 7: Visualize
        visualize_device_outliers(df_feat, scores_df, device_id, output_dir)

        all_results[device_id] = scores_df

    # Installation heatmaps
    print("\n" + "=" * 60)
    print("Generating installation heatmaps...")
    print("=" * 60)
    installations: dict[str, dict[str, pd.DataFrame]] = {}
    for device_id, res in all_results.items():
        inst_id = devices[device_id].attrs.get("installation_id", "unknown")
        installations.setdefault(inst_id, {})[device_id] = res

    for inst_id, inst_results in installations.items():
        visualize_installation_heatmap(inst_results, inst_id, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_points = sum(len(r) for r in all_results.values())
    total_outliers = sum(r["is_outlier"].sum() for r in all_results.values())
    print(f"  Devices processed: {len(all_results)}")
    print(f"  Total data points: {total_points:,}")
    print(f"  Total outliers: {total_outliers:,} ({100 * total_outliers / max(1, total_points):.2f}%)")
    print(f"  Results saved to: {output_dir}/")

    return all_results


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Senzomatic Outlier Detection")
    parser.add_argument(
        "--data-dir",
        default=r"G:\My Drive\Rozinet\RMind\Clients\Senzomatic\Data_sensors\exported_data_2026-01-22\exported_data",
        help="Path to exported_data directory",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-devices", type=int, default=None, help="Limit devices (for testing)")
    parser.add_argument("--fast", action="store_true", default=True, help="Use fast Mahalanobis")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_devices=args.max_devices,
        use_fast_mahal=args.fast,
    )
