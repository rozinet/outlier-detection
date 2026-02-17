"""Data loading utilities for sensor data."""

import json
import os
import numpy as np
import pandas as pd
from typing import Optional


def load_single_channel(filepath: str) -> Optional[pd.Series]:
    """Load a single sensor channel from a .json_line file.

    Args:
        filepath: Path to .json_line file

    Returns:
        Series with sensor data or None if loading fails
    """
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


def load_device_data(data_dir: str, sensor_types: list[str], column_names: list[str],
                     resample_interval: str = "5min") -> dict[str, pd.DataFrame]:
    """Load all devices, align channels on time grid.

    Args:
        data_dir: Root directory containing sensor data files
        sensor_types: List of sensor type names to load
        column_names: Column names for the sensor types
        resample_interval: Resampling interval (default: "5min")

    Returns:
        Dictionary mapping device_id to DataFrame
    """
    device_files: dict[str, dict[str, str]] = {}
    installation_map: dict[str, str] = {}

    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".json_line"):
                continue
            device_id = fname[:36]
            sensor_type = fname[37:].replace(".json_line", "")
            if sensor_type not in sensor_types:
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
        for sensor_type, col_name in zip(sensor_types, column_names):
            if sensor_type in files_map:
                series = load_single_channel(files_map[sensor_type])
                if series is not None and len(series) > 0:
                    channels[col_name] = series
        if len(channels) < 2:
            continue
        df = pd.DataFrame(channels)
        df = df.resample(resample_interval).mean()
        df = df.dropna(how="all")
        df = df.ffill(limit=6)
        df.attrs["device_id"] = device_id
        df.attrs["installation_id"] = installation_map.get(device_id, "unknown")
        devices[device_id] = df

    print(f"Loaded {len(devices)} devices")
    return devices


def load_building_sensors(building, data_dir: str, sensor_types: list[str],
                          column_names: list[str], resample_interval: str = "5min") -> dict[str, pd.DataFrame]:
    """Load sensor data for all sensors in a building.

    Args:
        building: Building object with sensors
        data_dir: Root directory containing sensor data files
        sensor_types: List of sensor type names to load
        column_names: Column names for the sensor types
        resample_interval: Resampling interval

    Returns:
        Dictionary mapping sensor_id to DataFrame
    """
    sensors = building.get_all_sensors()
    active_sensors = [s for s in sensors if s.is_active]

    # Build a map of sensor_id -> data_path for sensors that have a specific path
    sensor_paths = {}
    for sensor in active_sensors:
        if sensor.data_path:
            sensor_paths[sensor.sensor_id] = os.path.join(data_dir, sensor.data_path)
        else:
            sensor_paths[sensor.sensor_id] = data_dir

    building_devices = {}

    # Load each sensor individually with error handling
    for sensor in active_sensors:
        sensor_id = sensor.sensor_id
        search_path = sensor_paths.get(sensor_id, data_dir)

        try:
            # Check if sensor files exist in the specified path
            sensor_files = {}
            found_files = False

            for root, _dirs, files in os.walk(search_path):
                for fname in files:
                    if not fname.endswith(".json_line"):
                        continue
                    # Check if this file belongs to this sensor
                    file_device_id = fname[:36]
                    if file_device_id != sensor_id:
                        continue

                    sensor_type = fname[37:].replace(".json_line", "")
                    if sensor_type not in sensor_types:
                        continue

                    sensor_files[sensor_type] = os.path.join(root, fname)
                    found_files = True

            if not found_files:
                print(f"  Warning: No data files found for sensor {sensor_id[:12]}... in {search_path}")
                continue

            # Load channels for this sensor
            channels = {}
            for sensor_type, col_name in zip(sensor_types, column_names):
                if sensor_type in sensor_files:
                    series = load_single_channel(sensor_files[sensor_type])
                    if series is not None and len(series) > 0:
                        channels[col_name] = series

            # Need at least 2 channels
            if len(channels) < 2:
                print(f"  Warning: Insufficient channels for sensor {sensor_id[:12]}... (found {len(channels)})")
                continue

            # Create DataFrame and resample
            df = pd.DataFrame(channels)
            df = df.resample(resample_interval).mean()
            df = df.dropna(how="all")
            df = df.ffill(limit=6)

            # Add metadata
            df.attrs["device_id"] = sensor_id
            df.attrs["installation_id"] = "unknown"

            building_devices[sensor_id] = df
            print(f"  Loaded sensor {sensor_id[:12]}... ({len(df)} samples)")

        except Exception as e:
            print(f"  Error loading sensor {sensor_id[:12]}...: {e}")
            # Continue with next sensor instead of crashing
            continue

    print(f"Successfully loaded {len(building_devices)} sensors for building")
    return building_devices
