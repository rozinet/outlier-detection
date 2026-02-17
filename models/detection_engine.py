"""Detection engine wrapper for outlier detection algorithms."""

import sys
import os
import pandas as pd
from typing import Optional

# Import detection functions from original outlier.py
# Add parent directory to path to import outlier module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import all detection functions from original script
try:
    from outlier import (
        detect_moisture_intrusion,
        detect_condensation_risk,
        detect_drying_failure,
        detect_sensor_malfunction,
        detect_rapid_moisture_change,
        compute_health_score,
        detect_installation_outliers,
    )
    DETECTORS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import detection functions from outlier.py: {e}")
    DETECTORS_AVAILABLE = False


class DetectionEngine:
    """Wrapper for running outlier detection algorithms."""

    def __init__(self, config):
        """Initialize detection engine.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        if not DETECTORS_AVAILABLE:
            print("Warning: Detection functions not available. Detection will be disabled.")

    def detect_all_problems(self, df: pd.DataFrame, device_id: str,
                           fleet_seasonal: Optional[pd.DataFrame] = None) -> list:
        """Run all detectors on a device's data.

        Args:
            df: Device DataFrame with preprocessed data
            device_id: Device identifier
            fleet_seasonal: Fleet seasonal profile (optional)

        Returns:
            List of Problem objects
        """
        if not DETECTORS_AVAILABLE:
            return []

        problems = []

        # Detector 1: Moisture intrusion
        try:
            probs = detect_moisture_intrusion(df, device_id)
            problems.extend(probs)
        except Exception as e:
            print(f"  Error in moisture intrusion detector for {device_id}: {e}")

        # Detector 2: Condensation risk
        try:
            probs = detect_condensation_risk(df, device_id, fleet_seasonal=fleet_seasonal)
            problems.extend(probs)
        except Exception as e:
            print(f"  Error in condensation detector for {device_id}: {e}")

        # Detector 3: Drying failure
        try:
            probs = detect_drying_failure(df, device_id)
            problems.extend(probs)
        except Exception as e:
            print(f"  Error in drying failure detector for {device_id}: {e}")

        # Detector 4: Sensor malfunction
        try:
            probs = detect_sensor_malfunction(df, device_id)
            problems.extend(probs)
        except Exception as e:
            print(f"  Error in sensor malfunction detector for {device_id}: {e}")

        # Detector 5: Rapid moisture change
        try:
            probs = detect_rapid_moisture_change(df, device_id)
            problems.extend(probs)
        except Exception as e:
            print(f"  Error in rapid moisture change detector for {device_id}: {e}")

        return problems

    def compute_device_health_score(self, problems: list, data_span_days: float) -> dict:
        """Compute health score for a device.

        Args:
            problems: List of Problem objects
            data_span_days: Data span in days

        Returns:
            Dictionary with health score and breakdown
        """
        if not DETECTORS_AVAILABLE:
            return {"score": 100, "breakdown": {}}

        try:
            # Convert days to hours as the function expects hours
            data_span_hours = data_span_days * 24
            return compute_health_score(problems, data_span_hours)
        except Exception as e:
            print(f"  Error computing health score: {e}")
            return {"score": 100, "breakdown": {}}

    def detect_outlier_installations(self, all_problems: dict, devices: dict,
                                    installations: dict = None) -> dict:
        """Detect outlier installations.

        Args:
            all_problems: Dictionary mapping device_id to list of Problems
            devices: Dictionary mapping device_id to DataFrame
            installations: Dictionary mapping installation_id to dict of device_ids to Problems.
                          If None, creates a single installation with all devices.

        Returns:
            Dictionary mapping device_id to list of outlier Problems
        """
        if not DETECTORS_AVAILABLE:
            return {}

        try:
            # If no installations provided, create a single installation with all devices
            if installations is None:
                installations = {
                    "default_installation": {
                        device_id: all_problems.get(device_id, [])
                        for device_id in devices.keys()
                    }
                }

            # Call with correct parameter order: devices, all_problems, installations
            return detect_installation_outliers(devices, all_problems, installations)
        except Exception as e:
            print(f"  Error detecting installation outliers: {e}")
            return {}
