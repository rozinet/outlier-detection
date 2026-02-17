"""Controller for detection pipeline operations."""

import os
from typing import Optional
from models import (
    ConfigManager,
    Building,
    load_building_sensors,
    preprocess_device_data,
    compute_fleet_seasonal_profile,
    DetectionEngine,
)


class DetectionController:
    """Handles detection pipeline execution."""

    def __init__(self, config: ConfigManager):
        """Initialize detection controller.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.detection_engine = DetectionEngine(config)
        self.results = {}  # Store detection results

    def run_detection(self, building: Building, data_dir: str,
                     progress_callback=None) -> dict:
        """Run detection pipeline on a building.

        Args:
            building: Building object with sensors to analyze
            data_dir: Root directory containing sensor data
            progress_callback: Optional callback function(message: str, progress: float)

        Returns:
            Dictionary with detection results:
            {
                'devices': {device_id: DataFrame},
                'problems': {device_id: [Problem, ...]},
                'health_scores': {device_id: {'score': float, 'breakdown': dict}},
                'fleet_seasonal': DataFrame or None,
                'outliers': [device_id, ...]
            }
        """
        if progress_callback:
            progress_callback("Loading sensor data...", 0.1)

        # Load data for building sensors
        devices = load_building_sensors(
            building,
            data_dir,
            self.config.SENSOR_TYPES,
            self.config.COLUMN_NAMES,
            self.config["resample_interval"]
        )

        if not devices:
            return {
                'devices': {},
                'problems': {},
                'health_scores': {},
                'fleet_seasonal': None,
                'outliers': []
            }

        if progress_callback:
            progress_callback(f"Loaded {len(devices)} sensors", 0.2)

        # Preprocess all devices
        min_hours = self.config["min_data_hours"]
        for device_id in list(devices.keys()):
            try:
                df = devices[device_id]
                if len(df) < min_hours * 12:  # 12 samples per hour (5-min intervals)
                    print(f"  Skipping {device_id[:12]}... (insufficient data)")
                    del devices[device_id]
                    continue
                devices[device_id] = preprocess_device_data(df, self.config.config)
            except Exception as e:
                print(f"  Error preprocessing {device_id[:12]}...: {e}")
                del devices[device_id]
                continue

        if progress_callback:
            progress_callback(f"Preprocessed {len(devices)} sensors", 0.3)

        # Compute fleet seasonal profile
        fleet_seasonal = compute_fleet_seasonal_profile(devices, "hum_cavity")

        # Run detectors on each device
        all_problems = {}
        health_scores = {}
        total_devices = len(devices)

        for i, (device_id, df) in enumerate(devices.items()):
            if progress_callback:
                progress = 0.3 + (0.6 * (i / total_devices))
                progress_callback(f"Analyzing sensor {i+1}/{total_devices}...", progress)

            try:
                # Run all detectors
                problems = self.detection_engine.detect_all_problems(
                    df, device_id, fleet_seasonal
                )
                all_problems[device_id] = problems

                # Compute health score
                data_span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
                health_scores[device_id] = self.detection_engine.compute_device_health_score(
                    problems, data_span_days
                )
            except Exception as e:
                print(f"  Error analyzing {device_id[:12]}...: {e}")
                # Add empty results for this device so it doesn't break reporting
                all_problems[device_id] = []
                health_scores[device_id] = {"score": 100, "breakdown": {}}

        if progress_callback:
            progress_callback("Detecting outlier installations...", 0.9)

        # Detect outlier installations
        outliers = self.detection_engine.detect_outlier_installations(
            all_problems, devices
        )

        if progress_callback:
            progress_callback("Detection complete!", 1.0)

        results = {
            'devices': devices,
            'problems': all_problems,
            'health_scores': health_scores,
            'fleet_seasonal': fleet_seasonal,
            'outliers': outliers
        }

        self.results = results
        return results

    def get_results_summary(self) -> dict:
        """Get summary of detection results.

        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {
                'total_devices': 0,
                'total_problems': 0,
                'problems_by_type': {},
                'problems_by_severity': {},
                'avg_health_score': 0.0
            }

        all_problems = self.results.get('problems', {})
        health_scores = self.results.get('health_scores', {})

        # Count problems by type and severity
        problems_by_type = {}
        problems_by_severity = {}
        total_problems = 0

        for device_id, problems in all_problems.items():
            for problem in problems:
                total_problems += 1
                problems_by_type[problem.problem_type] = problems_by_type.get(problem.problem_type, 0) + 1
                problems_by_severity[problem.severity] = problems_by_severity.get(problem.severity, 0) + 1

        # Calculate average health score
        if health_scores:
            avg_score = sum(h['score'] for h in health_scores.values()) / len(health_scores)
        else:
            avg_score = 0.0

        return {
            'total_devices': len(all_problems),
            'total_problems': total_problems,
            'problems_by_type': problems_by_type,
            'problems_by_severity': problems_by_severity,
            'avg_health_score': avg_score
        }
