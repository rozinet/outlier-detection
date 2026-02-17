"""Configuration management for the outlier detection system."""

import json
import os
from typing import Any


class ConfigManager:
    """Manages application configuration with save/load capabilities."""

    DEFAULT_CONFIG = {
        "resample_interval": "5min",

        # Preprocessing (G1: multi-resolution)
        "median_filter_window": 7,
        "smoothing_window": 6,
        "ewma_halflife_hours": 6,
        "use_ewma_trend": True,
        "min_data_hours": 48,

        # Moisture intrusion (F4: stricter AND logic)
        "moisture_drop_threshold_24h": 3.0,
        "moisture_drop_threshold_7d": 5.0,
        "cavity_rise_threshold_24h": 8.0,
        "moisture_intrusion_min_hours": 24,
        "moisture_drop_min": 1.0,
        "cavity_rise_min": 1.0,

        # Condensation risk (A1: seasonal-aware, A2+F1-F3: chronic/recurring merge)
        "condensation_warning_pct": 80.0,
        "condensation_danger_pct": 90.0,
        "condensation_critical_pct": 95.0,
        "condensation_min_hours": 48,
        "condensation_chronic_pct_warning": 40.0,
        "condensation_chronic_pct_severe": 50.0,
        "condensation_chronic_min_months": 6,
        "condensation_recurring_min_episodes": 6,
        "condensation_warning_merge_gap_days": 21,
        "seasonal_baseline_window_days": 30,
        "condensation_hysteresis_band": 2.0,
        "condensation_use_abs_humidity": True,
        "abs_humidity_warning_gkg": 14.0,
        "fleet_seasonal_offset_pct": 8.0,

        # Rapid moisture change (A5)
        "rapid_moisture_drop_3d": 4.0,
        "rapid_moisture_drop_14d": 8.0,
        "rapid_change_min_hours": 12,

        # Rapid moisture change: CUSUM (G5)
        "cusum_threshold": 5.0,
        "cusum_drift": 0.5,
        "cusum_confirmation_window_hours": 72,

        # Drying failure (G6: exponential curve)
        "drying_eval_window_weeks": 4,
        "drying_plateau_tolerance": 0.5,
        "drying_reversal_threshold": 1.0,
        "drying_tau_warning_days": 365,
        "drying_tau_danger_days": 730,
        "drying_plateau_pct": 80.0,
        "drying_initial_wet_threshold": 85.0,
        "drying_exp_fit_min_days": 60,

        # Sensor malfunction (A3: saturation-aware)
        "flatline_window_hours": 24,
        "jump_threshold_temp": 10.0,
        "jump_threshold_humidity": 25.0,
        "jump_threshold_moisture": 20.0,
        "jump_min_count": 3,
        "temp_range": (-40.0, 60.0),
        "humidity_range": (0.0, 100.0),
        "moisture_range": (0.0, 100.0),
        "saturation_values": {
            "hum_ambient": [0.0, 100.0],
            "hum_cavity": [0.0, 100.0],
            "moisture": [0.0, 100.0],
        },

        # Sensor malfunction: v6 additions (G8)
        "hampel_window": 25,
        "hampel_threshold": 3.0,
        "sensor_residual_window_days": 14,
        "sensor_drift_threshold_std": 3.0,

        # Installation outliers (G9)
        "outlier_mad_zscore_threshold": 3.0,
        "outlier_min_devices": 3,

        # Episode merging
        "episode_merge_gap_hours": 6,

        # Health score (C11)
        "health_score_weights": {
            "condensation_risk": {"warning": 5, "danger": 15, "critical": 30},
            "moisture_intrusion": {"warning": 10, "danger": 25, "critical": 40},
            "drying_failure": {"warning": 8, "danger": 20, "critical": 35},
            "sensor_malfunction": {"warning": 3, "danger": 10, "critical": 20},
            "rapid_moisture_change": {"warning": 8, "danger": 20, "critical": 35},
        },
    }

    SENSOR_TYPES = [
        "temperature_ambient_celsius",
        "rel_humidity_ambient_pct",
        "rel_humidity_cavity_pct",
        "moisture_resistance_pct",
    ]

    COLUMN_NAMES = ["temp", "hum_ambient", "hum_cavity", "moisture"]

    # Severity levels
    OK = "ok"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration manager.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple configuration values.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        self.config.update(updates)

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()

    def load(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: Failed to load config from {self.config_file}: {e}")

    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except OSError as e:
            print(f"Error: Failed to save config to {self.config_file}: {e}")

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        self.config[key] = value
