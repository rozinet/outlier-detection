"""Problem data structure for detected building issues."""

from dataclasses import dataclass, field
import pandas as pd


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
        """Get duration in hours."""
        return (self.end - self.start).total_seconds() / 3600

    @property
    def is_chronic(self) -> bool:
        """Check if problem is chronic (>90 days)."""
        return self.duration_hours > 90 * 24  # >90 days

    def __str__(self):
        """String representation."""
        dur = f"{self.duration_hours:.0f}h"
        tag = " [CHRONIC]" if self.is_chronic else ""
        return f"[{self.severity.upper():8s}] {self.problem_type:22s} | {self.device_id[:12]} | {self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d} ({dur}){tag} | {self.description}"
