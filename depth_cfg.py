# depth_cfg.py
# -*- coding: utf-8 -*-
"""Configuration objects for depth/point-cloud transforms."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

import yaml


@dataclass
class SensorConfig:
    """Camera sensor configuration."""

    fov_deg: Tuple[float, float] = (90.0, 90.0)
    dist_scale: float = 1.0


@dataclass
class TransformConfig:
    """Point-cloud pre-processing configuration."""

    rotate_points: list = field(default_factory=lambda: [["x", -30]])
    filter_points: list = field(default_factory=lambda: [["y", -0.25, 0.25]])


@dataclass
class ProjectionConfig:
    """Occupancy grid projection configuration."""

    map_resolution: float = 0.2
    map_size: int = 100


@dataclass
class LaserScanConfig:
    """Laser scan extraction configuration."""

    aggregation: str = "min"
    n_intervals: int = 30
    default_value: float = 3.0


@dataclass
class Config:
    """Unified configuration for all depth transforms."""

    coordinate_system: str = "opengl"
    sensor: SensorConfig = field(default_factory=SensorConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    laserscan: LaserScanConfig = field(default_factory=LaserScanConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from new or legacy depth-transform YAML keys."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        sensor = data.get("sensor") or data.get("sensor_cfg") or {}
        transform = data.get("transform") or data.get("transform_cfg") or {}
        projection = data.get("projection") or data.get("projection_cfg") or {}
        laserscan = data.get("laserscan") or data.get("laserscan_cfg") or {}
        coordinate_system = (
            data.get("coordinate_system")
            or data.get("corrdinate_system")
            or "opengl"
        )

        return cls(
            coordinate_system=str(coordinate_system).lower(),
            sensor=SensorConfig(**sensor),
            transform=TransformConfig(**transform),
            projection=ProjectionConfig(**projection),
            laserscan=LaserScanConfig(**laserscan),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "SensorConfig",
    "TransformConfig",
    "ProjectionConfig",
    "LaserScanConfig",
    "Config",
]
