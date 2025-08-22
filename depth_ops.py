# depth_ops.py
# -*- coding: utf-8 -*-
"""
Core algorithms for depth/point cloud processing.

This module contains no visualization code. It provides:
- Data classes for configuration
- Depth <-> point cloud transforms
- Point-cloud filtering / rotation helpers
- Laser-scan extraction from point-cloud
- Occupancy grid projection
- Convenience API wrappers that accept raw params

Coordinate systems:
- 'opengl':  X right, Y up, Z forward-negative (visible points have Z<0)
- 'opencv' : X right, Y down, Z forward-positive (visible points have Z>0)

This refactor keeps parity with the previous implementation and also exposes
some old function names as aliases for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class SensorConfig:
    """Camera sensor configuration."""
    fov_deg: Tuple[float, float] = (90.0, 90.0)   # (fov_x_deg, fov_y_deg)
    dist_scale: float = 1.0                       # multiply depth by this to get meters


@dataclass
class TransformConfig:
    """Point-cloud pre-processing configuration."""
    # Each item: (axis, angle_deg)  where axis in {'x','y','z'} or a 3-vector
    rotate_points: List = field(default_factory=lambda: [['x', -30]])
    # Each item: [axis, min_val, max_val] — axis in {'x','y','z'} or index {0,1,2}
    filter_points: List = field(default_factory=lambda: [['y', -0.25, 0.25]])


@dataclass
class ProjectionConfig:
    """Occupancy grid projection configuration."""
    map_resolution: float = 0.2    # meters per cell
    map_size: int = 100            # square grid size (map_size x map_size)


@dataclass
class LaserScanConfig:
    """Laser scan extraction configuration."""
    aggregation: str = 'min'       # 'min', 'max', 'mean', 'median'
    n_intervals: int = 30
    default_value: float = 3.0     # meters (cap for no-returns)


@dataclass
class Config:
    """
    Unified configuration for all transforms.
    """
    coordinate_system: str = 'opengl'  # 'opengl' | 'opencv'
    sensor: SensorConfig = field(default_factory=SensorConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    laserscan: LaserScanConfig = field(default_factory=LaserScanConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Load configuration from a YAML file.
        Accepts new nested keys (sensor, transform, projection, laserscan)
        or legacy flat keys (sensor_cfg, transform_cfg, ...).
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Support both new and legacy keys
        sensor = data.get("sensor") or data.get("sensor_cfg") or {}
        transform = data.get("transform") or data.get("transform_cfg") or {}
        projection = data.get("projection") or data.get("projection_cfg") or {}
        laserscan = data.get("laserscan") or data.get("laserscan_cfg") or {}
        coordinate_system = data.get("coordinate_system") or data.get("corrdinate_system") or 'opengl'

        return cls(
            coordinate_system=str(coordinate_system).lower(),
            sensor=SensorConfig(**sensor),
            transform=TransformConfig(**transform),
            projection=ProjectionConfig(**projection),
            laserscan=LaserScanConfig(**laserscan),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Camera intrinsics helpers
# -----------------------------------------------------------------------------

def intrinsics_from_fov(width: int, height: int, fov_deg: Sequence[float]) -> Tuple[float, float, float, float]:
    """
    Compute pinhole intrinsics (fx, fy, cx, cy) from image size and FOV.

    Parameters
    ----------
    width, height : int
    fov_deg : (float, float)
        (fov_x_deg, fov_y_deg). Must match the convention used in the dataset.

    Returns
    -------
    fx, fy, cx, cy : float
    """
    fov_x, fov_y = np.deg2rad(fov_deg[0]), np.deg2rad(fov_deg[1])
    fx = width / (2.0 * np.tan(fov_x * 0.5))
    fy = height / (2.0 * np.tan(fov_y * 0.5))
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    return fx, fy, cx, cy


# -----------------------------------------------------------------------------
# Depth <-> point-cloud transforms
# -----------------------------------------------------------------------------

def depth_to_pointcloud(
    depth: np.ndarray,
    fov_deg: Sequence[float] = (90.0, 90.0),
    dist_scale: float = 1.0,
    coordinate_system: str = 'opengl',
) -> np.ndarray:
    """
    Convert a depth map (H×W) to a point cloud (H×W×3) in the camera frame.

    Depth convention
    ----------------
    For 'opengl': visible points have Z_cam < 0 and depth = -Z_cam (>0).
    For 'opencv' : visible points have Z_cam > 0 and depth = +Z_cam (>0).

    Parameters
    ----------
    depth : (H,W) array
        Depth values in meters / dist_scale (i.e. raw_depth * dist_scale = meters).
    fov_deg : (fov_x_deg, fov_y_deg)
    dist_scale : float
    coordinate_system : {'opengl','opencv'}

    Returns
    -------
    pts : (H,W,3) float32
        Camera-frame coordinates following `coordinate_system`.
    """
    assert depth.ndim == 2, "depth must be (H,W)"
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics_from_fov(W, H, fov_deg)

    # Convert raw depth to metric distance along the camera forward axis
    Z = depth.astype(np.float32) * dist_scale  # positive scalar

    # Construct pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Project back to rays and scale by depth
    X = (uu - cx) * Z / fx

    cs = coordinate_system.lower()
    if cs == 'opencv':
        Y = (vv - cy) * Z / fy   # Y down
        Zcam = +Z
    elif cs == 'opengl':
        Y = -(vv - cy) * Z / fy  # Y up
        Zcam = -Z
    else:
        raise ValueError("coordinate_system must be 'opencv' or 'opengl'")

    pts = np.stack([X, Y, Zcam], axis=-1)
    return pts


def pointcloud_to_depth(
    pts: np.ndarray,
    height: int,
    width: int,
    fov_deg: Sequence[float] = (90.0, 90.0),
    dist_scale: float = 1.0,
    coordinate_system: str = "opengl",
    aggregation: str = "zmin",
    radius: int = 0,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Project a camera-frame point cloud (N,3) to a pinhole depth image (H,W).

    The function supports simple "splatting" with a square kernel of radius `r`,
    and three aggregation rules for collisions: 'zmin' (closest), 'zmax', 'zmean'.

    Returns
    -------
    depth : (H,W) float32, where depth = forward_distance / dist_scale.
    """
    assert pts.ndim == 2 and pts.shape[1] == 3, "pts must be (N,3)"
    H, W = int(height), int(width)
    fx, fy, cx, cy = intrinsics_from_fov(W, H, fov_deg)

    Zfwd = -pts[:, 2] if coordinate_system.lower() == 'opengl' else pts[:, 2]
    valid = Zfwd > 0.0
    if not np.any(valid):
        return np.full((H, W), fill_value, dtype=np.float32)

    X = pts[valid, 0]
    Y = pts[valid, 1]
    Zfwd = Zfwd[valid]

    u = fx * (X / Zfwd) + cx
    if coordinate_system.lower() == "opencv":
        v = fy * (Y / Zfwd) + cy
    else:  # 'opengl'
        v = -fy * (Y / Zfwd) + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not np.any(in_bounds):
        return np.full((H, W), fill_value, dtype=np.float32)

    ui, vi, Zfwd = ui[in_bounds], vi[in_bounds], Zfwd[in_bounds]

    r = int(radius)
    if aggregation == "zmin":
        buf = np.full((H, W), np.inf, dtype=np.float32)
        for px, py, z in zip(ui, vi, Zfwd):
            x0 = max(0, px - r); x1 = min(W - 1, px + r)
            y0 = max(0, py - r); y1 = min(H - 1, py + r)
            block = buf[y0:y1+1, x0:x1+1]
            buf[y0:y1+1, x0:x1+1] = np.minimum(block, z)
        out = np.full_like(buf, fill_value, dtype=np.float32)
        mask = np.isfinite(buf)
        out[mask] = (buf[mask] / dist_scale).astype(np.float32)
        return out

    elif aggregation == "zmax":
        buf = np.full((H, W), -np.inf, dtype=np.float32)
        for px, py, z in zip(ui, vi, Zfwd):
            x0 = max(0, px - r); x1 = min(W - 1, px + r)
            y0 = max(0, py - r); y1 = min(H - 1, py + r)
            block = buf[y0:y1+1, x0:x1+1]
            buf[y0:y1+1, x0:x1+1] = np.maximum(block, z)
        out = np.full_like(buf, fill_value, dtype=np.float32)
        mask = np.isfinite(buf)
        out[mask] = (buf[mask] / dist_scale).astype(np.float32)
        return out

    elif aggregation == "zmean":
        acc = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.int32)
        for px, py, z in zip(ui, vi, Zfwd):
            x0 = max(0, px - r); x1 = min(W - 1, px + r)
            y0 = max(0, py - r); y1 = min(H - 1, py + r)
            acc[y0:y1+1, x0:x1+1] += z
            cnt[y0:y1+1, x0:x1+1] += 1
        out = np.full((H, W), fill_value, dtype=np.float32)
        valid = cnt > 0
        out[valid] = (acc[valid] / cnt[valid] / dist_scale).astype(np.float32)
        return out

    else:
        raise ValueError("aggregation must be 'zmin', 'zmax', or 'zmean'")
    


# -----------------------------------------------------------------------------
# Point-cloud transforms: rotation and filtering
# -----------------------------------------------------------------------------

def _axis_to_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' formula for arbitrary axis rotation."""
    axis = np.asarray(axis, dtype=float)
    axis /= (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-np.around(axis[1], 12), axis[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def rotate_points(points: np.ndarray, rotates: Optional[Iterable] = None) -> np.ndarray:
    """
    Apply one or more rotations to a point-cloud.

    rotates: iterable of (axis, angle_deg), where axis in {'x','y','z'} or a 3-vector.
    """
    if not rotates:
        return points
    R = np.eye(3, dtype=float)
    for axis, theta_deg in rotates:
        theta = np.deg2rad(theta_deg)
        if isinstance(axis, str):
            a = axis.lower()
            if a == 'x':
                R_axis = np.array([[1, 0, 0],
                                   [0, np.cos(theta), -np.sin(theta)],
                                   [0, np.sin(theta),  np.cos(theta)]], dtype=float)
            elif a == 'y':
                R_axis = np.array([[ np.cos(theta), 0, np.sin(theta)],
                                   [0, 1, 0],
                                   [-np.sin(theta), 0, np.cos(theta)]], dtype=float)
            elif a == 'z':
                R_axis = np.array([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta),  np.cos(theta), 0],
                                   [0, 0, 1]], dtype=float)
            else:
                raise ValueError("axis must be 'x', 'y', or 'z' when given as str")
        else:
            R_axis = _axis_to_matrix(np.asarray(axis, dtype=float), theta)
        R = R_axis @ R
    return (points @ R.T)


def filter_points(points: np.ndarray, colors: Optional[np.ndarray], filters: Optional[Iterable] = None
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generic multi-axis range filter for point-clouds.

    filters: iterable where each item is [axis, min_val, max_val].
             axis in {'x','y','z'} or index {0,1,2}.
    """
    if not filters:
        return points, colors

    mask = np.ones(points.shape[0], dtype=bool)
    for item in filters:
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            raise ValueError("Each filter item must be [axis, min_val, max_val]")
        axis, min_val, max_val = item
        if isinstance(axis, str):
            idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        else:
            idx = int(axis)
        if min_val is not None:
            mask &= points[:, idx] >= float(min_val)
        if max_val is not None:
            mask &= points[:, idx] <= float(max_val)

    points_f = points[mask]
    colors_f = colors[mask] if (colors is not None) else None
    return points_f, colors_f


# -----------------------------------------------------------------------------
# High-level pipelines
# -----------------------------------------------------------------------------

def depth_to_filtered_pointcloud(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    height: Optional[float] = None,
    cfg: Optional[Config] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert a depth map to a filtered, optionally-rotated point-cloud.

    Returns
    -------
    pts : (N,3)
    color : (N,3) in [0,1] or None
    """
    cfg = cfg or Config()

    pts_img = depth_to_pointcloud(
        depth=depth,
        fov_deg=cfg.sensor.fov_deg,
        dist_scale=cfg.sensor.dist_scale,
        coordinate_system=cfg.coordinate_system,
    )

    pts = pts_img.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    if cfg.coordinate_system.lower() == 'opengl':
        valid &= pts[:, 2] < 0                # visible: Z<0 in OpenGL
        valid &= (-pts[:, 2]) > 1e-3          # >1mm
    else:
        valid &= pts[:, 2] > 0                # visible: Z>0 in OpenCV
        valid &= (pts[:, 2]) > 1e-3

    pts = pts[valid]

    color = None
    if rgb is not None:
        color = rgb.reshape(-1, 3)[valid] / 255.0

    # Apply geometric transforms
    if cfg.transform.rotate_points:
        pts = rotate_points(pts, cfg.transform.rotate_points)

    if cfg.transform.filter_points:
        pts, color = filter_points(pts, color, cfg.transform.filter_points)

    # Optional ground removal using camera height
    if height is not None:
        if cfg.coordinate_system.lower() == 'opengl':
            ground_filter = ['y', -height, None]  # Y up; ground near -height
        else:  # 'opencv' (Y down)
            ground_filter = ['y', None, height]
        pts, color = filter_points(pts, color, [ground_filter])

    return pts, color


def map_pts_to_intervals(
    pts: np.ndarray,
    fov_deg: Sequence[float] = (90.0, 90.0),
    n_intervals: int = 30,
    default_value: float = 3.0,
    aggregation: str = 'min',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin points by horizontal angle into `n_intervals` wedges and aggregate distance.

    Angle convention
    ----------------
    We estimate the horizontal polar angle using X (right) and abs(Z) (forward distance).
    The default bins are in [0.5*FOVx, 1.5*FOVx], matching the original script.
    You can change this later to [-0.5*FOVx, +0.5*FOVx] if you prefer symmetry.

    Returns
    -------
    angle_intervals : (n,2) rad
    dist_intervals  : (n,) meters
    """
    z_abs = np.abs(pts[:, 2])
    x = pts[:, 0]
    angle = np.arctan2(z_abs, x)   # radians in [0, pi]

    angle_boundaries = np.linspace(np.deg2rad(fov_deg[0] / 2.0),
                                   np.deg2rad(fov_deg[0] * 1.5),
                                   n_intervals + 1)
    angle_intervals: List[Tuple[float, float]] = []
    dist_intervals: List[float] = []

    for i in range(n_intervals):
        start, end = angle_boundaries[i], angle_boundaries[i + 1]
        angle_intervals.append((start, end))
        if i == n_intervals - 1:
            mask = (angle >= start) & (angle <= end)
        else:
            mask = (angle >= start) & (angle < end)
        pts_i = pts[mask]

        if pts_i.size == 0:
            dist_intervals.append(float(default_value))
            continue

        d = np.sqrt(pts_i[:, 0] ** 2 + pts_i[:, 2] ** 2)
        if   aggregation == 'min':   dist_intervals.append(float(np.min(d)))
        elif aggregation == 'max':   dist_intervals.append(float(np.max(d)))
        elif aggregation == 'mean':  dist_intervals.append(float(np.mean(d)))
        elif aggregation == 'median':dist_intervals.append(float(np.median(d)))
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

    return np.asarray(angle_intervals, dtype=float), np.asarray(dist_intervals, dtype=float)


def depth_layer_scan(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    height: Optional[float] = None,
    cfg: Optional[Config] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute a (pseudo) laser scan on the X–Z plane from a depth image.

    Returns
    -------
    x_coord, y_coord : (n,)
        Cartesian coordinates for each angular bin center.
    angles : (n,) rad
    dist   : (n,) meters
    """
    cfg = cfg or Config()

    pts, _ = depth_to_filtered_pointcloud(depth, rgb=rgb, height=height, cfg=cfg)
    if pts.shape[0] == 0:
        return None, None, None, None

    angle_intervals, dist = map_pts_to_intervals(
        pts,
        fov_deg=cfg.sensor.fov_deg,
        n_intervals=cfg.laserscan.n_intervals,
        default_value=cfg.laserscan.default_value,
        aggregation=cfg.laserscan.aggregation,
    )

    angles = (angle_intervals[:, 0] + angle_intervals[:, 1]) * 0.5
    dist = dist.copy()
    dist[dist > cfg.laserscan.default_value] = cfg.laserscan.default_value  # clamp range

    x_coord = dist * np.cos(angles)
    y_coord = dist * np.sin(angles)
    return x_coord, y_coord, angles, dist


def depth_layer_proj(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    height: Optional[float] = None,
    cfg: Optional[Config] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Project a depth image into an occupancy grid on the X–Z plane.

    Returns
    -------
    layer : (N,3) filtered point cloud
    color : (N,3) or None
    occ_map : (S,S) uint8 occupancy map
    """
    cfg = cfg or Config()

    layer, color = depth_to_filtered_pointcloud(depth, rgb=rgb, height=height, cfg=cfg)

    size = cfg.projection.map_size
    res = cfg.projection.map_resolution

    occ_map = np.zeros((size, size), dtype=np.uint8)
    depth_map = np.full((size, size), np.inf)

    half = size // 2
    # Map camera Z (forward) to grid_x, and X (right) to grid_y.
    grid_x = -np.floor(layer[:, 2] / res).astype(int) + half  # forward becomes +x
    grid_y =  np.floor(layer[:, 0] / res).astype(int) + half  # right becomes +y

    valid = (grid_x >= 0) & (grid_x < size) & (grid_y >= 0) & (grid_y < size)
    gx, gy, z = grid_x[valid], grid_y[valid], layer[valid, 1]  # use Y as height for z-buffering

    for i in range(gx.shape[0]):
        x, y, h = int(gx[i]), int(gy[i]), float(z[i])
        if h < depth_map[x, y]:
            depth_map[x, y] = h
            occ_map[x, y] = 1

    return layer, color, occ_map


# -----------------------------------------------------------------------------
# Public convenience APIs (param-based, no Config required by caller)
# -----------------------------------------------------------------------------

def depth_to_filtered_pointcloud_api(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    height: Optional[float] = None,
    fov_deg: Sequence[float] = (90.0, 90.0),
    dist_scale: float = 1.0,
    rotate_points: Optional[Iterable] = None,
    filter_points_: Optional[Iterable] = None,
    coordinate_system: str = 'opengl',
):
    cfg = Config(
        coordinate_system=coordinate_system,
        sensor=SensorConfig(fov_deg=tuple(fov_deg), dist_scale=float(dist_scale)),
        transform=TransformConfig(rotate_points=list(rotate_points or []),
                                  filter_points=list(filter_points_ or [])),
    )
    return depth_to_filtered_pointcloud(depth, rgb=rgb, height=height, cfg=cfg)


def depth_layer_scan_api(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    height: Optional[float] = None,
    fov_deg: Sequence[float] = (90.0, 90.0),
    dist_scale: float = 1.0,
    rotate_points: Optional[Iterable] = None,
    filter_points_: Optional[Iterable] = None,
    aggregation: str = 'min',
    n_intervals: int = 30,
    default_value: float = 3.0,
    coordinate_system: str = 'opengl',
):
    cfg = Config(
        coordinate_system=coordinate_system,
        sensor=SensorConfig(fov_deg=tuple(fov_deg), dist_scale=float(dist_scale)),
        transform=TransformConfig(rotate_points=list(rotate_points or []),
                                  filter_points=list(filter_points_ or [])),
        laserscan=LaserScanConfig(aggregation=aggregation,
                                  n_intervals=int(n_intervals),
                                  default_value=float(default_value)),
    )
    x, y, ang, dist = depth_layer_scan(depth, rgb=rgb, height=height, cfg=cfg)
    if ang is None or dist is None:
        # return a default equally spaced angle array with ones for distance
        angle_boundaries = np.linspace(fov_deg[0] / 2.0, fov_deg[0] * 1.5, n_intervals + 1)
        angles = (angle_boundaries[:-1] + angle_boundaries[1:]) * 0.5
        return angles, np.ones_like(angles, dtype=float)
    # Convert radians to degrees and normalize distances
    angles_deg = np.rad2deg(ang)
    dists_norm = dist / float(default_value)
    return angles_deg, dists_norm


def depth_layer_proj_api(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    height: Optional[float] = None,
    fov_deg: Sequence[float] = (90.0, 90.0),
    dist_scale: float = 1.0,
    rotate_points: Optional[Iterable] = None,
    filter_points_: Optional[Iterable] = None,
    map_resolution: float = 0.2,
    map_size: int = 100,
    coordinate_system: str = 'opengl',
):
    cfg = Config(
        coordinate_system=coordinate_system,
        sensor=SensorConfig(fov_deg=tuple(fov_deg), dist_scale=float(dist_scale)),
        transform=TransformConfig(rotate_points=list(rotate_points or []),
                                  filter_points=list(filter_points_ or [])),
        projection=ProjectionConfig(map_resolution=float(map_resolution),
                                    map_size=int(map_size)),
    )
    return depth_layer_proj(depth, rgb=rgb, height=height, cfg=cfg)


# -----------------------------------------------------------------------------
# Backwards-compatibility aliases (optional)
# -----------------------------------------------------------------------------


__all__ = [
    # Configs
    "SensorConfig", "TransformConfig", "ProjectionConfig", "LaserScanConfig", "Config",
    # Intrinsics
    "intrinsics_from_fov",
    # Core transforms
    "depth_to_pointcloud", "pointcloud_to_depth",
    # Point-cloud ops
    "rotate_points", "filter_points",
    # Pipelines
    "depth_to_filtered_pointcloud", "depth_layer_scan", "depth_layer_proj",
    # APIs
    "depth_to_filtered_pointcloud_api", "depth_layer_scan_api", "depth_layer_proj_api",
]
