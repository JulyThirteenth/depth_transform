# depth_viz.py
# -*- coding: utf-8 -*-
"""
Visualization utilities and CLI for depth/point cloud processing.

This module depends on matplotlib and (optionally) Open3D for interactive viewing.
It imports all algorithm functions from `depth_ops` and keeps plotting separate.
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

# Open3D is optional; we import lazily inside the viewer class.
import depth_ops as ops


# -----------------------------------------------------------------------------
# Plotting: laser scan and multi-pane figure
# -----------------------------------------------------------------------------

def plot_laser_scan(ax, angles_deg: np.ndarray, dists_norm: np.ndarray) -> None:
    """Simple 2D polar-like scatter of (angle, distance)."""
    ax.clear()
    ax.set_aspect('equal')
    x = dists_norm * np.cos(np.deg2rad(angles_deg))
    y = dists_norm * np.sin(np.deg2rad(angles_deg))
    ax.scatter(x, y, marker='o', s=6, alpha=0.9)
    for xi, yi in zip(x, y):
        ax.plot([0, xi], [0, yi], linestyle='--', linewidth=0.6, alpha=0.25)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-0.0, 1.0])
    ax.grid(True, linestyle=':', linewidth=0.5)


def plot_data_frame(rgb: np.ndarray,
                    depth: np.ndarray,
                    height: Optional[np.ndarray] = None,
                    cfg: ops.Config = ops.Config()) -> None:
    """
    Multi-pane viewer for one sequence. Use ←/→ to flip frames.
    Shows: RGB, Depth, Occupancy, XY scan, normalized laser scan, reprojection check.
    """
    num_frames = depth.shape[0]
    idx = [0]

    fig, axes = plt.subplots(1, 6, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle(f"Frame {idx[0] + 1}/{num_frames}, idx {idx[0]}", fontsize=14)

    img_rgb = axes[0].imshow(rgb[idx[0]])
    axes[0].set_title("RGB")

    img_depth = axes[1].imshow(depth[idx[0]], cmap='viridis')
    axes[1].set_title("Depth")

    _, _, occ_map = ops.depth_layer_proj(
        depth[idx[0]],
        rgb[idx[0]] if rgb is not None else None,
        height=height[idx[0]] if height is not None else None,
        cfg=cfg,
    )
    img_occ = axes[2].imshow(occ_map, cmap="gray", origin="lower")
    axes[2].set_title("Occ Map")

    # XY scan (cartesian points from angular bins)
    x, y, _, _ = ops.depth_layer_scan(
        depth[idx[0]],
        rgb=rgb[idx[0]] if rgb is not None else None,
        height=height[idx[0]] if height is not None else None,
        cfg=cfg,
    )
    axes[3].set_aspect('equal')
    scatter = axes[3].scatter(x, y, marker='o', s=6, alpha=0.9)
    axes[3].set_title("XY Scan")

    # Laser scan (angles + normalized dists)
    angles_deg, dists_norm = ops.depth_layer_scan_api(
        depth[idx[0]],
        rgb=rgb[idx[0]] if rgb is not None else None,
        height=height[idx[0]] if height is not None else None,
        fov_deg=cfg.sensor.fov_deg,
        dist_scale=cfg.sensor.dist_scale,
        rotate_points=cfg.transform.rotate_points,
        filter_points_=cfg.transform.filter_points,
        aggregation=cfg.laserscan.aggregation,
        n_intervals=cfg.laserscan.n_intervals,
        default_value=cfg.laserscan.default_value,
        coordinate_system=cfg.coordinate_system,
    )
    plot_laser_scan(axes[4], angles_deg, dists_norm)
    axes[4].set_title("Laser Scan (normalized)")

    # Reprojection check: Depth → PC → Depth
    pts, _ = ops.depth_to_filtered_pointcloud_api(
        depth[idx[0]],
        rgb=rgb[idx[0]] if rgb is not None else None,
        height=height[idx[0]] if height is not None else None,
        fov_deg=cfg.sensor.fov_deg,
        dist_scale=cfg.sensor.dist_scale,
        rotate_points=cfg.transform.rotate_points,
        coordinate_system=cfg.coordinate_system,
    )
    H, W = depth[idx[0]].shape
    dpt = ops.pointcloud_to_depth(
        pts, H, W,
        fov_deg=cfg.sensor.fov_deg,
        dist_scale=cfg.sensor.dist_scale,
        coordinate_system=cfg.coordinate_system,
        aggregation="zmin",
        radius=1,
        fill_value=0.0,
    )
    dpt_vis = np.where(dpt > 0, dpt, np.nan)
    img_dpt = axes[5].imshow(
        dpt_vis,
        cmap="viridis",
        vmin=np.nanmin(depth[idx[0]]),
        vmax=np.nanmax(depth[idx[0]]),
    )
    axes[5].set_title("Reprojected Depth")
    for ax in axes[:3]:
        ax.axis('off')

    # Key handler
    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % num_frames
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % num_frames
        else:
            return

        fig.suptitle(f"Frame {idx[0] + 1}/{num_frames}, idx {idx[0]}", fontsize=14)
        img_rgb.set_data(rgb[idx[0]])
        img_depth.set_data(depth[idx[0]])

        _, _, occ = ops.depth_layer_proj(
            depth[idx[0]],
            rgb[idx[0]] if rgb is not None else None,
            height=height[idx[0]] if height is not None else None,
            cfg=cfg,
        )
        img_occ.set_data(occ)

        x_i, y_i, _, _ = ops.depth_layer_scan(
            depth[idx[0]],
            rgb=rgb[idx[0]] if rgb is not None else None,
            height=height[idx[0]] if height is not None else None,
            cfg=cfg,
        )
        nonlocal scatter
        scatter.remove()
        scatter = axes[3].scatter(x_i, y_i, marker='o', s=6, alpha=0.9)

        ang_deg, dist_norm = ops.depth_layer_scan_api(
            depth[idx[0]],
            rgb=rgb[idx[0]] if rgb is not None else None,
            height=height[idx[0]] if height is not None else None,
            fov_deg=cfg.sensor.fov_deg,
            dist_scale=cfg.sensor.dist_scale,
            rotate_points=cfg.transform.rotate_points,
            filter_points_=cfg.transform.filter_points,
            aggregation=cfg.laserscan.aggregation,
            n_intervals=cfg.laserscan.n_intervals,
            default_value=cfg.laserscan.default_value,
            coordinate_system=cfg.coordinate_system,
        )
        plot_laser_scan(axes[4], ang_deg, dist_norm)

        pts_i, _ = ops.depth_to_filtered_pointcloud_api(
            depth[idx[0]],
            rgb=rgb[idx[0]] if rgb is not None else None,
            # height=height[idx[0]] if height is not None else None,
            fov_deg=cfg.sensor.fov_deg,
            dist_scale=cfg.sensor.dist_scale,
            rotate_points=cfg.transform.rotate_points,
            coordinate_system=cfg.coordinate_system,
        )
        H, W = depth[idx[0]].shape
        dpt_i = ops.pointcloud_to_depth(
            pts_i, H, W,
            fov_deg=cfg.sensor.fov_deg,
            dist_scale=cfg.sensor.dist_scale,
            coordinate_system=cfg.coordinate_system,
            aggregation="zmin",
            radius=1,
            fill_value=0.0,
        )
        dpt_vis_i = np.where(dpt_i > 0, dpt_i, np.nan)
        img_dpt.set_data(dpt_vis_i)
        img_dpt.set_clim(vmin=np.nanmin(depth[idx[0]]), vmax=np.nanmax(depth[idx[0]]))

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# -----------------------------------------------------------------------------
# Optional interactive Open3D viewer
# -----------------------------------------------------------------------------

class PointCloudViewer:
    """
    Minimal Open3D viewer showing filtered points with a Y-range slider and frame slider.
    """
    def __init__(self, depths, rgbs=None, heights=None, cfg: ops.Config = ops.Config()):
        try:
            import open3d as o3d
        except Exception as e:
            raise RuntimeError("Open3D import failed. Please install open3d to use this viewer.") from e

        self.depths = depths
        self.rgbs = rgbs
        self.heights = heights
        self.cfg = cfg
        self.num = len(depths)
        self.cur = 0

        pts, col = ops.depth_to_filtered_pointcloud(depths[0],
                                                    rgb=(rgbs[0] if rgbs is not None else None),
                                                    height=(heights[0] if heights is not None else None),
                                                    cfg=cfg)
        self.pts_orig = pts
        self.col_orig = col
        self.min_y = float(np.min(pts[:, 1])); self.max_y = float(np.max(pts[:, 1]))

        gui = o3d.visualization.gui
        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window("Depth Viewer", 1024, 768)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        self.panel = gui.Vert()
        self.window.add_child(self.panel)

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.pts_orig)
        if self.col_orig is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(self.col_orig)

        mat = o3d.visualization.rendering.MaterialRecord()
        self.scene_widget.scene.add_geometry("points", self.pcd, mat)

        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

        # Axes frame
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.scene_widget.scene.add_geometry("axis", self.axis, mat)

        # Sliders
        self.min_slider = gui.Slider(gui.Slider.DOUBLE)
        self.min_slider.set_limits(self.min_y, self.max_y)
        self.min_slider.double_value = self.min_y
        self.min_slider.set_on_value_changed(self._on_min)

        self.max_slider = gui.Slider(gui.Slider.DOUBLE)
        self.max_slider.set_limits(self.min_y, self.max_y)
        self.max_slider.double_value = self.max_y
        self.max_slider.set_on_value_changed(self._on_max)

        self.frame_slider = gui.Slider(gui.Slider.INT)
        self.frame_slider.set_limits(0, self.num - 1)
        self.frame_slider.int_value = 0
        self.frame_slider.set_on_value_changed(self._on_frame)

        self.panel.add_child(gui.Label("Min Y")); self.panel.add_child(self.min_slider)
        self.panel.add_child(gui.Label("Max Y")); self.panel.add_child(self.max_slider)
        self.panel.add_child(gui.Label("Frame")); self.panel.add_child(self.frame_slider)

        self.window.set_on_layout(self._on_layout)
        app.run()

    # ---- Open3D callbacks ----
    def _on_layout(self, ctx):
        import open3d as o3d
        r = self.window.content_rect
        panel_w = 220
        self.panel.frame = o3d.visualization.gui.Rect(r.x, r.y, panel_w, r.height)
        self.scene_widget.frame = o3d.visualization.gui.Rect(r.x + panel_w, r.y, r.width - panel_w, r.height)

    def _on_min(self, val): self._update_range(min_y=val)
    def _on_max(self, val): self._update_range(max_y=val)

    def _on_frame(self, val: int):
        self.cur = int(val)
        rgb = self.rgbs[self.cur] if self.rgbs is not None else None
        pts, col = ops.depth_to_filtered_pointcloud(
            self.depths[self.cur],
            rgb=rgb,
            height=self.heights[self.cur] if self.heights is not None else None,
            cfg=self.cfg,
        )
        self.pts_orig = pts; self.col_orig = col
        self.min_y = float(np.min(pts[:, 1])); self.max_y = float(np.max(pts[:, 1]))
        self.min_slider.set_limits(self.min_y, self.max_y)
        self.max_slider.set_limits(self.min_y, self.max_y)
        self.min_slider.double_value = self.min_y
        self.max_slider.double_value = self.max_y
        self._update_range()

    def _update_range(self, min_y: Optional[float] = None, max_y: Optional[float] = None):
        import open3d as o3d
        if min_y is not None: self.min_y = float(min_y)
        if max_y is not None: self.max_y = float(max_y)
        mask = (self.pts_orig[:, 1] >= self.min_y) & (self.pts_orig[:, 1] <= self.max_y)
        pts = self.pts_orig[mask]
        self.pcd.points = o3d.utility.Vector3dVector(pts)
        if self.col_orig is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(self.col_orig[mask])
        self.scene_widget.scene.clear_geometry()
        mat = o3d.visualization.rendering.MaterialRecord()
        self.scene_widget.scene.add_geometry("points", self.pcd, mat)
        self.scene_widget.scene.add_geometry("axis", self.axis, mat)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Depth/PointCloud visualization")
    parser.add_argument("--cfg", type=str, default="./depth_transform.yaml", help="YAML config path")
    parser.add_argument("--data", type=str, default="./data/data_batch_2.npz", help="NPZ path containing 'frame' and 'depth'")
    parser.add_argument("--mode", type=str, choices=["viewer", "plot"], default="plot",
                        help="viewer: Open3D UI; plot: Matplotlib figure")
    args = parser.parse_args()

    cfg = ops.Config.from_yaml(args.cfg)
    print("Loaded config:")
    print(cfg)

    data = np.load(args.data, mmap_mode='r')
    frames = data['frame']        # (T, H, W, 3)
    depths = data['depth']        # (T, H, W)
    heights = data['height'] if 'height' in data.keys() else None
    print(f"load heights: {heights}")

    if args.mode == 'viewer':
        PointCloudViewer(depths, frames, heights, cfg=cfg)
    else:
        plot_data_frame(frames, depths, heights, cfg=cfg)


if __name__ == "__main__":
    main()
