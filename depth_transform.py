import yaml
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any

class Config:
    def __init__(self,
                 sensor_cfg: Dict[str, Any] = None,
                 transform_cfg: Dict[str, Any] = None,
                 projection_cfg: Dict[str, Any] = None):
        # 默认参数
        self.sensor_cfg = sensor_cfg or {
            "fov_deg": [90.0, 90.0],
            "dist_scale": 10.0,
        }

        self.transform_cfg = transform_cfg or {
            "rotate_points": [['z', -30]],      # 旋转30°，Z轴
            "filter_points": [['y', -0.25, 0.25]],  # 过滤 Y 轴范围
        }

        self.projection_cfg = projection_cfg or {
            "map_resolution": 0.2,
            "map_size": 100,
        }

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """ 从 YAML 文件加载配置 """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            sensor_cfg=data.get("sensor_cfg", None),
            transform_cfg=data.get("transform_cfg", None),
            projection_cfg=data.get("projection_cfg", None)
        )

    def to_dict(self) -> Dict[str, Any]:
        """ 导出字典形式，方便调试或打印 """
        return {
            "sensor_cfg": self.sensor_cfg,
            "transform_cfg": self.transform_cfg,
            "projection_cfg": self.projection_cfg
        }

    def __repr__(self):
        return yaml.dump(self.to_dict(), allow_unicode=True, sort_keys=False)


def ratate_points(points, rotates:None):
    """
    左手坐标系 SO(3) 旋转矩阵
    pts: 点云坐标, (N, 3) ndarray
    axis: 旋转轴，形如 'x'/'y'/'z' 或 3 元 numpy 向量
    theta_deg: 旋转角度（单位：度，正方向为左手系规则）
    """
    R = np.eye(3)
    if rotates is not None:
        for axis, theta_deg in rotates:
            # 如果是字符，转成向量
            if isinstance(axis, str):
                axis = axis.lower()
                if axis == 'x':
                    axis = np.array([1.0, 0.0, 0.0])
                elif axis == 'y':
                    axis = np.array([0.0, 1.0, 0.0])
                elif axis == 'z':
                    axis = np.array([0.0, 0.0, 1.0])
                else:
                    raise ValueError("axis 必须是 'x', 'y', 'z' 或一个 3 元向量")
            else:
                axis = np.asarray(axis, dtype=float)

            # 归一化
            axis = axis / np.linalg.norm(axis)
            # 左手系相当于右手系角度取负
            theta = np.deg2rad(-theta_deg)

            # 叉乘矩阵
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = (np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)) @ R
    return points @ R.T

def filter_points(points, colors, filters=None):
    """
    多轴联合范围过滤

    参数:
        points: (N, 3) ndarray 点云
        colors: (N, 3) ndarray 颜色
        filters: [dict 或 list/tuple, 指定每个轴的范围]
            - 如果是 dict: {axis: (min, max)}
            - 如果是 list/tuple: [(axis, min, max), ...]

    返回:
        过滤后的 (points, colors)
    """
    mask = np.ones(points.shape[0], dtype=bool)

    if filters is None:
        return points, colors

    if isinstance(filters, dict):
        items = filters.items()
    else:
        items = filters

    for item in items:
        if isinstance(item, list) and len(item) == 3:
            axis, min_val, max_val = item
        else:  # dict 形式
            axis, (min_val, max_val) = item

        if isinstance(axis, str):
            axis = axis.lower()
            if axis == 'x':
                axis = 0
            elif axis == 'y':
                axis = 1
            elif axis == 'z':
                axis = 2
            else:
                raise ValueError("axis 必须是 'x', 'y', 'z' 或 0, 1, 2")
        if min_val is not None:
            mask &= points[:, axis] >= min_val
        if max_val is not None:
            mask &= points[:, axis] <= max_val

    return points[mask], colors[mask]

def depth_transform(pts, color, cfg: Config):
    """
    点云变换
    """
    return filter_points(
        ratate_points(pts, cfg.transform_cfg['rotate_points']), 
        color, 
        cfg.transform_cfg['filter_points'])

def depth_to_pointcloud(depth, 
                        rgb=None, 
                        height=None,
                        cfg: Config=Config()):
    """
    转换到相机坐标系下：X右为正， Y上为正，Z前为正
    """
    H, W = depth.shape
    fov_x, fov_y = np.deg2rad(cfg.sensor_cfg['fov_deg'][0]), np.deg2rad(cfg.sensor_cfg['fov_deg'][1])
    cx, cy = (W-1)/2.0, (H-1)/2.0
    fx = W / (2*np.tan(fov_x/2))
    fy = H / (2*np.tan(fov_y/2))
    Z = depth.astype(np.float32) * cfg.sensor_cfg['dist_scale']
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts = np.stack([X, -Y, -Z], axis=-1).reshape(-1,3)
    mask = Z.reshape(-1) > 0
    pts = pts[mask]
    color = None
    if rgb is not None:
        # 假设 rgb 是 HxWx3
        color = rgb.reshape(-1,3)[mask] / 255.0  # 转为 [0,1] 浮点
    if height is not None:
        cfg.transform_cfg['filter_points'].append(['y', [-height, None]])
    return depth_transform(pts, color, cfg)

def depth_layer_proj(depth, 
                     rgb = None, 
                     height = None,
                     cfg: Config = Config()):
    """
    传入点云：Nx3, 投影到二维平面 (X-Y)，考虑遮挡效应，生成占用栅格地图。
    
    参数:
        depth: np.ndarray
        color: np.ndarray
        resolution: float, 栅格分辨率 (米/格)
        size: int, 栅格大小 (size x size)
    
    返回:
        layer: np.ndarray, 过滤的点云
        color: np.ndarray, 点云对应的颜色
        occ_map: np.ndarray, 0-1 占用地图 (size x size)
    """
    layer, color = depth_to_pointcloud(depth, rgb=rgb, height=height, cfg=cfg)
    size = cfg.projection_cfg['map_size']
    resolution = cfg.projection_cfg['map_resolution']
    # 初始化占用地图和深度图
    occ_map = np.zeros((size, size), dtype=np.uint8)
    depth_map = np.full((size, size), np.inf)

    # 将点云转换为栅格坐标 
    # 假设地图中心是 (0,0)，所以要平移到正坐标
    half = size // 2
    grid_x = np.floor(-layer[:, 2] / resolution).astype(int) + half
    grid_y = np.floor(layer[:, 0] / resolution).astype(int) + half

    # 过滤掉超出范围的点
    valid_mask = (grid_x >= 0) & (grid_x < size) & (grid_y >= 0) & (grid_y < size)
    grid_x, grid_y, z = grid_x[valid_mask], grid_y[valid_mask], -layer[valid_mask, 1]

    # 遍历点云，更新占用栅格（只取最近点，模拟遮挡效应）
    for gx, gy, gz in zip(grid_x, grid_y, z):
        if gz < depth_map[gx, gy]:
            depth_map[gx, gy] = gz
            occ_map[gx, gy] = 1

    return layer, color, occ_map

# ---------------------- 可视化 ----------------------
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])

class PointCloudFilterApp:
    def __init__(self, depths, colors=None, heights=None, cfg: Config=Config()):
        self.depths = depths
        self.colors = colors
        self.heights = heights
        if self.colors is not None:
            assert len(depths) == len(colors)
        if self.heights is not None:
            assert len(depths) == len(heights)
        self.num_frames = len(depths)
        self.cfg = cfg
        self.current_frame = 0

        self.points, self.color = depth_to_pointcloud(
            depth=depths[0],
            rgb=colors[0] if colors is not None else None,
            height=heights[0] if heights is not None else None,
            cfg=cfg
        )
        self.points_orig = self.points.copy()
        self.colors_orig = np.array(self.color, copy=True) if self.colors is not None else None
        self.min_y = float(np.min(self.points[:, 1]))
        self.max_y = float(np.max(self.points[:, 1]))

        gui_app = o3d.visualization.gui.Application.instance
        gui_app.initialize()
        self.window = gui_app.create_window("Depth Viewer", 1024, 768)

        self.scene_widget = o3d.visualization.gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        self.panel = o3d.visualization.gui.Vert()
        self.window.add_child(self.panel)

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.colors is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(self.color)

        material = o3d.visualization.rendering.MaterialRecord()
        self.scene_widget.scene.add_geometry("points", self.pcd, material)

        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

        # 添加小坐标轴
        self.axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.scene_widget.scene.add_geometry("axis", self.axis_frame, material)

        # 滑条
        self.min_slider = o3d.visualization.gui.Slider(o3d.visualization.gui.Slider.DOUBLE)
        self.min_slider.set_limits(self.min_y, self.max_y)
        self.min_slider.double_value = self.min_y
        self.min_slider.set_on_value_changed(self.on_min_slider)
        self.panel.add_child(o3d.visualization.gui.Label("Min Y"))
        self.panel.add_child(self.min_slider)

        self.max_slider = o3d.visualization.gui.Slider(o3d.visualization.gui.Slider.DOUBLE)
        self.max_slider.set_limits(self.min_y, self.max_y)
        self.max_slider.double_value = self.max_y
        self.max_slider.set_on_value_changed(self.on_max_slider)
        self.panel.add_child(o3d.visualization.gui.Label("Max Y"))
        self.panel.add_child(self.max_slider)

        self.frame_slider = o3d.visualization.gui.Slider(o3d.visualization.gui.Slider.INT)
        self.frame_slider.set_limits(0, self.num_frames-1)
        self.frame_slider.int_value = 0
        self.frame_slider.set_on_value_changed(self.on_frame_slider)
        self.panel.add_child(o3d.visualization.gui.Label("Frame"))
        self.panel.add_child(self.frame_slider)

        self.panel.add_child(o3d.visualization.gui.Label("Camera Rotation (deg):"))
        self.rotation_label = o3d.visualization.gui.Label("X=0, Y=0, Z=0")
        self.panel.add_child(self.rotation_label)

        # 定时刷新
        self.window.set_on_tick_event(self.update_camera_info)
        self.window.set_on_layout(self._on_layout)
        gui_app.run()

    def update_camera_info(self):
        cam = self.scene_widget.scene.camera
        extrinsic = np.asarray(cam.get_model_matrix())
        R = extrinsic[:3, :3]
        euler_deg = rotation_matrix_to_euler_angles(R)
        self.rotation_label.text = f"X={euler_deg[0]:.1f}, Y={euler_deg[1]:.1f}, Z={euler_deg[2]:.1f}"

        # 同步小坐标轴姿态（放到原点并旋转）
        self.axis_frame.transform(np.eye(4))  # 重置
        self.axis_frame.rotate(R, center=(0, 0, 0))
        self.axis_frame.translate(np.array([-1., -1., 0]))  # 让它离点云远一些
        return True

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 200
        self.panel.frame = o3d.visualization.gui.Rect(r.x, r.y, panel_width, r.height)
        self.scene_widget.frame = o3d.visualization.gui.Rect(r.x + panel_width, r.y,
                                                             r.width - panel_width, r.height)

    def on_min_slider(self, value):
        self.min_y = value
        self.update_points()

    def on_max_slider(self, value):
        self.max_y = value
        self.update_points()

    def on_frame_slider(self, value):
        self.current_frame = int(value)
        rgb = self.colors[self.current_frame] if self.colors is not None else None
        self.points, self.color = depth_to_pointcloud(
            self.depths[self.current_frame],
            rgb=rgb, 
            height=self.heights[self.current_frame] if self.heights is not None else None,
            cfg=self.cfg
        )
        self.points_orig = self.points.copy()
        self.colors_orig = np.array(self.color, copy=True) if self.colors is not None else None
        self.min_y = float(np.min(self.points[:,1]))
        self.max_y = float(np.max(self.points[:,1]))
        self.min_slider.set_limits(self.min_y, self.max_y)
        self.max_slider.set_limits(self.min_y, self.max_y)
        self.min_slider.double_value = self.min_y
        self.max_slider.double_value = self.max_y
        self.update_points()

    def update_points(self):
        mask = (self.points_orig[:, 1] >= self.min_y) & (self.points_orig[:, 1] <= self.max_y)
        filtered_points = self.points_orig[mask]
        self.pcd.points = o3d.utility.Vector3dVector(filtered_points)
        if self.colors_orig is not None:
            filtered_colors = self.colors_orig[mask]
            self.pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        self.scene_widget.scene.clear_geometry()
        material = o3d.visualization.rendering.MaterialRecord()
        self.scene_widget.scene.add_geometry("points", self.pcd, material)
        # 重新添加小坐标轴
        self.scene_widget.scene.add_geometry("axis", self.axis_frame, material)

def plot_data_frame(rgb, depth, obj_attention, height=None, cfg: Config=Config()):
    num_frames = rgb.shape[0]
    idx = [0]  # 用列表封装，方便在内部修改

    fig, axes = plt.subplots(2, 2, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle(f"Frame {idx[0] + 1}/{num_frames}", fontsize=16)

    # 初始化三张图
    img_rgb = axes[0, 0].imshow(rgb[idx[0]])
    axes[0, 0].set_title("RGB")
    img_depth = axes[0, 1].imshow(depth[idx[0]], cmap='viridis')
    axes[0, 1].set_title("Depth")
    img_attn = axes[1, 0].imshow(obj_attention[idx[0]], cmap='hot')
    axes[1, 0].set_title("Object Attention")

    _, _, occ_map = depth_layer_proj(depth[idx[0]], 
                                     rgb[idx[0]], 
                                     height=height[idx[0]] if height is not None else None,
                                     cfg=cfg)
    img_occ = axes[1, 1].imshow(occ_map, cmap="gray", origin="lower")
    axes[1, 1].set_title("Occ Map")
 
    for ax in axes.flatten():
        ax.axis('off')

    # 按键事件
    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % num_frames
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % num_frames

        # 更新图像
        img_rgb.set_data(rgb[idx[0]])
        img_depth.set_data(depth[idx[0]])
        img_attn.set_data(obj_attention[idx[0]])

        _, _, occ_map = depth_layer_proj(depth[idx[0]], rgb[idx[0]], cfg=cfg)
        img_occ.set_data(occ_map)

        fig.suptitle(f"Frame {idx[0] + 1}/{num_frames}", fontsize=16)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# ---------------------- 运行调试 ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云可视化与过滤工具")
    parser.add_argument("--cfg", type=str,help="参数配置文件路径（.yaml)")
    parser.add_argument("--data", type=str,
                        help="输入数据文件路径 (.npz)")
    parser.add_argument("--mode", type=str, choices=["viewer", "plot"], default="app",
                        help="运行模式: 'app' 显示点云交互界面, 'plot' 绘制图像结果")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.cfg)
    data = np.load(args.data, mmap_mode='r')
    frames = data['frame']        # (50, 300, 300, 3)
    depths = data['depth']        # (50, 300, 300)
    attn   = data['obj_attention']# (50, 300, 300)
    heights = data['height']     

    if args.mode == 'viewer':
        app = PointCloudFilterApp(depths, frames,  heights, cfg=cfg)
    elif args.mode == 'plot':
        plot_data_frame(frames, depths, attn, heights, cfg=cfg)


