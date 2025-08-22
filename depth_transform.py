import yaml
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any

class Config:
    def __init__(self,
                 corrdinate_system: str = 'opengl',
                 # 默认配置
                 sensor_cfg: Dict[str, Any] = None,
                 transform_cfg: Dict[str, Any] = None,
                 projection_cfg: Dict[str, Any] = None,
                 laserscan_cfg: Dict[str, Any] = None):
        self.coordinate_system = corrdinate_system.lower()
        if self.coordinate_system not in ['opengl', 'opencv']:
            raise ValueError("coordinate_system 必须是 'opengl' 或 'opencv'")
        # 默认参数
        self.sensor_cfg = sensor_cfg or {
            "fov_deg": [90.0, 90.0],
            "dist_scale": 1.0,
        }

        self.transform_cfg = transform_cfg or {
            "rotate_points": [['x', -30]],      # 旋转30°
            "filter_points": [['y', -0.25, 0.25]],  # 过滤 Y 轴范围
        }

        self.projection_cfg = projection_cfg or {
            "map_resolution": 0.2,
            "map_size": 100,
        }
        self.laserscan_cfg = laserscan_cfg or {
            "aggregation": 'mean',  # 'mean', 'max', 'min', 'all'
            "n_intervals": 30,  # 深度扫描的区间数
            "default_value": 3,  # 空区间的默认值
        }

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """ 从 YAML 文件加载配置 """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            sensor_cfg=data.get("sensor_cfg", None),
            transform_cfg=data.get("transform_cfg", None),
            projection_cfg=data.get("projection_cfg", None),
            laserscan_cfg=data.get("laserscan_cfg", None),
        )

def depth_to_pointcloud(depth,
                        fov_deg=[90.0, 90.0],
                        dist_scale=1.0,
                        coordinate_system='opengl'):
    """ 将深度图转换为点云
    参数:
        depth: HxW 深度图
        fov_deg: 相机水平和垂直视场角 (默认 [90.0, 90.0])
        dist_scale: 深度缩放比例 (默认 1.0)
        coordinate_system: 坐标系 ('opengl' 或 'opencv', 默认 'opengl')
    返回:
        pts: 点云坐标 (HxW, 3) ndarray
    """
    H, W = depth.shape
    fov_x, fov_y = np.deg2rad(fov_deg[0]), np.deg2rad(fov_deg[1])
    # 计算内参
    fx = W / (2 * np.tan(fov_x / 2))
    fy = H / (2 * np.tan(fov_y / 2))
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    
    # 应用深度缩放
    Z = depth.astype(np.float32) * dist_scale
    
    # 创建像素坐标网格
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    
    # 转换到相机坐标系
    X = (uu - cx) * Z / fx
    
    if coordinate_system == 'opencv':
        # OpenCV坐标系：Y向下
        Y = (vv - cy) * Z / fy
    elif coordinate_system == 'opengl':
        # OpenGL坐标系：Y向上
        Y = -(vv - cy) * Z / fy
    else:
        raise ValueError("coordinate_system 必须是 'opencv' 或 'opengl'")
    # 组合点云
    pts = np.stack([X, Y, -Z], axis=-1)

    return pts

def rotate_points(points, rotates=None):
    """
    旋转点云 - 使用标准的旋转矩阵
    points: 点云坐标, (N, 3) ndarray
    rotates: 旋转参数列表 [(axis, angle_deg), ...]
    """
    if rotates is None or len(rotates) == 0:
        return points
    
    R = np.eye(3)
    
    for axis, theta_deg in rotates:
        theta = np.deg2rad(theta_deg)
        
        if isinstance(axis, str):
            axis = axis.lower()
            if axis == 'x':
                # 绕X轴旋转
                R_axis = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]
                ])
            elif axis == 'y':
                # 绕Y轴旋转
                R_axis = np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])
            elif axis == 'z':
                # 绕Z轴旋转
                R_axis = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
            else:
                raise ValueError("axis 必须是 'x', 'y', 'z'")
        else:
            # 使用Rodrigues公式处理任意轴旋转
            axis = np.asarray(axis, dtype=float)
            axis = axis / np.linalg.norm(axis)
            
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_axis = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        
        R = R_axis @ R
    
    return points @ R.T

def filter_points(points, colors, filters=None):
    """
    多轴联合范围过滤
    """
    # print(f"filters:{filters}")
    if filters is None or len(filters) == 0:
        return points, colors
        
    mask = np.ones(points.shape[0], dtype=bool)

    for item in filters:
        if isinstance(item, list) and len(item) == 3:
            axis, min_val, max_val = item
        else:
            raise ValueError("过滤项必须是 [axis, min_val, max_val] 格式")
        if isinstance(axis, str):
            axis = axis.lower()
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        else:
            axis_idx = axis

        if min_val is not None:
            mask &= points[:, axis_idx] >= min_val
        if max_val is not None:
            mask &= points[:, axis_idx] <= max_val

    return points[mask], colors[mask] if colors is not None else None                        

def depth_to_filted_pointcloud(depth, 
                       rgb=None, 
                       height=None,
                       cfg: Config = None):
    """
    深度图转换为点云
    
    参数:
        depth: HxW 深度图
        rgb: HxWx3 RGB图像 (可选)
        height: 相机高度，用于地面过滤 (可选)
        cfg: 配置对象
    """
    if cfg is None:
        cfg = Config()
    pts_ori = depth_to_pointcloud(depth,
                              fov_deg=cfg.sensor_cfg['fov_deg'],
                              dist_scale=cfg.sensor_cfg['dist_scale'],
                              coordinate_system=cfg.coordinate_system)
    pts = pts_ori.copy().reshape(-1, 3)  # (H*W, 3)
    mask = pts[:, 2] <= 0  # 过滤掉深度大于0的点
    pts = pts[mask]  # (有效点数, 3)
    
    # 处理颜色
    color = None
    if rgb is not None:
        color = rgb.reshape(-1, 3)[mask] / 255.0
    
    # 应用变换
    if 'rotate_points' in cfg.transform_cfg:
        pts = rotate_points(pts, cfg.transform_cfg['rotate_points'])
    
    if 'filter_points' in cfg.transform_cfg:
        pts, color = filter_points(pts, color, cfg.transform_cfg['filter_points'])

    # 根据相机高度添加地面过滤
    if height is not None:
        if cfg.coordinate_system == 'opengl':# X对应深度
            # Y向上的坐标系，地面在-height附近
            ground_filter = ['y', -height, None]
        else:
            # Y向下的坐标系，地面在height附近
            ground_filter = ['y', None, height]
        
        pts, color = filter_points(pts, color, [ground_filter])
    
    return pts, color

def map_pts_to_intervals(pts, 
                         fov_deg=[90.0, 90.0],
                         n_intervals=30, 
                         default_value=3.0, 
                         aggregation='min'):
    """
    高级版本：支持多种聚合方法
    
    Parameters:
    -----------
    aggregation : str
        聚合方法: 'first', 'last', 'mean', 'median', 'sum', 'count', 'all', 'min'
    """
    
    

    # x = pts[:, 0][pts[:, 0] < 0]  # 只考虑X<0的点
    # y = pts[:, 2][pts[:, 0] < 0]  # 对应的Y值
    # angles = np.atan2(np.abs(y), x)  # 计算每个点的角度
    # print("y:", y)
    # print("x:", x)
    # print("angles x<0:", angles)
    # print("max_angle:", np.rad2deg(np.max(angles)), "min_angle:", np.rad2deg(np.min(angles)))
    # x = pts[:, 0][pts[:, 0] > 0]  # 只考虑X<0的点
    # y = pts[:, 2][pts[:, 0] > 0]  # 对应的Y值
    # angles = np.atan2(np.abs(y), x)  # 计算每个点的角度
    # print("y:", y)
    # print("x:", x)
    # print("angles x>0:", angles)
    # print("max_angle:", np.rad2deg(np.max(angles)), "min_angle:", np.rad2deg(np.min(angles)))
    angle = np.atan2(np.abs(pts[:, 2]), pts[:, 0])  # 计算每个点的角度，弧度
    angle_boundaries = np.linspace(np.deg2rad(fov_deg[0] / 2), np.deg2rad(fov_deg[0]*1.5), n_intervals + 1)
    angle_intervals, dist_intervals = [], []
    for i in range(n_intervals):
        start = angle_boundaries[i]
        end = angle_boundaries[i + 1]
        angle_intervals.append([start, end])
        if i == n_intervals - 1:
            mask = (angle >= start) & (angle <= end)
        else:
            mask = (angle >= start) & (angle < end)
        pts_in_interval = pts[mask]
        if len(pts_in_interval) > 0:
            if aggregation == 'min':
                dist_intervals.append(np.min(np.sqrt(pts_in_interval[:, 0]**2 + pts_in_interval[:, 2]**2)))
            elif aggregation == 'max':
                dist_intervals.append(np.max(np.sqrt(pts_in_interval[:, 0]**2 + pts_in_interval[:, 2]**2)))
            elif aggregation == 'mean':
                dist_intervals.append(np.mean(np.sqrt(pts_in_interval[:, 0]**2 + pts_in_interval[:, 2]**2)))
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")
        else:
            dist_intervals.append(default_value)
    return np.array(angle_intervals), np.array(dist_intervals)
    
    
    # min_x = np.min(pts[:, 0])
    # max_x = np.max(pts[:, 0])

    # boundaries = np.linspace(min_x, max_x, n_intervals + 1)
    # result_values = []
    # intervals = []
    # for i in range(n_intervals):
    #     start = boundaries[i]
    #     end = boundaries[i + 1]
    #     intervals.append((start, end))
        
    #     if i == n_intervals - 1:
    #         mask = (pts[:, 0] >= start) & (pts[:, 0] <= end)
    #     else:
    #         mask = (pts[:, 0] >= start) & (pts[:, 0] < end)
        
    #     values_in_interval = pts[:, 2][mask]
        
    #     if len(values_in_interval) > 0:
    #         if aggregation == 'first':
    #             result_values.append(values_in_interval[0])
    #         elif aggregation == 'last':
    #             result_values.append(values_in_interval[-1])
    #         elif aggregation == 'mean':
    #             result_values.append(np.mean(values_in_interval))
    #         elif aggregation == 'median':
    #             result_values.append(np.median(values_in_interval))
    #         elif aggregation == 'sum':
    #             result_values.append(np.sum(values_in_interval))
    #         elif aggregation == 'count':
    #             result_values.append(len(values_in_interval))
    #         elif aggregation == 'all':
    #             result_values.append(values_in_interval.tolist())
    #         elif aggregation == 'min':
    #             print(f"Using min aggregation for interval {i}: {start} to {end}")
    #             result_values.append(np.min(values_in_interval))
    #         else:
    #             raise ValueError(f"Unsupported aggregation method: {aggregation}")
    #     else:
    #         if aggregation == 'all':
    #             result_values.append([])
    #         else:
    #             result_values.append(default_value)
    
    # if aggregation == 'all':
    #     return intervals, result_values
    # else:
    #     return intervals, np.array(result_values)

def depth_layer_scan(depth,
                    rgb=None,
                    height=None,
                    cfg: Config = None):
    if cfg is None:
        cfg = Config()
    # 转换为点云
    pts, _ = depth_to_filted_pointcloud(depth,
                                        rgb=rgb, 
                                        height=height,
                                        cfg=cfg)
    if pts.shape[0] == 0: # 过滤后点云为空，则直接返回None
        return None, None
    # min_x = np.min(pts[:, 0])
    # max_x = np.max(pts[:, 0])
    # print(f"Depth scan range: X [{min_x:.2f}, {max_x:.2f}]")
    # min_y = np.min(pts[:, 1])
    # max_y = np.max(pts[:, 1])
    # print(f"Depth scan range: Y [{min_y:.2f}, {max_y:.2f}]")
    # intervals, y_coord = map_pts_to_intervals(
    #                         pts,
    #                         fov_deg=cfg.sensor_cfg['fov_deg'],
    #                         n_intervals=cfg.laserscan_cfg['n_intervals'], 
    #                         default_value=cfg.laserscan_cfg['default_value'], 
    #                         aggregation=cfg.laserscan_cfg['aggregation']
    #                     )
    # start = np.array([interval[0] for interval in intervals])
    # end = np.array([interval[1] for interval in intervals])
    # x_coord = (start + end) / 2.0  # 中心点
    # y_coord = np.abs(y_coord)  # 取绝对值
    # return x_coord, y_coord, _, _
    angles_intervals, dist = map_pts_to_intervals(
                            pts,    
                            fov_deg=cfg.sensor_cfg['fov_deg'],
                            n_intervals=cfg.laserscan_cfg['n_intervals'],
                            default_value=cfg.laserscan_cfg['default_value'],
                            aggregation=cfg.laserscan_cfg['aggregation']
                        )
    angles = (angles_intervals[:, 0] + angles_intervals[:, 1]) / 2.0
    dist[dist > cfg.laserscan_cfg['default_value']] = cfg.laserscan_cfg['default_value']  # 限制最大距离
    x_coord = dist * np.cos(angles)  # X坐标
    y_coord = dist * np.sin(angles)  # Y坐标
    # print(f"x_coord: {x_coord}")
    # print(f"y_coord: {y_coord}")
    return x_coord, y_coord, angles, dist
    

def depth_layer_scan_api(depth,
                         rgb=None,
                         height=None,
                         fov_deg=[90.0, 90.0],
                         dist_scale=1.0,
                         rotate_points=[['x', -30]],
                         filter_points=[['y', -0.25, 0.25]],
                         aggregation='min',
                         n_intervals=30,
                         default_value=3.0):
    """ 深度图投影到激光扫描 - API接口
    参数:   
        depth: HxW 深度图
        rgb: HxWx3 RGB图像 (可选)
        height: 相机高度，用于地面过滤 (可选)
        n_intervals: 深度扫描的区间数 (默认 30)
        default_value: 空区间的默认值 (默认 3) 
    returns:
        angles: 激光扫描角度 (n_intervals,) ndarray
        dists: 激光扫描距离 (n_intervals,) ndarray
    """
    # angles = np.linspace(fov_deg[0]/2, -fov_deg[0]/2, n_intervals)
    cfg = Config(
        corrdinate_system='opengl',
        sensor_cfg={"fov_deg": fov_deg, "dist_scale": dist_scale},
        transform_cfg={"rotate_points": rotate_points, "filter_points": filter_points},
        laserscan_cfg={"aggregation": aggregation, "n_intervals": n_intervals, "default_value": default_value}
    )
    x, y, angles, dist = depth_layer_scan(depth, rgb=rgb, height=height, cfg=cfg)
    if x is not None and y is not None:
        dist = dist / default_value # 归一化距离
    else: # 过滤点云为None，则直接返回最大距离
        dist = np.ones_like(angles)
    return np.rad2deg(angles), dist



def depth_layer_proj(depth, 
                    rgb=None, 
                    height=None,
                    cfg: Config = None):
    """
    深度图投影到占用栅格地图
    """
    if cfg is None:
        cfg = Config()
    
    # 转换为点云
    layer, color = depth_to_filted_pointcloud(depth, rgb=rgb, height=height, cfg=cfg)
    depth_layer_scan(depth, rgb=rgb, height=height, cfg=cfg)
    
    size = cfg.projection_cfg['map_size']
    resolution = cfg.projection_cfg['map_resolution']
    
    # 初始化占用地图和深度图
    occ_map = np.zeros((size, size), dtype=np.uint8)
    depth_map = np.full((size, size), np.inf)

    # 将点云转换为栅格坐标
    half = size // 2
    grid_x = -np.floor(layer[:, 2] / resolution).astype(int) + half  # X对应栅格X
    grid_y = np.floor(layer[:, 0] / resolution).astype(int) + half  # Z对应栅格Y

    # 过滤掉超出范围的点
    valid_mask = (grid_x >= 0) & (grid_x < size) & (grid_y >= 0) & (grid_y < size)
    grid_x, grid_y, z = grid_x[valid_mask], grid_y[valid_mask], layer[valid_mask, 1]  # 使用Y作为高度

    # 遍历点云，更新占用栅格（只取最近点，模拟遮挡效应）
    for gx, gy, gz in zip(grid_x, grid_y, z):
        if gz < depth_map[gx, gy]:
            depth_map[gx, gy] = gz
            occ_map[gx, gy] = 1

    return layer, color, occ_map

# For development and testing in RL
def depth_layer_proj_api(
    depth, 
    rgb=None, 
    height=None, 
    fov_deg=[90.0, 90.0], # 默认90度视场角
    dist_scale=1.0, # 默认深度缩放比例1.0
    rotate_points=[['x', -30]],  # 默认旋转30°
    filter_points=[['y', -0.25, 0.25]],  # 默认过滤Y轴范围上下25cm
    map_resolution=0.2, # 默认地图分辨率20cm
    map_size=100, # 默认地图大小100x100
    coordinate_system='opengl'):
    """    深度图投影到占用栅格地图 - API接口
    参数:
        depth: HxW 深度图
        rgb: HxWx3 RGB图像 (可选)
        height: 相机高度，用于地面过滤 (可选)
        fov_deg: 相机水平和垂直视场角 (默认 [90.0, 90.0])
        dist_scale: 深度缩放比例 (默认 1.0)
        rotate_points: 旋转参数列表 (默认 [['x', -30]] 旋转30°)
        filter_points: 过滤参数列表 (默认 [['y', -0.25, 0.25]] 过滤Y轴范围上下25cm)
        map_resolution: 占用栅格地图分辨率 (默认 0.2m)
        map_size: 占用栅格地图大小 (默认 100x100)
        coordinate_system: 坐标系 ('opengl' 或 'opencv', 默认 'opengl')
    returns:
        layer: 点云坐标 (N, 3) ndarray
        color: 点云颜色 (N, 3) ndarray (如果有RGB图像)
        occ_map: 占用栅格地图 (size, size) ndarray
    """
    cfg = Config(
        corrdinate_system=coordinate_system,
        sensor_cfg={"fov_deg": fov_deg, "dist_scale": dist_scale},
        transform_cfg={"rotate_points": rotate_points, "filter_points": filter_points},
        projection_cfg={"map_resolution": map_resolution, "map_size": map_size}
    )
    layer, color, occ_map = depth_layer_proj(depth, rgb=rgb, height=height, cfg=cfg)
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

        self.points, self.color = depth_to_filted_pointcloud(
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
        self.points, self.color = depth_to_filted_pointcloud(
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

def plot_laser_scan(ax, angles, dists):
    ax.clear()
    ax.set_title("Laser Scan")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    x = dists * np.cos(np.deg2rad(angles))
    y = dists * np.sin(np.deg2rad(angles))
    ax.scatter(x, y, marker='o')
    for i in range(len(angles)):
        ax.plot([0, x[i]], [0, y[i]], 'r--', alpha=0.3)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-0.2, 1.2])
    ax.grid(True)

def plot_data_frame(rgb, depth, height=None, cfg: Config=Config()):
    num_frames = rgb.shape[0]
    idx = [0]  # 用列表封装，方便在内部修改

    fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle(f"Frame {idx[0] + 1}/{num_frames}, idx {idx[0]}", fontsize=16)

    # 初始化三张图
    img_rgb = axes[0].imshow(rgb[idx[0]])
    axes[0].set_title("RGB")
    img_depth = axes[1].imshow(depth[idx[0]], cmap='viridis')
    axes[1].set_title("Depth")
    # img_attn = axes[1, 0].imshow(obj_attention[idx[0]], cmap='hot')
    # axes[1, 0].set_title("Object Attention")

    _, _, occ_map = depth_layer_proj(depth[idx[0]], 
                                     rgb[idx[0]], 
                                     height=height[idx[0]] if height is not None else None,
                                     cfg=cfg)
    # test API
    # _, _, occ_map = depth_layer_proj_api(depth[idx[0]], 
    #                                      rgb=rgb[idx[0]] if rgb is not None else None,
    #                                      height=height[idx[0]] if height is not None else None,
    #                                      fov_deg=cfg.sensor_cfg['fov_deg'],
    #                                      dist_scale=cfg.sensor_cfg['dist_scale'],
    #                                      rotate_points=cfg.transform_cfg['rotate_points'],
    #                                      filter_points=cfg.transform_cfg['filter_points'],
    #                                      map_resolution=cfg.projection_cfg['map_resolution'],
    #                                      map_size=cfg.projection_cfg['map_size'],
    #                                      coordinate_system=cfg.coordinate_system)
    img_occ = axes[2].imshow(occ_map, cmap="gray", origin="lower")
    axes[2].set_title("Occ Map")
    
    x, y, _, _ = depth_layer_scan(depth[idx[0]], 
                                  rgb=rgb[idx[0]], 
                                  height=height[idx[0]] if height is not None else None,
                                  cfg=cfg)
    
    axes[3].set_aspect('equal')
    global scatter 
    scatter = axes[3].scatter(x, y, marker='o')
    axes[3].set_title("Depth Scan")

    angles, dists = depth_layer_scan_api(depth[idx[0]], 
                                        rgb=rgb[idx[0]], 
                                        height=height[idx[0]] if height is not None else None,
                                        fov_deg=cfg.sensor_cfg['fov_deg'],
                                        dist_scale=cfg.sensor_cfg['dist_scale'],
                                        rotate_points=cfg.transform_cfg['rotate_points'],
                                        filter_points=cfg.transform_cfg['filter_points'],
                                        aggregation=cfg.laserscan_cfg['aggregation'],
                                        n_intervals=cfg.laserscan_cfg['n_intervals'],
                                        default_value=cfg.laserscan_cfg['default_value'])
    plot_laser_scan(axes[4], angles, dists)
    for ax in axes.flatten()[:-2]:
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
        # img_attn.set_data(obj_attention[idx[0]])

        _, _, occ_map = depth_layer_proj(depth[idx[0]], rgb[idx[0]], cfg=cfg)
        img_occ.set_data(occ_map)

        x, y, _, _ = depth_layer_scan(depth[idx[0]], rgb=rgb[idx[0]], height=height[idx[0]] if height is not None else None, cfg=cfg)
        global scatter
        scatter.remove()
        # 创建新的散点图
        axes[3].set_aspect('equal')
        scatter = axes[3].scatter(x, y, s=50, c='blue', alpha=0.6)

        angles, dists = depth_layer_scan_api(depth[idx[0]], 
                                            rgb=rgb[idx[0]], 
                                            height=height[idx[0]] if height is not None else None,
                                            fov_deg=cfg.sensor_cfg['fov_deg'],
                                            dist_scale=cfg.sensor_cfg['dist_scale'],
                                            rotate_points=cfg.transform_cfg['rotate_points'],
                                            filter_points=cfg.transform_cfg['filter_points'],
                                            aggregation=cfg.laserscan_cfg['aggregation'],
                                            n_intervals=cfg.laserscan_cfg['n_intervals'],
                                            default_value=cfg.laserscan_cfg['default_value'])
        plot_laser_scan(axes[4], angles, dists)

        

        fig.suptitle(f"Frame {idx[0] + 1}/{num_frames}, idx {idx[0]}", fontsize=16)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# ---------------------- 运行调试 ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云可视化与过滤工具")
    parser.add_argument("--cfg", type=str,help="参数配置文件路径（.yaml)")
    parser.add_argument("--data", type=str,
                        help="输入数据文件路径 (.npz)")
    parser.add_argument("--mode", type=str, choices=["viewer", "plot"], default="plot",
                        help="运行模式: 'viewer' 显示点云交互界面, 'plot' 绘制图像结果")
    args = parser.parse_args()
    if not args.cfg:
        args.cfg = "./depth_transform.yaml"
    if not args.data:
        args.data = "./data/data_batch_2.npz"
    cfg = Config.from_yaml(args.cfg)
    data = np.load(args.data, mmap_mode='r')
    frames = data['frame']        # (50, 300, 300, 3)
    depths = data['depth']        # (50, 300, 300)
    # attn   = data['obj_attention']# (50, 300, 300)
    if 'height' in data.keys():
        print("Height data found, using it for filtering.")
        heights = data['height'] 
        plt.plot(heights)
        plt.title("Drone Heights Visualization")
        plt.show()  
    else:
        heights = None  

    if args.mode == 'viewer':
        app = PointCloudFilterApp(depths, frames,  heights, cfg=cfg)
    elif args.mode == 'plot':
        plot_data_frame(frames, depths, heights, cfg=cfg)


