# depth_transform
A Script for transform from RGBD or Depth to Point Cloud, Occupancy Map

# Usage

Here are two functions which can be used for other projects:
* depth_to_pointcloud: The transform from RGBD or Depth to Point Cloud:
  * depth: depth information
  * rgb: img, can be none
  * cfg: transform config 
  * coordinate_system: 坐标系约定 ('opencv' 或 'opengl')\
            - opencv: X右, Y下, Z前\
            - opengl: X右, Y上, Z前
* depth_layer_proj: The transform from RGBD or Depth to Occupancy Map:
  * depth: depth information
  * rgb: img, can be none
  * cfg: transform config

Here are also two visual tools which will help you a lot: 
* PoinCloudFilterApp: have a view of point cloud\
  input same as above
* plot_data_frame: hav a view of rgb, depth and occ_map\
  input same as above

Open the shell or cmd and use below two commands to have a visual of input and ouput:
* ``` python .\depth_transform.py --cfg ./depth_transform.yaml --data /path/to/data --mode plot```\
  * You can use <-, -> to change frame\
![plot](./assets/plot.png)\
* ``` python .\depth_transform.py --cfg ./depth_transform.yaml --data /path/to/data --mode viewer```\
  * You can change frame through silder of Frame on the left side bar
![viewer](./assets/viewer.png)\
A statement of parameters in depth_tranform.yaml
* fov_deg: fov of RGBD, unit degree
* dist_scale:max distance of RGBD, unit meter
* rotate_points: SO3 tranform of output Point Cloud,  a example: [['z', -30], ['x', 30], ...]
* filter_points: Filter of output Point Cloud, a example: [['y', -0.25, 0.25], ...]
* map_resolution: Resolution of Occupancy Map
* map_size: Side length of Occupancy Map



