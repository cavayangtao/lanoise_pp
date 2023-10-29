# noise_seg

[![License](https://img.shields.io/badge/License-BSD%203--Clause-gree.svg)](https://opensource.org/licenses/BSD-3-Clause)

Denoising the point cloud.

The package is tested in Ubuntu 18.04/ROS Melodic, Python 3.7 and Ubuntu 20.04/ROS Noedic, Python 3.8.

Requirements are same as [RandLA-Net](../RandLA-Net).

example:

download the noise_seg package and decompress in ./src of your catkin workspace (e.g. catkin_ws).

in a new terminal:

```
cd ./catkin_ws

catkin_make
```

in the terminal:

`roscore`

in a new terminal:

`rviz`

play the reference rosbag (point clouds recorded by a velodyne LiDAR under adverse weather conditions):

```
rosbag play -l --clock snow_day_new.bag

rosrun nodelet nodelet standalone velodyne_pointcloud/CloudNodelet _calibration:="/opt/ros/melodic/share/velodyne_pointcloud/params/32db.yaml"
```

in rviz, change the Fixed frame to "velodyne".

add the topic "/velodyne_points" in rviz to show the noising data.

in a new terminal:

```
cd ./catkin_ws

source devel/setup.bash

roslaunch denoise_seg denoise_seg.launch
```

add the topic "/filtered_points" in rviz to show the denoising point clouds.