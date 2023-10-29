# lanoise_pp

[![License](https://img.shields.io/badge/License-BSD%203--Clause-gree.svg)](https://opensource.org/licenses/BSD-3-Clause)

Noising the point cloud.

The package is tested in Ubuntu 18.04/ROS Melodic, Python 3.7 and Ubuntu 20.04/ROS Noedic, Python 3.8.

Requirements:

numpy 1.22.3

scikit-learn 0.23.1

matplotlib 3.1.2

pytorch 1.11.0

example:

download the lanoise_pp package and decompress in ./src of your catkin workspace (e.g. catkin_ws).

in a new terminal:

```
cd ./catkin_ws

catkin_make
```

in the terminal:

`roscore`

in a new terminal:

`rviz`

play the reference rosbag (point clouds recorded by a velodyne LiDAR under clear weather conditions):

```
rosbag play -l --clock clear_day_new.bag

rosrun nodelet nodelet standalone velodyne_pointcloud/CloudNodelet _calibration:="/opt/ros/melodic/share/velodyne_pointcloud/params/32db.yaml"
```

in rviz, change the Fixed frame to "velodyne".

add the topic "/velodyne_points" in rviz to show the reference data.

set the visibility in lanoise_pp.py.

in a new terminal:

```
cd ./catkin_ws

source devel/setup.bash

roslaunch lanoise_pp lanoise_pp.launch
```

add the topic "/filtered_points" in rviz to show the noising point cloud.