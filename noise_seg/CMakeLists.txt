cmake_minimum_required(VERSION 2.8.3)
project(noise_seg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs
)
