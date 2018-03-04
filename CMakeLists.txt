cmake_minimum_required(VERSION 2.8.3)
project(lvins)

add_compile_options(-std=c++11)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
Set(EIGEN_INCLUDE_DIR "/usr/include/eigen3")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    rosbag
    geometry_msgs
    pcl_ros
    sensor_msgs
    pcl_conversions
    velodyne_pointcloud
    message_generation
    std_msgs
    nav_msgs
    tf
    tf2
    tf_conversions
    message_filters
    map_msgs
    visualization_msgs
    interactive_markers
    message_generation
    #cv_bridge
    image_transport
    )

#set(CMAKE_PREFIX_PATH "/home/ren/opencv3/opencv3_build/share/OpenCV")
find_package(OpenCV 2 REQUIRED)# HINTS "/home/ren/opencv3/opencv3_build/share/OpenCV")
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#find_package(Eigen REQUIRED)

#add_message_files(FILES
#)

#add_service_files( FILES
#)

generate_messages(
  DEPENDENCIES
  sensor_msgs nav_msgs std_msgs geometry_msgs actionlib_msgs
)


###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs velodyne_pointcloud roscpp rospy rosbag geometry_msgs nav_msgs
  tf tf2 tf_conversions message_filters map_msgs visualization_msgs interactive_markers
  message_runtime
)

###########
## Build ##
###########

include_directories(include 
	${catkin_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
        ${PROJECT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIR}
)

#add_executable(scan2scan_imu_odom
#    nodes/scan2scan_imu_odom.cpp
#    nodes/parameters.cpp
#    nodes/utility.cpp
#    nodes/pose_local_parameterization.cpp)

#target_link_libraries(scan2scan_imu_odom
#    ${catkin_LIBRARIES}
#    ${PCL_LIBRARIES}
#    ${Boost_LIBRARIES}
#    ${OpenCV_LIBRARIES}
#    ${CERES_LIBRARIES}
#)

add_executable(scan2mapping_imu_odom
    nodes/scan2mapping_imu_odom.cpp)

target_link_libraries(scan2mapping_imu_odom
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES}
)

add_executable(save_bin
    nodes/save_bin.cpp)

target_link_libraries(save_bin
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)
