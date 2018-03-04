
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include "velodyne_pointcloud/point_types.h"
#include "velodyne_pointcloud/rawdata.h"
#include <stdio.h>
#include <stdlib.h>
#include <tf/transform_listener.h>
#include "tf/message_filter.h"
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

static std::string PARENT_FRAME;
static std::string CHILD_FRAME;
static std::string POINTS_TOPIC;
static std::string GROUND_POINTS_TOPIC;
static std::string OUTPUT_DIR;

static pcl::PointCloud<pcl::PointXYZI> ground_map;
static pcl::PointCloud<pcl::PointXYZI> map;
static tf::TransformListener *tf_listener;
static std::string filename;
static std::string pose_filename;

static int added_scan_num = 0;
static int map_id = 0;
static int count = 0;

static int ground_added_scan_num = 0;
static int ground_map_id = 0;
static int ground_count = 0;

ros::Publisher _pub_ground_cloud;
ros::Publisher _pub_cloud;
Eigen::Vector3d last_pose;
Eigen::Vector3d last_ground_pose;

static bool save_bin=true;

std::ofstream pose_ofs;

void ground_points_callback(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &input)
{
    ground_count++;

    pcl::PointCloud<pcl::PointXYZI> pcl_out;
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_input (new pcl::PointCloud<pcl::PointXYZI>);
    std_msgs::Header header;
    pcl_conversions::fromPCL(input->header, header);

    tf::StampedTransform transform;
    if(input->size() > 0)
    {
      try
      {
        tf_listener->waitForTransform(PARENT_FRAME, CHILD_FRAME, header.stamp, ros::Duration(1));
        tf_listener->lookupTransform(PARENT_FRAME, CHILD_FRAME, header.stamp, transform);
      }
      catch (tf::TransformException ex)
      {
        std::cout << "Transform not found" << std::endl;
        return;
      }

      Eigen::Vector3d current_pose;
      current_pose<<transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ();
      double poseShift = sqrt(pow(current_pose.x()-last_ground_pose.x(), 2.0) + pow(current_pose.y()-last_ground_pose.y(), 2.0) + pow(current_pose.z()-last_ground_pose.z(), 2.0));

      if(poseShift>0.1)
      {
          pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr (new pcl::PointCloud<pcl::PointXYZI>());
          pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
          voxel_grid_filter.setLeafSize(0.1, 0.1, 0.1);
          voxel_grid_filter.setInputCloud(input);
          voxel_grid_filter.filter(*filtered_scan_ptr);

          for (int i = 0; i < (int)filtered_scan_ptr->size(); i++)
          {
              tf::Point pt(filtered_scan_ptr->points[i].x, filtered_scan_ptr->points[i].y, filtered_scan_ptr->points[i].z);
              tf::Point pt_world = transform * pt;
              pcl::PointXYZI wp;
              double distance = pt.x() * pt.x() + pt.y() * pt.y() + pt.z() * pt.z();
              if (distance < 0.5)
                continue;
              wp.x = pt_world.x();
              wp.y = pt_world.y();
              wp.z = pt_world.z();
              wp.intensity = input->points[i].intensity;

              pcl_out.push_back(wp);
              ground_map.points.push_back(wp);
          }
          pcl_out.header = input->header;
          ground_map.header = input->header;

          pcl_out.header.frame_id = "map";
          ground_map.header.frame_id="odom";

          // Set log file name.
          std::ofstream ofs;
          filename = OUTPUT_DIR + PARENT_FRAME + "-" + CHILD_FRAME + "_" + POINTS_TOPIC + "_"+ std::to_string(map_id) + ".csv";
      //    ofs.open(filename.c_str(), std::ios::app);

      //    if (!ofs)
      //    {
      //      std::cerr << "Could not open " << filename << "." << std::endl;
      //      exit(1);
      //    }

      //    for (int i = 0; i < (int)pcl_out.points.size(); i++)
      //    {
      //      ofs << pcl_out.points[i].x << "," << pcl_out.points[i].y << "," << pcl_out.points[i].z << "," << pcl_out.points[i].intensity << std::endl;
      //    }
      //    std::cout << "Wrote " << pcl_out.size() << " points to " << filename << "." << std::endl;

          ground_added_scan_num++;
          if(ground_added_scan_num == 50)
          {
            ground_added_scan_num = 0;
            ground_map_id++;
          }

          sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
          pcl::toROSMsg(ground_map, *map_msg_ptr);
          _pub_ground_cloud.publish(*map_msg_ptr);

          last_ground_pose=current_pose;
      }
    }
}

void points_callback(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &input)
{
  count++;

  pcl::PointCloud<pcl::PointXYZI> pcl_out;
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_input (new pcl::PointCloud<pcl::PointXYZI>);
  std_msgs::Header header;
  pcl_conversions::fromPCL(input->header, header);

  tf::StampedTransform transform;
  if(input->size() > 0)
  {
    try
    {
      tf_listener->waitForTransform(PARENT_FRAME, CHILD_FRAME, header.stamp, ros::Duration(1));
      tf_listener->lookupTransform(PARENT_FRAME, CHILD_FRAME, header.stamp, transform);
    }
    catch (tf::TransformException ex)
    {
      std::cout << "Transform not found" << std::endl;
      return;
    }

    Eigen::Vector3d current_pose;
    current_pose<<transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ();
    double poseShift = sqrt(pow(current_pose.x()-last_pose.x(), 2.0) + pow(current_pose.y()-last_pose.y(), 2.0) + pow(current_pose.z()-last_pose.z(), 2.0));

    if(poseShift>4.0)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr (new pcl::PointCloud<pcl::PointXYZI>());
        pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
        voxel_grid_filter.setLeafSize(2.0, 2.0, 2.0);
        voxel_grid_filter.setInputCloud(input);
        voxel_grid_filter.filter(*filtered_scan_ptr);

        for (int i = 0; i < (int)filtered_scan_ptr->size(); i++)
        {
            tf::Point pt(filtered_scan_ptr->points[i].x, filtered_scan_ptr->points[i].y, filtered_scan_ptr->points[i].z);
            tf::Point pt_world = transform * pt;
            pcl::PointXYZI wp;
            double distance = pt.x() * pt.x() + pt.y() * pt.y() + pt.z() * pt.z();
            if (distance < 0.5)
              continue;
            wp.x = pt_world.x();
            wp.y = pt_world.y();
            wp.z = pt_world.z();
            wp.intensity = input->points[i].intensity;

            //pcl_out.push_back(wp);
            map.points.push_back(wp);
        }
        //pcl_out.header = input->header;
        map.header = input->header;

        //pcl_out.header.frame_id = "map";
        map.header.frame_id="odom";

        sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
        pcl::toROSMsg(map, *map_msg_ptr);
        _pub_cloud.publish(*map_msg_ptr);

        last_pose=current_pose;
    }
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "tf_mapping");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  last_pose<<0.0,0.0,0.0;


  private_nh.getParam("parent_frame", PARENT_FRAME);
  private_nh.getParam("child_frame", CHILD_FRAME);
  private_nh.getParam("points_topic", POINTS_TOPIC);
  private_nh.getParam("ground_points_topic", GROUND_POINTS_TOPIC);
  private_nh.getParam("output_dir", OUTPUT_DIR);
  private_nh.getParam("save_bin", save_bin);
  private_nh.getParam("pose_filename", pose_filename);

  std::cout << "parent_frame: " << PARENT_FRAME << std::endl;
  std::cout << "child_frame: " << CHILD_FRAME << std::endl;
  std::cout << "points_topic: " << POINTS_TOPIC << std::endl;
  std::cout << "ground_points_topic: " << GROUND_POINTS_TOPIC << std::endl;
  std::cout << "output_dir: " << OUTPUT_DIR << std::endl;
  std::cout << "save_bin: " << save_bin << std::endl;
  std::cout << "pose_filename: " << pose_filename << std::endl;


  tf_listener = new tf::TransformListener();

  ros::Subscriber points_sub = nh.subscribe(POINTS_TOPIC, 10, points_callback);
  ros::Subscriber ground_points_sub = nh.subscribe(GROUND_POINTS_TOPIC, 10, ground_points_callback);
  _pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/points_ground_map",1);
  _pub_cloud = nh.advertise<sensor_msgs::PointCloud2>("/points_map",1);

  ros::spin();

  return 0;
}
