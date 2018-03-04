
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
static std::string OUTPUT_DIR;

static tf::TransformListener *tf_listener;
static std::string filename;
static std::string pose_filename;

static int added_scan_num = 0;
static int map_id = 0;
static int count = 0;

static bool save_bin=true;

std::ofstream pose_ofs;

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

    if(save_bin)
    {
        for (int i = 0; i < (int)input->size(); i++)
        {
            tf::Point pt(input->points[i].x, input->points[i].y, input->points[i].z);
            tf::Point pt_world = transform * pt;
            pcl::PointXYZI wp;
            double distance = pt.x() * pt.x() + pt.y() * pt.y() + pt.z() * pt.z();
            wp.x = pt.x();
            wp.y = pt.y();
            wp.z = pt.z();
            wp.intensity = input->points[i].intensity;

            pcl_out.push_back(wp);
        }
        pcl_out.header = input->header;
        pcl_out.header.frame_id = "map";

        // Set log file name.
        std::ofstream ofs;
        std::stringstream ss;
        ss<<std::setw(6)<<std::setfill('0')<<map_id;
        filename = OUTPUT_DIR + ss.str() + ".bin";
        ofs.open(filename.c_str());

        if (!ofs)
        {
          std::cerr << "Could not open " << filename << "." << std::endl;
          exit(1);
        }

        for (int i = 0; i < (int)pcl_out.points.size(); i++)
        {
          ofs << pcl_out.points[i].x << "," << pcl_out.points[i].y << "," << pcl_out.points[i].z << "," << pcl_out.points[i].intensity << std::endl;
        }
        std::cout << "Wrote " << pcl_out.size() << " points to " << filename << "." << std::endl;

        Eigen::Quaterniond q(transform.getRotation().getW(),transform.getRotation().getX(),transform.getRotation().getY(),transform.getRotation().getZ());
        Eigen::Matrix3d r;
        r=q;
        pose_ofs << r(0,0)<<" "<<r(0,1)<<" "<<r(0,2)<<" "<<transform.getOrigin().x()
                <<" "<<r(1,0)<<" "<<r(1,1)<<" "<<r(1,2)<<" "<<transform.getOrigin().y()
                <<" "<<r(2,0)<<" "<<r(2,1)<<" "<<r(2,2)<<" "<<transform.getOrigin().z()<<"\n";

        ofs.close();
        map_id++;
    }
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "tf_mapping");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");


  private_nh.getParam("parent_frame", PARENT_FRAME);
  private_nh.getParam("child_frame", CHILD_FRAME);
  private_nh.getParam("points_topic", POINTS_TOPIC);
  private_nh.getParam("output_dir", OUTPUT_DIR);
  private_nh.getParam("save_bin", save_bin);
  private_nh.getParam("pose_filename", pose_filename);

  std::cout << "parent_frame: " << PARENT_FRAME << std::endl;
  std::cout << "child_frame: " << CHILD_FRAME << std::endl;
  std::cout << "points_topic: " << POINTS_TOPIC << std::endl;
  std::cout << "output_dir: " << OUTPUT_DIR << std::endl;
  std::cout << "save_bin: " << save_bin << std::endl;
  std::cout << "pose_filename: " << pose_filename << std::endl;

  pose_ofs.open(pose_filename.c_str());
  if (!pose_ofs)
  {
    std::cerr << "Could not open " << pose_filename << "." << std::endl;
    exit(1);
  }

  tf_listener = new tf::TransformListener();

  ros::Subscriber points_sub = nh.subscribe(POINTS_TOPIC, 10, points_callback);

  ros::spin();

  return 0;
}
