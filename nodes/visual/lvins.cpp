#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <boost/version.hpp>
#include <boost/thread.hpp>
#if BOOST_VERSION >= 104100
    #include <boost/thread/future.hpp>
#endif // BOOST_VERSION >=  104100

#include <fstream>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <functional>
#include <atomic>
#include <condition_variable>

#include <ceres/ceres.h>

#include "integration_base.h"
#include "tic_toc.h"
#include "utility.h"
#include "parameters.h"
#include "imu_factor.h"
#include "pose_local_parameterization.h"

using namespace cv;
using namespace std;

/*
   ndt_mapping:使用全部的地图点云作为参考配准，速度更慢，但配准效果最好
*/
static ros::Time current_scan_time;
static ros::Time previous_scan_time;
static ros::Duration scan_duration;

static pcl::PointCloud<pcl::PointXYZI> mapCloud,filterMapCloud;
static pcl::PointCloud<pcl::PointXYZI> reference_map;
static pcl::PointCloud<pcl::PointXYZI> pointCurrent;

static pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;

static int ndt_iter = 30; // 最大迭代次数
static double ndt_res = 2.0; // 分辨率
static double step_size = 0.1; // 步进
static double trans_eps = 0.01; // Transformation epsilon

// 栅格点滤波的叶子大小
static double voxel_leaf_size = 2.0;

static ros::Publisher map_pub;
static ros::Publisher pubOdometry;

static int initial_scan_loaded = 0;

static double RANGE = 0.0;
static double POSESHIFT = 4.0,ANGLESHIFT=3.14,FRAMESHIFT=100;

static bool isMapUpdate = true;
static bool lazy = true;
static bool z_constraint=false;

static std::string _imu_topic = "/imu/data";//"/imu_sync";
static std::string _odom_topic = "/odom_pose";
static std::string _points_topic = "/velodyne_points";//"/imu_laser_point";

static std::string _parent_frame="/odom";
static std::string _child_frame="/base_link";
static int lazy_num=20;

//匹配得分
static double fitness_score;

static std::vector<pcl::PointCloud<pcl::PointXYZI> > previous_scans;

typedef boost::packaged_task<sensor_msgs::PointCloud2::Ptr> MapBuildingTask;
typedef boost::unique_future<sensor_msgs::PointCloud2::Ptr> MapBuildingFuture;    //异步建图返回值
boost::thread mapBuildingThread;
MapBuildingTask mapBuildingTask;
MapBuildingFuture mapBuildingFuture;
bool mapBuildingInProgress;

tf::TransformBroadcaster *tfBroadcaster2Pointer = NULL;
tf::StampedTransform laserOdometryTrans;

int frameCurrentCount,frameLastAddCount;


//每一帧处理时用到的数据
struct MeasurementData
{
    std::vector<sensor_msgs::ImuConstPtr> imu_list;
    sensor_msgs::PointCloud2ConstPtr laser;
};

//总共等待
int sum_of_wait = 0;

//各种互斥锁
//数据锁
std::mutex m_buf;
//状态锁
std::mutex m_state;
//图像锁
std::mutex i_buf;

//协方差变量，线程池的消息队列时使用
std::condition_variable con;
//imu的数据
std::queue<sensor_msgs::ImuConstPtr> imu_buf;
//激光点数据
std::queue<sensor_msgs::PointCloud2ConstPtr> laser_buf;

//当前的时间
double current_time = -1;
//上一时刻的时间
double latest_time;
//位姿
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Matrix3d tmp_R;
//imu的速度
Eigen::Vector3d tmp_V;
//加速度计偏差
Eigen::Vector3d tmp_Ba;
//陀螺仪偏差
Eigen::Vector3d tmp_Bg;
//上一时刻的线性加速度和角速度
Eigen::Vector3d tmp_acc_0;
Eigen::Vector3d tmp_gyr_0;

Eigen::Vector3d g={0,0,9.79362};

//滑动位置
Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
//滑动速度
Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];
//滑动旋转
Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
//滑动加速度计和陀螺仪偏差
Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];

//上一时刻位姿
Eigen::Matrix3d last_R,last_R0,add_R;
Eigen::Vector3d last_P,last_P0,add_P;
//头信息
std_msgs::Header Headers[(WINDOW_SIZE + 1)];

//预处理预积分滑动
IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
IntegrationBase *tmp_pre_integration;
//加速度和陀螺仪信息
Eigen::Vector3d acc_0, gyr_0;

//时间的滑动buf
vector<double> dt_buf[(WINDOW_SIZE + 1)];
//线性加速度和角速度
vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

vector<Eigen::Quaterniond> angle_buf[(WINDOW_SIZE + 1)];

//帧数
int frame_count=0;
//第一帧imu
bool first_imu=false;
//边缘化状态
enum MarginalizationFlag
{
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
};
bool marginalization_flag=MARGIN_OLD;

//位姿滑动窗口
double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
//速度偏差滑动窗口
double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
////特征点滑动窗口
//double para_Feature[NUM_OF_F][SIZE_FEATURE];
////相机与imu的相对位姿
//double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
////相对位姿
//double para_Retrive_Pose[SIZE_POSE];

//是否失败
bool failure_occur;

/*
 * 发布最近的里程计
 */
void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Matrix3d &R, const Eigen::Vector3d &V, const ros::Time time)
{
    Eigen::Quaterniond quadrotor_Q(R);

    nav_msgs::Odometry odometry;
    odometry.header.stamp = time;
    odometry.header.frame_id = _parent_frame;
    odometry.child_frame_id = _child_frame;
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = quadrotor_Q.x();
    odometry.pose.pose.orientation.y = quadrotor_Q.y();
    odometry.pose.pose.orientation.z = quadrotor_Q.z();
    odometry.pose.pose.orientation.w = quadrotor_Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pubOdometry.publish(odometry);

    tf::Quaternion q;
    q.setX(quadrotor_Q.x());
    q.setY(quadrotor_Q.y());
    q.setZ(quadrotor_Q.z());
    q.setW(quadrotor_Q.w());
    laserOdometryTrans.stamp_ = time;
    laserOdometryTrans.setRotation(q);
    laserOdometryTrans.setOrigin(tf::Vector3(P.x(), P.y(), P.z()));

    tfBroadcaster2Pointer->sendTransform(laserOdometryTrans);
}

/**
 * @brief predict
 * @param imu_msg
 * 处理imu数据
 */
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //imu的时间戳
    double t = imu_msg->header.stamp.toSec();
    //间隔时间
    double dt = t - latest_time;
    //上一时刻时间
    latest_time = t;

    //线性加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    //角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    //当前角速度
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    //上一时刻线性加速度,为x,y,z方向的重力的值，例如0.000001,0,9.8156
    Eigen::Vector3d un_acc_0 = tmp_Q * (tmp_acc_0 - tmp_Ba - tmp_Q.inverse() * g);

    //上一时刻角速度和当前角速度的平均减去当前的偏差
    Eigen::Vector3d un_gyr = 0.5 * (tmp_gyr_0 + angular_velocity) - tmp_Bg;
    //四元数表示的角度,q=角速度*t
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);
    tmp_R = tmp_Q;

    //当前线性加速度
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba - tmp_Q.inverse() * g);

    //当前的线性加速度=（上一时刻的线性加速度和当前线性速度的平均-acc的偏差-Q的逆*重力g）*Q
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    //位姿：p=p+t*v+0.5*t^2*acc
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    //速度：v=v+t*acc
    tmp_V = tmp_V + dt * un_acc;

    //加速度值,线性加速度值
    tmp_acc_0 = linear_acceleration;
    //陀螺仪的值,角速度值
    tmp_gyr_0 = angular_velocity;
}

/**
 * @brief processIMU
 * @param dt
 * @param linear_acceleration
 * @param angular_velocity
 * @param q
 *
 * 处理imu
 */
void processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity,const Eigen::Quaterniond q)
{
    //是否是第一帧imu
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    //如果该帧滑动窗口的预积分为空，则新建一个
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    if (frame_count != 0)
    {
        //调用imu的预积分
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        //加入时间，线性加速度，角速度,角度
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        angle_buf[frame_count].push_back(q);

        //提供优化的初始值
        int j = frame_count;
        Eigen::Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Eigen::Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        //当前帧imu得到的位姿和速度
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/*
 * imu预积分
 */
void send_imu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //获取imu的当前时间
    double t = imu_msg->header.stamp.toSec();
    //设置当前时间
    if (current_time < 0)
        current_time = t;
    //间隔时间
    double dt = t - current_time;
    current_time = t;

    double ba[]{0.0, 0.0, 0.0};
    double bg[]{0.0, 0.0, 0.0};

    //速度和角速度
    double dx = imu_msg->linear_acceleration.x - ba[0];
    double dy = imu_msg->linear_acceleration.y - ba[1];
    double dz = imu_msg->linear_acceleration.z - ba[2];

    double rx = imu_msg->angular_velocity.x - bg[0];
    double ry = imu_msg->angular_velocity.y - bg[1];
    double rz = imu_msg->angular_velocity.z - bg[2];

    //当前imu帧的角度
    Eigen::Quaterniond q(imu_msg->orientation.w,imu_msg->orientation.x, imu_msg->orientation.y, imu_msg->orientation.z);

    //处理imu，进行预积分
    processIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz),q);
}

/**
 * @brief imu_callback
 * @param imu_msg
 * imu数据获取回调函数，并按以imu的速度来发布里程计
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if(imu_msg->angular_velocity.x!=0 && imu_msg->angular_velocity.y!=0 && imu_msg->angular_velocity.z!=0
       && imu_msg->linear_acceleration.x!=0 && imu_msg->linear_acceleration.y!=0 && imu_msg->linear_acceleration.z!=0)
    {
        //数据锁上锁
        m_buf.lock();

        //imu数据加入队列
        imu_buf.push(imu_msg);

        //数据解锁
        m_buf.unlock();
        //条件变量解锁
        con.notify_one();

        //自动上锁解锁，保证区域自动解锁
        std::lock_guard<std::mutex> lg(m_state);
        //处理imu数据
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        //发布上一时刻的里程计
        //pubLatestOdometry(tmp_P, tmp_R, tmp_V, header.stamp);
    }
}

/**
 * @brief laser_callback
 * @param laser
 *
 * 原始激光点
 */
void laser_callback(const sensor_msgs::PointCloud2ConstPtr &laser)
{
    //数据上锁
    m_buf.lock();
    //加入特征点
    laser_buf.push(laser);
    //数据解锁
    m_buf.unlock();
    //条件变量解锁
    con.notify_one();
}

/**
 * @brief getMeasurements
 * @return
 * 获取测量数据，imu和特征点对
 */
std::vector<MeasurementData> getMeasurements()
{
    std::vector<MeasurementData> measurements;

    while (true)
    {
       //如果有数据为空，则返回空数据
       if (imu_buf.empty() || laser_buf.empty())
            return measurements;

       //激光的时间在imu队列的最开始和最后的之间
       if (!(imu_buf.back()->header.stamp > laser_buf.front()->header.stamp))
       {
           ROS_WARN("wait for imu, only should happen at the beginning");
           sum_of_wait++;
           return measurements;
       }

       if (!(imu_buf.front()->header.stamp < laser_buf.front()->header.stamp))
       {
           ROS_WARN("throw img, only should happen at the beginning");
           laser_buf.pop();
           continue;
       }

       sensor_msgs::PointCloud2ConstPtr laser = laser_buf.front();
       laser_buf.pop();

       std::vector<sensor_msgs::ImuConstPtr> IMUs;
       //获取激光时间戳之间的imu数据
       while (imu_buf.front()->header.stamp <= laser->header.stamp)
       {
           IMUs.emplace_back(imu_buf.front());
           //imu数据弹出
           imu_buf.pop();
       }

        MeasurementData m;
        m.imu_list=IMUs;
        m.laser=laser;

        measurements.emplace_back(m);
    }
    //返回当前帧数据
    return measurements;
}

/**
 * @brief update
 *
 * 更新状态
 */
void update()
{
    TicToc t_predict;
    //更新当前时间
    latest_time = current_time;
    //更新当前位姿
    tmp_P = Ps[WINDOW_SIZE];
    tmp_Q = Rs[WINDOW_SIZE];
    tmp_R = Rs[WINDOW_SIZE];
    //更新当前速度
    tmp_V = Vs[WINDOW_SIZE];
    //更新当前imu的偏差
    tmp_Ba = Bas[WINDOW_SIZE];
    tmp_Bg = Bgs[WINDOW_SIZE];
    //更新当前的角速度和线性加速度
    tmp_acc_0 = acc_0;
    tmp_gyr_0 = gyr_0;

    //更新imu数据池里的imu，用最新的imu更新位姿，速度，线性加速度，角速度
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

sensor_msgs::PointCloud2::Ptr updateMap(pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr,Eigen::Matrix4d T,bool update)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*scan_ptr,*transformed_scan_ptr, T);

    sensor_msgs::PointCloud2::Ptr filter_map_msg_ptr(new sensor_msgs::PointCloud2);

    if(lazy)
    {
        //先前加入的点云帧数大于REFERENCE_MAP_SIZE,3
        //清楚之前的点云
        if(previous_scans.size() >= (unsigned int)lazy_num)
        {
            previous_scans.erase(previous_scans.begin());
        }
        previous_scans.push_back(*transformed_scan_ptr);

        reference_map.clear();

        //只保留之前REFERENCE_MAP_SIZE(3)帧作为参考配准点云
        for(auto item = previous_scans.begin(); item != previous_scans.end(); item++)
        {
            reference_map += *item;
        }
    }
    else
    {
      //地图点云加上变换后的点云
      mapCloud += *transformed_scan_ptr;
    }

    add_P=Ps[frame_count];
    add_R=Rs[frame_count];

    frameLastAddCount = frameCurrentCount;

    isMapUpdate = true;


//    //地面剔除
//    pcl::PointCloud<pcl::PointXYZI>::Ptr filter_transformed_scan_ptr (new pcl::PointCloud<pcl::PointXYZI>());
//    filter(transformed_scan_ptr,filter_transformed_scan_ptr);

//    pcl::PointCloud<pcl::PointXYZI>::Ptr filter_map_ptr(new pcl::PointCloud<pcl::PointXYZI>(filterMapCloud));
//    filterMapCloud +=*filter_transformed_scan_ptr;
//    //发布地图
//    if(show_map)
//    {
//       pcl::toROSMsg(*filter_map_ptr, *filter_map_msg_ptr);
//       filter_map_msg_ptr->header.stamp=ros::Time::now();
//       filter_ndt_map_pub.publish(*filter_map_msg_ptr);
//    }
    return filter_map_msg_ptr;

}

void processNewMapIfAvailable()
{
    if (mapBuildingInProgress && mapBuildingFuture.has_value())
    {
        //地图更新
        if(isMapUpdate == true)
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(mapCloud));
            pcl::PointCloud<pcl::PointXYZI>::Ptr reference_map_ptr(new pcl::PointCloud<pcl::PointXYZI>(reference_map));

            //输入匹配的目标点云
            if(lazy)
            {
                ndt.setInputTarget(reference_map_ptr);
            }
            else
            {
                ndt.setInputTarget(map_ptr);
            }
            isMapUpdate = false;
        }
        mapBuildingInProgress = false;
    }
}

/**
 * @brief processLaser
 * @param input
 *
 */
void processLaser(sensor_msgs::PointCloud2ConstPtr& input)
{
    Headers[frame_count]=input->header;
    current_scan_time = input->header.stamp;

    double r;
    pcl::PointXYZI p;
    pcl::PointCloud<pcl::PointXYZI> scan;
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr (new pcl::PointCloud<pcl::PointXYZI>());

    Eigen::Matrix4d t_localizer(Eigen::Matrix4d::Identity());

    pcl::fromROSMsg(*input, pointCurrent);

    for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = pointCurrent.begin(); item != pointCurrent.end(); item++)
    {
        p.x = (double) item->x;
        p.y = (double) item->y;
        p.z = (double) item->z;
        p.intensity = (double) item->intensity;

        r = sqrt(pow(p.x, 2.0) + pow(p.y, 2.0));
        if(r > RANGE)
        {
            scan.push_back(p);
        }
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));
    pcl::PointCloud<pcl::PointXYZI>::Ptr filter_scan_ptr (new pcl::PointCloud<pcl::PointXYZI>());

    if(initial_scan_loaded == 0)
    {
      updateMap(scan_ptr,t_localizer,false);

      pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(mapCloud));
      pcl::PointCloud<pcl::PointXYZI>::Ptr reference_map_ptr(new pcl::PointCloud<pcl::PointXYZI>(reference_map));

      if(lazy)
      {
          ndt.setInputTarget(reference_map_ptr);
      }
      else
      {
          ndt.setInputTarget(map_ptr);
      }
      initial_scan_loaded = 1;

      last_P=Ps[frame_count];
      last_R=Rs[frame_count];
      last_P=Ps[0];
      last_R=Rs[0];
      previous_scan_time = current_scan_time;

      return;
    }

    // 栅格滤波
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.setInputCloud(scan_ptr);
    voxel_grid_filter.filter(*filtered_scan_ptr);

    processNewMapIfAvailable();

    TicToc t;

    //ndt设置滤波后的当前帧点云为输入点云
    ndt.setInputSource(filtered_scan_ptr);

//    Eigen::Isometry3d init_T=Eigen::Isometry3d::Identity();
//    init_T.rotate(Rs[frame_count]);
//    init_T.pretranslate(Ps[frame_count]);
    Eigen::Matrix4d init_T;
    init_T.block<3,3>(0,0)=Rs[frame_count];
    init_T(0,3)=Ps[frame_count][0];
    init_T(1,3)=Ps[frame_count][1];
    init_T(2,3)=Ps[frame_count][2];

    //ndt配准，输入预测位置，输出配准后的点云，得到当前得分
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    ndt.align(*output_cloud, init_T.cast<float>());
    fitness_score = ndt.getFitnessScore();
    t_localizer = ndt.getFinalTransformation().cast<double>();

    if(z_constraint)
    {
        t_localizer(2,0) = 0;
        t_localizer(2,1) = 0;
        t_localizer(2,2) = 1;
        t_localizer(0,2) = 0;
        t_localizer(1,2) = 0;
        t_localizer(2,3) = 0;//z=0
    }

    Rs[frame_count]<<static_cast<double>(t_localizer(0, 0)), static_cast<double>(t_localizer(0, 1)), static_cast<double>(t_localizer(0, 2)),
             static_cast<double>(t_localizer(1, 0)), static_cast<double>(t_localizer(1, 1)), static_cast<double>(t_localizer(1, 2)),
             static_cast<double>(t_localizer(2, 0)), static_cast<double>(t_localizer(2, 1)), static_cast<double>(t_localizer(2, 2));
    Ps[frame_count]<<t_localizer(0, 3),t_localizer(1, 3),t_localizer(2, 3);

    Eigen::Matrix4d current,last;
    current.block<3,3>(0,0)=Rs[frame_count];
    current(0,3)=Ps[frame_count].x();
    current(1,3)=Ps[frame_count].y();
    current(2,3)=Ps[frame_count].z();
    last.block<3,3>(0,0)=last_R;
    last(0,3)=last_P.x();
    last(1,3)=last_P.y();
    last(2,3)=last_P.z();

    Eigen::Matrix4d relta=last.inverse()*current;

    double secs = (current_scan_time - previous_scan_time).toSec();
    double diff_x = Ps[frame_count].x() - last_P.x();
    double diff_y = Ps[frame_count].y() - last_P.y();
    double diff_z = Ps[frame_count].z() - last_P.z();
    Vs[frame_count]<<diff_x/secs,diff_y/secs,diff_z/secs;
    //Vs[frame_count]<<relta(0,3)/secs,relta(1,3)/secs,relta(2,3)/secs;
//    pubLatestOdometry(Ps[frame_count],Rs[frame_count],Vs[frame_count],current_scan_time);


    Eigen::Vector3d current_r_ = Rs[frame_count].eulerAngles(0,1,2);//roll pitch yaw 顺序
    Eigen::Vector3d added_r_ = add_R.eulerAngles(0,1,2);//roll pitch yaw 顺序

    // 计算当前位置和上一时刻更新地图时的位置之间的x,y欧式距离
    double poseShift = sqrt(pow(Ps[frame_count].x()-add_P.x(), 2.0) + pow(Ps[frame_count].y()-add_P.y(), 2.0) + pow(Ps[frame_count].z()-add_P.z(), 2.0));
    // 角度偏差 roll,pitch
    double angleShift = sqrt(pow(current_r_(0)-added_r_(0),2)+pow(current_r_(1)-added_r_(1),2));
    // 帧差
    double frameShift=frameCurrentCount-frameLastAddCount;

    //若欧式距离大于一定阈值，则更新地图
    if((poseShift >= POSESHIFT || angleShift>=ANGLESHIFT || frameShift>=FRAMESHIFT)&&(!mapBuildingInProgress))
    {
        mapBuildingTask = MapBuildingTask(boost::bind(updateMap, scan_ptr,t_localizer, true));
        mapBuildingFuture = mapBuildingTask.get_future();
        mapBuildingThread = boost::thread(boost::move(boost::ref(mapBuildingTask)));
        mapBuildingInProgress = true;
    }

    last_P=Ps[frame_count];
    last_R=Rs[frame_count];
    previous_scan_time = current_scan_time;
    frameCurrentCount++;
}

void vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
//    for (int i = 0; i < NUM_OF_CAM; i++)
//    {
//        para_Ex_Pose[i][0] = tic[i].x();
//        para_Ex_Pose[i][1] = tic[i].y();
//        para_Ex_Pose[i][2] = tic[i].z();
//        Quaterniond q{ric[i]};
//        para_Ex_Pose[i][3] = q.x();
//        para_Ex_Pose[i][4] = q.y();
//        para_Ex_Pose[i][5] = q.z();
//        para_Ex_Pose[i][6] = q.w();
//    }
}

void double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

//    for (int i = 0; i < NUM_OF_CAM; i++)
//    {
//        tic[i] = Vector3d(para_Ex_Pose[i][0],
//                          para_Ex_Pose[i][1],
//                          para_Ex_Pose[i][2]);
//        ric[i] = Quaterniond(para_Ex_Pose[i][6],
//                             para_Ex_Pose[i][3],
//                             para_Ex_Pose[i][4],
//                             para_Ex_Pose[i][5]).toRotationMatrix();
//    }
}
/**
 * @brief optimization
 */
void optimization()
{
  //构建优化问题
  ceres::Problem problem;
  //构建loss_function,这里使用CauchyLoss
  ceres::LossFunction *loss_function;
  //loss_function = new ceres::HuberLoss(1.0);
  loss_function = new ceres::CauchyLoss(1.0);

  TicToc t_whole, t_prepare;
  vector2double();

  //将滑动窗口的中每一帧，添加第kth frame的state，(p,v,q,b_a,b_g)，添加位姿参数
  for (int i = 0; i < WINDOW_SIZE + 1; i++)
  {
      ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
      //添加位置参数项，7维
      problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
      //添加imu偏置参数项，9维
      problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
  }

  //添加imu的residual
  for (int i = 0; i < WINDOW_SIZE; i++)
  {
      int j = i + 1;
      if (pre_integrations[j]->sum_dt > 10.0)
          continue;
      //构建imu因子
      IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
      //添加imu因子误差项，待优化的para_Pose[i],para_SpeedBias[i],para_Pose[j],para_SpeedBias[j]
      problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
  }

  ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

  //设置优化器
  ceres::Solver::Options options;

  //设为舒尔补
  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.num_threads = 2;
  options.trust_region_strategy_type = ceres::DOGLEG;
  //设置最大迭代次数
  NUM_ITERATIONS=10;
  options.max_num_iterations = NUM_ITERATIONS;
  //options.use_explicit_schur_complement = true;
  //options.minimizer_progress_to_stdout = true;
  //options.use_nonmonotonic_steps = true;

  //设置了优化的最长时间，保证实时性
  //如果不是关键帧，则0.8*SOLVER_TIME，否则是SOLVER_TIME
  SOLVER_TIME=0.05;
  if (marginalization_flag == MARGIN_OLD)
      options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
  else
      options.max_solver_time_in_seconds = SOLVER_TIME;
  TicToc t_solver;
  ceres::Solver::Summary summary;
  //开始优化
  ceres::Solve(options, &problem, &summary);
  //cout << summary.BriefReport() << endl;
  ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
  ROS_DEBUG("solver costs: %f", t_solver.toc());

  //数组转向量和矩阵
  double2vector();
}

void slideWindow()
{
    //WINDOW_SIZE中的参数的之间调整，同时FeatureManager进行管理feature
    TicToc t_margin;

    //是关键帧
    if (marginalization_flag == MARGIN_OLD)
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
            angle_buf[WINDOW_SIZE].clear();
        }
    }
    //不是关键帧
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];
                Quaterniond tmp_angle = angle_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                angle_buf[frame_count -1].push_back(tmp_angle);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
            angle_buf[WINDOW_SIZE].clear();
        }
    }
}

/**
 * @brief process
 */
void measurementProcess()
{
    while(true)
    {
        //获取当前帧的数据，以图像特征点来获取一帧，即获取当前帧特征点与上一特征点之间的数据
        //包括多帧的imu,一帧的图像，一帧的原始激光点云，一帧激光特征点
        std::vector<MeasurementData> measurements;
        //锁住数据锁
        std::unique_lock<std::mutex> lk(m_buf);
        //等待解锁信号，获取当前时刻和上一时刻的全部imu和特征点对数据
        con.wait(lk, [&]
        {
            return (measurements = getMeasurements()).size() != 0;
        });
        //数据解锁
        lk.unlock();

        //遍历每一帧的数据对,一般只有一个
        for (auto &measurement : measurements)
        {
            //预积分imu数据,这里imu个数为imu频率/特征点的频率，例如imu频率为200Hz,特征点频率为10Hz，则应该有20个
            for (auto &imu_msg : measurement.imu_list)
                send_imu(imu_msg);

            //激光数据
            auto laser_msg = measurement.laser;

            tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

            processLaser(laser_msg);

            if(frame_count==WINDOW_SIZE)
            {
                //处理激光
                TicToc t_s;
                //优化
                //optimization();

                //滑动窗口，更新
                slideWindow();

                pubLatestOdometry(Ps[frame_count],Rs[frame_count],Vs[frame_count],current_scan_time);
            }
            else
            {
                frame_count++;
            }

        }

        //数据和状态上锁
        m_buf.lock();
        m_state.lock();
        update();
        //数据和状态解锁
        m_state.unlock();
        m_buf.unlock();
    }
}

/**
 * @brief clearState
 */
void clearState()
{
    last_R.setIdentity();
    last_P.setZero();
    //清除每一个滑动窗口的值
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();
        angle_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    first_imu = false,
    frame_count = 0;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;

    tmp_pre_integration = nullptr;
}

/**
 * @brief main
 * @param argc
 * @param argv
 * @return
 *
 * 主函数
 */
int main(int argc,char** argv)
{
    //ros节点初始化
    ros::init(argc,argv,"lvins");
    ros::NodeHandle nh("~");
    ros::NodeHandle private_nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    //设置ndt参数，正太分布变换配准
    ndt.setTransformationEpsilon(trans_eps);
    ndt.setStepSize(step_size);
    ndt.setResolution(ndt_res);
    //ndt最大迭代次数
    ndt.setMaximumIterations(ndt_iter);

    //初始化，清除全部状态
    clearState();

    //发布
    tf::TransformBroadcaster tfBroadcaster;
    tfBroadcaster2Pointer = &tfBroadcaster;
    laserOdometryTrans.frame_id_ = _parent_frame;
    laserOdometryTrans.child_frame_id_ = _child_frame;

    ACC_N=0.02;
    GYR_N=0.02;
    ACC_W=2.0e-5;
    GYR_W=2.0e-5;

    pubOdometry = nh.advertise<nav_msgs::Odometry> ("/odometry", 5);

    private_nh.param<std::string>("points_topic", _points_topic, "/velodyne_points");	ROS_INFO("points_topic: %s", _points_topic.c_str());
    private_nh.param<std::string>("odom_topic", _odom_topic, "/odom_pose");	ROS_INFO("odom_topic: %s", _imu_topic.c_str());
    private_nh.param<std::string>("imu_topic", _imu_topic, "/imu/data");	ROS_INFO("imu_topic: %s", _imu_topic.c_str());
    private_nh.param<std::string>("parent_frame", _parent_frame, "/odom");	ROS_INFO("parent_frame: %s", _parent_frame.c_str());
    private_nh.param<std::string>("child_frame", _child_frame, "/base_link");	ROS_INFO("child_frame: %s", _child_frame.c_str());


    //订阅激光点
    ros::Subscriber points_sub = nh.subscribe(_points_topic, 2000, laser_callback);
    //订阅imu
    ros::Subscriber imu_sub = nh.subscribe(_imu_topic, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    //订阅图像特征点
    ros::Subscriber image_feature_sub = nh.subscribe(_image_feature_topic,2000,image_feature_callback);
    //订阅图像
    ros::Subscriber image_sub = nh.subscribe(_image_topic,2000,image_callback);
    //订阅里程计
    ros::Subscriber wheel_odom_sub = nh.subscribe(_vheel_odom_topic,2000,wheel_callback);
    //订阅gps
    ros::Subscriber gps_sub = nh.subscribe(_gps_topic,2000,gps_callback);

    //测量数据处理的线程，主要的测量数据处理函数
    std::thread measurement_process{measurementProcess};

    //开始循环
    ros::spin();

    return 0;
}

