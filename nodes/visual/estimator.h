#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/marginalization_factor.h"

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <thread>
#include <pthread.h>
#include <syscall.h>
#include <sys/types.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <boost/version.hpp>
#include <boost/thread.hpp>
#if BOOST_VERSION >= 104100
    #include <boost/thread/future.hpp>
#endif // BOOST_VERSION >=  104100

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

//相对数据
struct RetriveData
{
    /* data */
    //旧的id
    int old_index;
    //当前id
    int cur_index;
    //头
    double header;
    //旧的T，R
    Vector3d P_old;
    Quaterniond Q_old;
    //当前的T，R
    Vector3d P_cur;
    Quaterniond Q_cur;
    //特征点测量值
    vector<cv::Point2f> measurements;
    //特征点
    vector<int> features_ids; 
    //是否使用
    bool use;
    //相对位姿
    Vector3d relative_t;
    Quaterniond relative_q;
    //相对yaw
    double relative_yaw;
    //闭环位姿
    double loop_pose[7];
};

//估计优化器
class Estimator
{
  public:
    Estimator(bool _use_lasr=true,bool _use_imu=true,bool _use_visual=false,bool _use_wheel_odom=false,bool _use_gps=false);

    //设置参数
    void setParameter();

    //接口，处理imu,处理图像
    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Vector3d>>> &image, const std_msgs::Header &header);
    void processLaser(sensor_msgs::PointCloud2ConstPtr& input);
    void processWheelOdom();
    void processGps();
    void process();

    // internal
    //清除状态
    void clearState();
    //初始化结构体
    bool initialStructure();
    //初始化对准
    bool visualInitialAlign();
    //相对位姿
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    //滑动窗口
    void slideWindow();
    //计算里程计
    void solveOdometry();
    //滑动新窗口
    void slideWindowNew();
    //滑动旧窗口
    void slideWindowOld();
    //优化
    void optimization();
    //向量2double
    void vector2double();
    //double2向量
    void double2vector();
    //失败的检测
    bool failureDetection();

    //是初始化状态，非线性状态
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    //边缘化状态
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    //优化状态
    SolverFlag solver_flag;
    //边缘化状态
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    //？？？好像没用到
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    //每个相机与imu的相对位姿
    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    //滑动位置
    Vector3d Ps[(WINDOW_SIZE + 1)];
    //滑动速度
    Vector3d Vs[(WINDOW_SIZE + 1)];
    //滑动旋转
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    //滑动加速度计和陀螺仪偏差
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];

    //上一时刻位姿
    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    //头信息
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    //预处理预积分滑动
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    //加速度和陀螺仪信息
    Vector3d acc_0, gyr_0;

    //时间的滑动buf
    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    //线性加速度和角速度
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    //帧数
    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    //特征点管理
    FeatureManager f_manager;
    //运动估计
    MotionEstimator m_estimator;
    //初始化旋转
    InitialEXRotation initial_ex_rotation;

    //第一帧imu
    bool first_imu;
    //是否有效，是否是关键帧
    bool is_valid, is_key;
    //是否失败
    bool failure_occur;

    //特征点点云
    vector<Vector3d> point_cloud;
    //边缘化点云
    vector<Vector3d> margin_cloud;
    //关键位姿
    vector<Vector3d> key_poses;
    //初始化时间
    double initial_timestamp;


    //位姿滑动窗口
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    //速度偏差滑动窗口
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    //特征点滑动窗口
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    //相机与imu的相对位姿
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    //相对位姿
    double para_Retrive_Pose[SIZE_POSE];

    //相对位姿，上一个位姿
    RetriveData retrive_pose_data, front_pose;
    //闭环窗口index
    int loop_window_index;

    //边缘化信息
    MarginalizationInfo *last_marginalization_info;
    //上一次边缘化参数
    vector<double *> last_marginalization_parameter_blocks;

    //全部的图像帧
    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;


    //是否使用
    bool use_laser;
    bool use_imu;
    bool use_visual;
    bool use_wheel_odom;
    bool use_gps;

    //激光的变量
    //当期激光和上一时刻激光时间
    ros::Time current_scan_time;
    ros::Time previous_scan_time;

    //参考点云
    pcl::PointCloud<pcl::PointXYZI> reference_map;

    //ndt
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;

    //激光初始
    int initial_scan_loaded = 0;

    //是否更新激光地图
    bool isMapUpdate = true;

    //激光参考地图累积帧
    int lazy_num=20;

    //滑动的局部帧激光
    std::vector<pcl::PointCloud<pcl::PointXYZI> > previous_scans;

    //更新参考地图线程
    typedef boost::packaged_task<sensor_msgs::PointCloud2::Ptr> MapBuildingTask;
    typedef boost::unique_future<sensor_msgs::PointCloud2::Ptr> MapBuildingFuture;    //异步建图返回值
    boost::thread mapBuildingThread;
    MapBuildingTask mapBuildingTask;
    MapBuildingFuture mapBuildingFuture;
    bool mapBuildingInProgress;

    //当前激光帧号，和上一次累积参考地图的帧号
    int frameCurrentCount,frameLastAddCount;
    //上一次累积增加参考地图时的位姿
    Eigen::Matrix3d add_R;
    Eigen::Vector3d add_P;

};
