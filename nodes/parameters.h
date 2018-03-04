#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_LASER = 1;
const int NUM_OF_F = 1000;
const double LOOP_INFO_VALUE = 50.0;
//#define DEPTH_PRIOR
//#define GT
#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> CAM_RIC;
extern std::vector<Eigen::Vector3d> CAM_TIC;
extern std::vector<Eigen::Matrix3d> LASER_RIC;
extern std::vector<Eigen::Vector3d> LASER_TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string VINS_FOLDER_PATH;

extern int LOOP_CLOSURE;
extern int MIN_LOOP_NUM;
extern std::string PATTERN_FILE;
extern std::string VOC_FILE;
extern std::string CAM_NAMES;
extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string IMAGE_FEATURE_TOPIC;
extern std::string LASER_TOPIC;
extern std::string LASER_FEATURE_TOPIC;

void readParameters(ros::NodeHandle &n);

//各种优化变量的维度
//位姿7维
//速度9维
//特征1维
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

//状态信息
enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

//噪声
enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};


#endif // PARAMETERS_H
