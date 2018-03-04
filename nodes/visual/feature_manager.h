#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

//每一个特征点在每一帧中类
class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Vector3d &_point)
    {
        z = _point(2);
        point = _point / z;
    }
    //点位姿
    Vector3d point;
    //深度值
    double z;
    //是否使用
    bool is_used;
    //视差
    double parallax;
    MatrixXd A;
    VectorXd b;
    //梯度
    double dep_gradient;
};

//每一个id特征
class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    //该特征点在所有帧中的
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

//特征点管理
class FeatureManager
{
  public:
    //初始化
    FeatureManager(Matrix3d _Rs[]);

    //设置ric
    void setRic(Matrix3d _ric[]);

    //清除状态
    void clearState();

    //获取特征点个数
    int getFeatureCount();

    //添加特征检查视差
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Vector3d>>> &image);
    //调试显示
    void debugShow();
    //获取两个关联
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    //设置深度
    void setDepth(const VectorXd &x);
    //移除失败的
    void removeFailures();
    //清除深度
    void clearDepth(const VectorXd &x);
    //获取深度向量
    VectorXd getDepthVector();
    //三角化
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif
