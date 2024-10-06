#ifndef WINDMILL_H_
#define WINDMILL_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <random>

namespace WINDMILL
{
    class WindMill
    {
    private:
        int cnt;
        bool direct;
        double A;
        double w;
        double A0;
        double fai;
        double now_angle;
      
        cv::Point2i R_center;
        void drawR(cv::Mat &img, const cv::Point2i &center);
        void drawHitFan(cv::Mat &img, const cv::Point2i &center, double angle);
        void drawOtherFan(cv::Mat &img, const cv::Point2i &center, double angle);
        
        // 计算给定角度和半径的点的位置
        cv::Point calPoint(const cv::Point2f &center, double angle_deg, double r)
        {
            return center + cv::Point2f((float)cos(angle_deg / 180 * 3.1415926), (float)-sin(angle_deg / 180 * 3.1415926)) * r;
        }

        // 计算当前时间的总角度
        double SumAngle(double angle_now, double t0, double dt)
        {
            double dangle = A0 * dt + (A / w) * (cos(w * t0 + 1.81) - cos(w * (t0 + dt) + 1.81));
            angle_now += dangle / 3.1415926 * 180;
            if (angle_now < 0)
            {
                angle_now = 360 + angle_now;
            }
            if (angle_now > 360)
            {
                angle_now -= 360;
            }
            return angle_now;
        }

    public:
      double start_time;
        WindMill(double time = 0);
        cv::Mat getMat(double time);

        // Getter 方法来访问 R_center
        cv::Point2i getR_center() const { return R_center; }
        
        // 计算当前时间的总角度
        double getCurrentAngle(double t0, double currentTime) const {
            double dt = currentTime - t0;
            double dangle = A0 * dt + (A / w) * (cos(w * t0 + 1.81) - cos(w * (t0 + dt) + 1.81));
            double angle_now = dangle / 3.1415926 * 180;
            if (angle_now < 0)
            {
                angle_now = 360 + angle_now;
            }
            if (angle_now > 360)
            {
                angle_now -= 360;
            }
            return angle_now;
        }
        
        // 提供公共方法来计算给定角度和半径的位置
        cv::Point calculatePoint(const cv::Point2f &center, double angle_deg, double r) const {
            return center + cv::Point2f((float)cos(angle_deg / 180 * 3.1415926), (float)-sin(angle_deg / 180 * 3.1415926)) * r;
        }
    };
} // namespace WINDMILL

#endif