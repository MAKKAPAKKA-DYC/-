#include "windmill.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;

int main()
{
    // 初始化风车
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    WINDMILL::WindMill wm(t.count());
    cv::Mat src;

    // 加载 r 模板图像
    cv::Mat rTemplate = imread("/home/ubuntu/project01/image/R.png", IMREAD_GRAYSCALE);
    if (rTemplate.empty()) {
        cerr << "Failed to load r template image!" << endl;
        return -1;
    }

    // r 模板参数
    int rWidth = rTemplate.cols;
    int rHeight = rTemplate.rows;

    // 锤子的参数
    double hammerRadius1 = 40;   // 锤子到风车中心的第一段距离
    double hammerRadius2 = 150;  // 锤子到风车中心的第二段距离
    double hammerRadius3 = 190;  // 锤子到风车中心的第三段距离
    int hammerWidth = 60;        

    // 用于存储每帧的灰度图像
    cv::Mat graySrc;

    // 圆的参数
    Point hammerCenter, rCenter;

    while (true)
    {
        // 获取当前时间
        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double currentTime = t.count() / 1000.0; // 确保转换为秒

        // 获取当前帧
        src = wm.getMat(currentTime);

        //====================计算锤子位置并绘制蓝框====================//

        // 获取风车中心位置
        cv::Point2i R_center = wm.getR_center();
        
        // 计算风车当前角度
        double now_angle = wm.getCurrentAngle(wm.start_time, currentTime);

        // 计算锤子的位置，drawHitFan 中锤子的位置计算
        cv::Point mid1 = wm.calculatePoint(R_center, now_angle, hammerRadius1);
        cv::Point mid2 = wm.calculatePoint(R_center, now_angle, hammerRadius2);
        cv::Point mid3 = wm.calculatePoint(R_center, now_angle, hammerRadius3);

        // 计算锤子的边界框
        int hammerHalfWidth = hammerWidth / 2;

        // 获取锤子位置的最小和最大坐标
        int minX = std::min({mid1.x, mid2.x - hammerHalfWidth, mid2.x + hammerHalfWidth, mid3.x - hammerHalfWidth, mid3.x + hammerHalfWidth});
        int maxX = std::max({mid1.x, mid2.x - hammerHalfWidth, mid2.x + hammerHalfWidth, mid3.x - hammerHalfWidth, mid3.x + hammerHalfWidth});
        int minY = std::min({mid1.y, mid2.y - hammerHalfWidth, mid2.y + hammerHalfWidth, mid3.y - hammerHalfWidth, mid3.y + hammerHalfWidth});
        int maxY = std::max({mid1.y, mid2.y - hammerHalfWidth, mid2.y + hammerHalfWidth, mid3.y - hammerHalfWidth, mid3.y + hammerHalfWidth});

        // 绘制锤子的边界框
        cv::Rect hammerRect(minX, minY, maxX - minX, maxY - minY);
        rectangle(src, hammerRect, Scalar(0, 255, 0), 2); 

        // 计算锤子的中心点
        hammerCenter = cv::Point((minX + maxX) / 2, (minY + maxY) / 2);

        //====================检测 r 并绘制绿框====================//

        // 将当前帧转换为灰度图像
        cvtColor(src, graySrc, COLOR_BGR2GRAY);

        // 模板匹配来查找 r 的位置
        cv::Mat result;
        matchTemplate(graySrc, rTemplate, result, TM_CCOEFF_NORMED);

        // 找到匹配的最大位置
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        // 设置匹配阈值
        double matchThreshold = 0.8; // 根据实际效果调整阈值
        if (maxVal >= matchThreshold) {
            // 在原图中获取 r 的位置
            cv::Rect rRect(maxLoc.x, maxLoc.y, rWidth, rHeight);
            rectangle(src, rRect, Scalar(255, 0, 0), 2); 
            
            // 计算 r 中心点
            rCenter = cv::Point(maxLoc.x + rWidth / 2, maxLoc.y + rHeight / 2);
        }

        //====================绘制圆形====================//

        // 计算两中心点之间的距离
        double distance = norm(hammerCenter - rCenter);

        // 圆心为两个中心点的中点
        Point circleCenter = (hammerCenter + rCenter) * 0.5;

        // 绘制以两边框的中心点为圆上两点的圆
        circle(src, circleCenter, distance / 2, Scalar(255, 255, 0), 2); 

        // 显示风车
        imshow("Hammer and R Detection", src);

        //==========================================================//

            waitKey(1);
        }
    }



  
