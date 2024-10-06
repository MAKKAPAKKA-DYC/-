#include "windmill.hpp"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <future>   //多线程   
#include <iostream>    
#include <chrono>  
#include <cmath>    

using namespace std;
using namespace cv;

// 计算轮廓质心：给定一个轮廓（点集），计算所有点的平均值，返回轮廓的质心。
Point calculateCentroid(const vector<Point>& contour) {
    int x_sum = 0;
    int y_sum = 0;
    int n = contour.size();

    for (const auto& point : contour) {
        x_sum += point.x;
        y_sum += point.y;
    }

    return Point(x_sum / n, y_sum / n);
}

// Ceres的代价函数：该函数用于描述我们希望最小化的残差。
struct CostFunctor {
    CostFunctor(double time, double cosValue) : time_(time), cosValue_(cosValue) {}

    // 这是用于自动微分的代价函数实现
    template <typename T>
    bool operator()(const T* const amplitude0, const T* const amplitude, const T* const frequency, const T* const phaseShift, T* residual) const {
        // 计算模型的余弦值并与实际余弦值 (cosValue_) 进行比较
        residual[0] = cosValue_ - cos(amplitude0[0] * time_ + amplitude[0] / frequency[0] *
                        (cos(phaseShift[0] + M_PI / 2) - cos(frequency[0] * time_ + phaseShift[0] + M_PI / 2)));
        return true;
    }

private:
    const double time_;      // 当前时间，用于模型计算
    const double cosValue_;  // 实际余弦值，用于和模型值进行比较
};

// 检查参数组合是否满足预设的物理条件：该函数返回参数是否落在某些预定的区间内。
inline bool checkCombination(double amplitude0, double amplitude, double frequency, double phaseShift) {
    return (phaseShift > 0.22 && phaseShift < 0.25 &&         // 相位移必须在(0.22, 0.25)之间
            frequency > 1.78 && frequency < 1.98 &&           // 频率必须在(1.78, 1.98)之间
            amplitude0 > 1.23 && amplitude0 < 1.37 &&         // 初始振幅必须在(1.23, 1.37)之间
            amplitude > 0.74 && amplitude < 0.83);            // 振幅必须在(0.74, 0.83)之间
}

int main() {
    double total_time = 0;             // 记录总时间
    const int numIterations = 10;      // 总迭代次数

    // 迭代多次，进行优化实验
    for (int iteration = 0; iteration < numIterations; iteration++) {
        auto start_time = std::chrono::system_clock::now(); // 记录开始时间
        WINDMILL::WindMill windmill(std::chrono::duration_cast<std::chrono::milliseconds>(start_time.time_since_epoch()).count());
        Mat frame;  // 存储当前帧图像

        ceres::Problem ceresProblem;   // 创建Ceres优化问题
        double initialAmplitude = 0.305, amplitude = 1.785, frequency = 0.884, phaseShift = 1.24;  // 优化的初始参数

        int frameCount = 0;            // 记录帧的计数
        bool fittingSuccessful = false;    // 用于记录优化是否成功

        // 异步求解器：用于处理Ceres异步优化
        std::future<bool> ceresFuture;

        // 主循环，处理每一帧图像
        while (1) {
            frameCount++;  // 记录帧数
            auto current_time = std::chrono::system_clock::now();   // 记录当前时间
            double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(current_time.time_since_epoch()).count(); // 计算经过的时间
            frame = windmill.getMat(elapsedTime);  // 获取风车的当前图像帧

            // 图像处理：转换为灰度图和二值化处理
            Mat grayFrame, binaryFrame;
            cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
            threshold(grayFrame, binaryFrame, 50, 255, THRESH_BINARY);

            // 轮廓检测
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(binaryFrame, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            
            // 寻找中心轮廓和锤子轮廓
            int centerContourId = -1, hammerContourId = -1;
            for (size_t i = 0; i < contours.size(); i++) {
                if (centerContourId != -1 && hammerContourId != -1)
                    break;
                
                if (hierarchy[i][3] == -1) {  // 顶层轮廓
                    double area = contourArea(contours[i]);
                    if (area < 5000) {  // 根据轮廓面积判断
                        if (area < 200) {
                            centerContourId = i;  // 小轮廓，假设为中心
                            continue;
                        }
                        hammerContourId = hierarchy[i][2];  // 假设较大的轮廓是锤子
                        continue;
                    }
                }
            }

            // 如果找到了所需的轮廓
            if (centerContourId != -1 && hammerContourId != -1) {
                Point centerPoint = calculateCentroid(contours[centerContourId]);  // 计算中心质心
                Point hammerPoint = calculateCentroid(contours[hammerContourId]);  // 计算锤子质心

                // 计算从中心到锤子的向量及其单位向量
                Point centerToHammerVector = hammerPoint - centerPoint;
                Point2d unitVector = Point2d(centerToHammerVector) / norm(centerToHammerVector);

                double timeInSeconds = (elapsedTime - std::chrono::duration_cast<std::chrono::milliseconds>(start_time.time_since_epoch()).count()) / 1000.0;
                double cosAngle = unitVector.x;  // 计算与水平轴的余弦值

                // 每 10 帧添加一次残差块
                if (frameCount % 10 == 0) {
                    ceresProblem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1, 1, 1>(
                            new CostFunctor(timeInSeconds, cosAngle)), 
                        NULL, &initialAmplitude, &amplitude, &frequency, &phaseShift
                    );
                }

                // 在第700帧之后，每隔100帧中的前13帧执行异步优化任务
                if (frameCount > 700 && frameCount % 100 < 13) {
                    // 等待之前的异步任务完成
                    if (ceresFuture.valid())
                        ceresFuture.get();

                    auto async_start = std::chrono::high_resolution_clock::now();  // 开始异步优化

                    std::cout << "Starting Ceres optimization in iteration: " << iteration << ", frame: " << frameCount << std::endl;

                    // 异步调用Ceres优化
                    ceresFuture = std::async(std::launch::async, [&ceresProblem, &initialAmplitude, &amplitude, &frequency, &phaseShift]() {
                        ceres::Solver::Options options;
                        options.max_num_iterations = 50;  // 设置最大迭代次数
                        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 使用稠密求解器

                        // 设置参数的上下界
                        ceresProblem.SetParameterLowerBound(&initialAmplitude, 0, 0.5);
                        ceresProblem.SetParameterUpperBound(&initialAmplitude, 0, 5.0);
                        ceresProblem.SetParameterLowerBound(&amplitude, 0, 0.5);
                        ceresProblem.SetParameterUpperBound(&amplitude, 0, 5.0);
                        ceresProblem.SetParameterLowerBound(&frequency, 0, 0.5);
                        ceresProblem.SetParameterUpperBound(&frequency, 0, 5.0);
                        ceresProblem.SetParameterLowerBound(&phaseShift, 0, 0.1);
                        ceresProblem.SetParameterUpperBound(&phaseShift, 0, 3.14);


                        ceres::Solver::Summary summary;
                        Solve(options, &ceresProblem, &summary);  // 运行Ceres求解器

                        // 检查参数组合是否符合条件
                        return checkCombination(initialAmplitude, amplitude, frequency, phaseShift);
                    });

                    std::chrono::seconds timeoutDuration(2);  // 设置2秒超时

                    // 开始等待异步优化任务
                    if (ceresFuture.wait_for(timeoutDuration) != std::future_status::ready) {
                        // 如果任务在超时时间内没有完成，跳过当前优化周期
                        std::cout << "Optimization timed out, skipping this optimization cycle." << std::endl;
                        continue;  // 跳过当前帧，避免阻塞
                    } else {
                        // 如果任务在超时时间内完成，获取优化结果
                        fittingSuccessful = ceresFuture.get();

                        // 输出优化结果
                        auto async_end = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> async_duration = async_end - async_start;
                        std::cout << "Ceres optimization finished. Time taken: " << async_duration.count() << " seconds" << std::endl;
                        std::cout << "Optimization success: " << (fittingSuccessful ? "Yes" : "No") << std::endl;

                        // 如果优化成功，跳出循环
                        if (fittingSuccessful) {
                            auto end_time = std::chrono::system_clock::now();
                            std::chrono::duration<double> duration = end_time - start_time;
                            total_time += duration.count();
                            break;
                        }
                    }
                }
            }
        }
    }

    // 输出每次迭代的平均时间
    std::cout << "Average time per iteration: " << total_time / numIterations << " seconds" << std::endl;

    return 0;
}

