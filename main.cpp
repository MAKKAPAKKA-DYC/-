#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// 计算图像的信噪比 (SNR) 作为选择滤波核大小的依据
double computeSNR(const Mat& original, const Mat& filtered) {
    Mat noise = original - filtered;
    double signalPower = norm(original, NORM_L2);
    double noisePower = norm(noise, NORM_L2);
    return 20 * log10(signalPower / noisePower);
}
// 计算高斯滤波的标准差 (sigma)
double calculateSigma(int kernelSize) {
    return 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
}
int main() {
    // 读取图像
    cv::Mat image = cv::imread("/home/ubuntu/project/resources/Screenshot from 2024-09-13 20-45-24.png");
    
    // 检查图像是否加载成功
    if (image.empty()) {
        std::cerr << "无法加载图像!" << std::endl;
        return -1;
    }

    // 将图像转换为灰度图
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    cv::imwrite("../resources/gray_image.png", gray_image); // 保存灰度图

    // 将图像转换为 HSV 图
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    cv::imwrite("../resources/hsv_image.png", hsv_image); // 保存 HSV 图

    // 应用均值滤波
   // 不同的核大小进行均值滤波
    vector<int> kernelSizes = {3, 5, 7, 9, 11,13};
    Mat filteredImage, bestFilteredImage;
    double bestSNR = -1;  // 初始化信噪比
    int bestKernelSize = 0;

    for (int kernelSize : kernelSizes) {
        // 对图像应用均值滤波
        blur(image, filteredImage, Size(kernelSize, kernelSize));

        // 计算信噪比 SNR，作为滤波效果的评估标准
        double snr = computeSNR(image, filteredImage);
        cout << "Kernel size: " << kernelSize << "x" << kernelSize << ", SNR: " << snr << endl;

        // 选择最好的核大小并保存该图像
        if (snr > bestSNR) {
            bestSNR = snr;
            bestKernelSize = kernelSize;
            bestFilteredImage = filteredImage.clone(); // 保存最佳滤波效果图像
        }
    }

    // 保存最佳滤波效果的图像
    imwrite("../resources/best_filtered_image.jpg", bestFilteredImage);
        // 应用高斯滤波
  cv::Mat dst, bbestFilteredImage;
    
    // 核大小范围（必须为奇数）
    vector<int> kkernelSizes = {3, 5, 7, 9, 11, 13};
    double bbestSNR = -1;  // 初始化 SNR
    int bbestKernelSize = 0;  // 最佳核大小

    // 循环遍历不同的核大小
    for (int kernelSize : kernelSizes) {
        // 计算对应的 sigma
        double sigmaX = calculateSigma(kernelSize);

        // 对图像应用高斯滤波
        GaussianBlur(image, dst, Size(kernelSize, kernelSize), sigmaX);

        // 计算当前滤波结果的信噪比 (SNR)
        double snr = computeSNR(image, dst);
        cout << "Kernel size: " << kernelSize << "x" << kernelSize << ", SNR: " << snr << endl;

        // 保存 SNR 最高的图像
        if (snr > bestSNR) {
            bestSNR = snr;
            bestKernelSize = kernelSize;
            bestFilteredImage = dst.clone();  // 保存当前最佳的滤波结果
        }
    }
    imwrite("../resources/best_gaussian_blurred_image.jpg", bbestFilteredImage);


    

    // =======================
    // 方法一：使用 HSV 提取红色区域
    // =======================

    // 将最佳高斯滤波后的图像转换为 HSV 色彩空间
    Mat hsv;
    cvtColor(bbestFilteredImage, hsv, COLOR_BGR2HSV);

    // 定义红色区域的 HSV 范围
    Scalar lower_red1 = Scalar(0, 50, 50);    // 低端红色 (0° ~ 10°)
    Scalar upper_red1 = Scalar(10, 255, 255);
    Mat mask1;
    inRange(hsv, lower_red1, upper_red1, mask1);

    Scalar lower_red2 = Scalar(170, 50, 50);  // 高端红色 (170° ~ 180°)
    Scalar upper_red2 = Scalar(180, 255, 255);
    Mat mask2;
    inRange(hsv, lower_red2, upper_red2, mask2);

    // 合并两个掩码，得到完整的红色区域
    Mat mask;
    bitwise_or(mask1, mask2, mask);

    // 对掩码进行形态学操作（可选），修复小的空洞和噪声
    Mat kernell= getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernell);  // 闭操作：填充小的空洞
    morphologyEx(mask, mask, MORPH_OPEN, kernell);   // 开操作：去除小噪声

    // 使用掩码从原图中提取红色区域
    Mat redRegionh;
    bitwise_and(bestFilteredImage, bestFilteredImage, redRegionh, mask);
imwrite("../resources/red_region_after_gaussian_blur.jpg", redRegionh);

  
     cv::imwrite("../resources/red_area_hsv.png", redRegionh);  // 保存结果

    // =======================
    // 方法二：使用 BGR 提取红色区域
    // =======================

    // 提取红色区域
    Mat redMask;
    inRange(bbestFilteredImage, Scalar(0, 0, 100), Scalar(50, 50, 255), redMask);  // BGR 范围：提取红色

    // 对掩码进行形态学操作，修复小的空洞和噪声
    Mat kernelll = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(redMask, redMask, MORPH_CLOSE, kernelll);  // 闭操作：填充小的空洞
    morphologyEx(redMask, redMask, MORPH_OPEN, kernelll);   // 开操作：去除小噪声

    // 使用掩码从原图中提取红色区域
    Mat redRegionb;
    bitwise_and(bestFilteredImage, bestFilteredImage, redRegionb, redMask);

    cv::imwrite("../resources/red_area_bgr.png",redRegionb);  // 保存结果
    
    // 找到红色区域的轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(redRegionb, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
   
   // 创建掩膜，设定红色的 BGR 范围
    cv::Mat red_mask;
    cv::inRange(image, cv::Scalar(0, 0, 120), cv::Scalar(120, 120, 255), red_mask);  // 通过调整 BGR 范围检测红色

    // 创建一个副本图像用于绘制轮廓和 bounding box
    cv::Mat output_img = redRegionb.clone();

   double total_area = 0.0;
   //显示轮廓并绘制外轮廓、bounding box 和计算面积
    for (size_t i = 0; i < contours.size(); i++) {
        // 绘制红色轮廓
        cv::drawContours(output_img, contours, (int)i, cv::Scalar(0, 255, 0), 2);

        // 计算当前轮廓面积
        double area = cv::contourArea(contours[i]);
        total_area += area;  // 累加面积
        std::cout << "轮廓 " << i << " 面积: " << area << std::endl;

        // 找到轮廓的边界矩形
        cv::Rect bounding_box = cv::boundingRect(contours[i]);

        // 绘制矩形到图像上
        cv::rectangle(output_img, bounding_box, cv::Scalar(255, 0, 0), 2);

        // 在图像上绘制面积信息
        std::string area_text = "Area: " + std::to_string((int)area);
        cv::putText(output_img, area_text, bounding_box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }

    // 输出总面积
    std::cout << "所有轮廓的总面积: " << total_area << std::endl;

    //保存图片
    cv::imwrite("../resources/output_with_contours.jpg", output_img);

    // 转换到 HSV 颜色空间
    cv::Mat hsv_img;
    cv::cvtColor(image, hsv_img, cv::COLOR_BGR2HSV);

    // 提取高亮区域 (使用 HSV 颜色空间的 Value 通道)
    cv::Mat high_light_mask;
    // 设置 Value 通道的阈值范围，假设亮度在 [200, 255] 为高亮区域
    cv::inRange(hsv_img, cv::Scalar(0, 0, 200), cv::Scalar(180, 255, 255), high_light_mask);

    //  灰度化处理
    cv::Mat gray_img;
    cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);

    //  二值化处理
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    //  膨胀操作
    cv::Mat dilated_img;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(high_light_mask, dilated_img, kernel);

    //  腐蚀操作
    cv::Mat eroded_img;
    cv::erode(dilated_img, eroded_img, kernel);

    //  漫水填充操作
    cv::Mat flood_filled_img = eroded_img.clone();
    cv::floodFill(flood_filled_img, cv::Point(0, 0), cv::Scalar(255));

    //  反转漫水填充的结果
    cv::Mat flood_filled_inv;
    cv::bitwise_not(flood_filled_img, flood_filled_inv);

    // 显示所有处理后的图像
    cv::imwrite("../resources/high_light_image.jpg", high_light_mask); // 高亮区域掩膜
    cv::imwrite("../resources/Gray_Image.jpg", gray_img);
    cv::imwrite("../resources/Binary_Image.jpg", binary_img);
    cv::imwrite("../resources/Dilated_Image.jpg", dilated_img);
    cv::imwrite("../resources/Eroded_Image.jpg", eroded_img);
    cv::imwrite("../resources/Flood_Filled_Image.jpg", flood_filled_img);
    cv::imwrite("../resources/Flood_Filled_Inverted.jpg", flood_filled_inv);
     
     //图像旋转 35 度
    // 获取图像中心
    cv::Point2f center(image.cols / 2.0F, image.rows / 2.0F);
    // 获取旋转矩阵 (角度为 35 度，缩放因子为 1)
    cv::Mat rot_matrix = cv::getRotationMatrix2D(center, 35, 1.0);
    // 计算旋转后的图像边界大小
    cv::Rect bbox = cv::RotatedRect(center, image.size(), 35).boundingRect();
    // 调整旋转矩阵，考虑平移
    rot_matrix.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rot_matrix.at<double>(1, 2) += bbox.height / 2.0 - center.y;
    // 应用仿射变换（旋转）
    cv::Mat rotated_img;
    cv::warpAffine(image, rotated_img, rot_matrix, bbox.size());
   //保存
    cv::imwrite("../resources/Rotated Image.jpg", rotated_img);

    //剪切
   int width = image.cols;
    int height = image.rows;
    // 计算左上角 1/4 区域的宽和高
    int new_width = width / 2;  // 新宽度为原图的一半
    int new_height = height / 2; // 新高度为原图的一半

    //  裁剪左上角区域
    cv::Mat cropped_img = image(cv::Rect(0, 0, new_width, new_height)); // 使用 ROI 进行裁剪

    // 保存裁剪后的图像
   cv::imwrite("../resources/Cropped Image.jpg",cropped_img);
   // 创建一个白色背景的图像
    cv::Mat img = cv::Mat::zeros(400, 400, CV_8UC3);
    img.setTo(cv::Scalar(255, 255, 255)); // 设置背景为白色

    // 1. 绘制火车的车身（蓝色矩形）
    int rect_width = 150;
    int rect_height = 100;
    cv::Rect train_body(125, 200, rect_width, rect_height); // 车身的位置和大小
    cv::rectangle(img, train_body, cv::Scalar(255, 0, 0), -1); // 蓝色填充

    // 2. 绘制火车头的圆形（蓝色），和矩形相切
    int engine_radius = rect_width / 2;
    cv::Point engine_center(200, 200); // 圆心
    cv::circle(img, engine_center, engine_radius, cv::Scalar(255, 240, 200), -1); // 蓝色填充火车头圆形

    // 3. 绘制车轮（黑色圆形）
    int wheel_radius = 25;
    cv::circle(img, cv::Point(160, 320), wheel_radius, cv::Scalar(0, 0, 0), -1); // 左轮子
    cv::circle(img, cv::Point(240, 320), wheel_radius, cv::Scalar(0, 0, 0), -1); // 右轮子

    // 4. 绘制火车的烟囱（黑色细长矩形）
    int chimney_width = 20;
    int chimney_height = 50;
    // 计算烟囱的y坐标，使得烟囱和圆形顶部相切
    int chimney_y = engine_center.y - engine_radius - chimney_height; 
    cv::Rect chimney(engine_center.x - chimney_width / 2, chimney_y, chimney_width, chimney_height); // 烟囱的位置和大小
    cv::rectangle(img, chimney, cv::Scalar(0, 0, 0), -1); // 黑色填充

    // 5. 绘制托马斯的眼睛（白色圆形）
    cv::Point left_eye_center(180, 180);
    cv::Point right_eye_center(220, 180);
    int eye_radius = 15;
    cv::circle(img, left_eye_center, eye_radius, cv::Scalar(255, 255, 255), -1); // 左眼白色
    cv::circle(img, right_eye_center, eye_radius, cv::Scalar(255, 255, 255), -1); // 右眼白色

    // 6. 绘制托马斯的眼珠（黑色小圆形）
    cv::circle(img, left_eye_center, 5, cv::Scalar(0, 0, 0), -1); // 左眼黑色眼珠
    cv::circle(img, right_eye_center, 5, cv::Scalar(0, 0, 0), -1); // 右眼黑色眼珠

    // 7. 绘制托马斯的嘴巴（黑色弧线）
    cv::ellipse(img, cv::Point(200, 230), cv::Size(25, 15), 0, 0, 180, cv::Scalar(0, 0, 0), 2); // 黑色嘴巴，弧线形

  
    // 在头顶绘制文字
    std::string text = "Thomas";
    cv::putText(img, text, cv::Point(160, 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2); // 黑色文字
   // 8. 计算红色外轮廓点集，确保包围所有元素
    int min_x = std::min(125, engine_center.x - engine_radius); // 左边界
    int max_x = std::max(125 + rect_width, engine_center.x + engine_radius); // 右边界
    int min_y = std::min(chimney_y, 200); // 上边界（包含烟囱）
    int max_y = std::max(320 + wheel_radius, 200 + rect_height); // 下边界（包含车轮）

    // 绘制红色的外轮廓线
    std::vector<cv::Point> contour_points;
    contour_points.push_back(cv::Point(min_x, min_y)); // 左上角
    contour_points.push_back(cv::Point(max_x, min_y)); // 右上角
    contour_points.push_back(cv::Point(max_x, max_y)); // 右下角
    contour_points.push_back(cv::Point(min_x, max_y)); // 左下角

    // 绘制红色外轮廓
    cv::polylines(img, contour_points, true, cv::Scalar(0, 0, 255), 2); // 红色外轮廓线

    //绘制红色的 bounding box
    cv::Rect bounding_box(min_x, min_y, max_x - min_x, max_y - min_y); // 使用上下左右的边界点计算bounding box
    cv::rectangle(img, bounding_box, cv::Scalar(0, 0, 255), 2); // 绘制红色bounding box

    //  显示结果
   
    cv::imwrite("../resources/Thomas the Train with Full Red Contour and Bounding Box.jpg", img);
    // 等待按键
    cv::waitKey(0);

    return 0;}
