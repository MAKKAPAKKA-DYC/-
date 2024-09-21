#include <opencv2/opencv.hpp>
#include <iostream>

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
    cv::Mat mean_blurred_image;
    cv::blur(image, mean_blurred_image, cv::Size(5, 5)); // 核大小为 5x5
    cv::imwrite("../resources/mean_blurred_image.png", mean_blurred_image); // 保存均值滤波后的图像
     
     // 应用高斯滤波
    cv::Mat gaussian_blurred_image;
    cv::GaussianBlur(image, gaussian_blurred_image, cv::Size(5, 5), 0); // 核大小为 5x5，标准差为 0
    cv::imwrite("../resources/gaussian_blurred_image.png", gaussian_blurred_image); // 保存高斯滤波后的图像

    // =======================
    // 方法一：使用 HSV 提取红色区域
    // =======================

    // 定义红色的 HSV 范围（由于红色在色环的两端，通常分两部分来处理红色）
    cv::Mat mask1, mask2, red_mask_hsv;
    // 低范围的红色 (0° - 15°)
    cv::inRange(hsv_image, cv::Scalar(0, 60, 60), cv::Scalar(15, 255, 255), mask1);
    // 高范围的红色 (165° - 180°)
    cv::inRange(hsv_image, cv::Scalar(165, 60,60), cv::Scalar(180, 255, 255), mask2);

    // 合并两个掩码
    cv::bitwise_or(mask1, mask2, red_mask_hsv);

    // 使用掩码提取红色区域
    cv::Mat red_area_hsv;
    cv::bitwise_and(image, image, red_area_hsv, red_mask_hsv);

     cv::imwrite("../resources/red_area_hsv.png", red_area_hsv);  // 保存结果

    // =======================
    // 方法二：使用 BGR 提取红色区域
    // =======================

    // 创建一个空的掩码，大小与原图相同
    cv::Mat red_mask_bgr = cv::Mat::zeros(image.size(), CV_8UC1);

    // 逐像素遍历图像，提取红色区域
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            // 获取 BGR 值
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);

            int B = pixel[0]; // 蓝色通道值
            int G = pixel[1]; // 绿色通道值
            int R = pixel[2]; // 红色通道值

            // 判断是否为红色区域：R 通道较高，B 和 G 通道较低
            if (R > 120 && G < 120 && B < 120) {
                red_mask_bgr.at<uchar>(i, j) = 255; // 将红色区域标记为白色 (255)
            }
        }
    }

    // 使用掩码提取红色区域
    cv::Mat red_area_bgr;
    cv::bitwise_and(image, image, red_area_bgr, red_mask_bgr);

    cv::imwrite("../resources/red_area_bgr.png", red_area_bgr);  // 保存结果
    
    // 找到红色区域的轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(red_mask_bgr, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
   
   // 创建掩膜，设定红色的 BGR 范围
    cv::Mat red_mask;
    cv::inRange(image, cv::Scalar(0, 0, 120), cv::Scalar(120, 120, 255), red_mask);  // 通过调整 BGR 范围检测红色

    // 创建一个副本图像用于绘制轮廓和 bounding box
    cv::Mat output_img = image.clone();

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
