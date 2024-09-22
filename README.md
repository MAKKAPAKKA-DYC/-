# 第二次任务
<h2>1.配置 C++ 开发环境</h2>
	<p style="text-indent:2em">在 Ubuntu 系统中安装vscode编译器和一系列插件。
<h2>2.安装 OpenCV 库</h2>
<p style="text-indent:2em">根据任务书使用编译安装的方式安装 OpenCV 的 C++ 版本。</p>
<h2>3.组织项目结构</h2>
<p style="text-indent:2em">使用cmakelist.txt组织项目结构如下图。</p>
<img src= "https://github.com/MAKKAPAKKA-DYC/-/blob/assets/%E7%AC%AC%E4%BA%8C%E5%91%A8%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%84.png" width="600" height="300">
	<img src="https://github.com/MAKKAPAKKA-DYC/-/blob/main/assets/%E7%AC%AC%E4%BA%8C%E5%91%A8%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%84%E2%80%98%E2%80%99.png" width="600" height="300">
<p style="text-indent:2em">根据(<a href="https://blog.csdn.net/weixin_42310154/article/details/118340458" title="示例网站" target="_blank">Github配置ssh key的步骤</a>)添加公钥，最终完成建立。</p>
<h2>4.实现基础图像处理操作</h2>
<p style="text-indent:2em">根据chatgpt的提示和调配一系列参数和变量，得到一系列处理后的图片见resources文件夹。</p>
<p style="text-indent:2em">
。
(2) 应用各种滤波操作
均值滤波：使用 cv2.blur(image, (kernel_size, kernel_size)) 实现均值滤波。
高斯滤波：使用 cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma) 实现高斯滤波。
(3) 特征提取
提取红色颜色区域：
HSV 方法：定义红色的 HSV 值区间，使用 cv2.inRange(hsv_image, lower_red, upper_red) 提取红色区域的掩码。
寻找红色的外轮廓：使用 cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 查找红色区域的外轮廓。
寻找红色的 bounding box：对轮廓使用 cv2.boundingRect(contour) 获取最小外接矩形的坐标和尺寸。
计算轮廓的面积：使用 cv2.contourArea(contour) 计算红色区域轮廓的面积。
(4) 提取高亮颜色区域并进行图形学处理
灰度化：同前，使用 cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)。
二值化：使用 cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY) 实现二值化。
膨胀：使用 cv2.dilate(binary_image, kernel, iterations) 对图像膨胀。
腐蚀：使用 cv2.erode(dilated_image, kernel, iterations) 对图像腐蚀。
漫水填充：使用 cv2.floodFill(image, mask, seed_point, new_color) 实现漫水填充。
(5) 图像绘制
绘制圆形、方形和文字：
使用 cv2.circle(image, center, radius, color, thickness) 绘制圆形。
使用 cv2.rectangle(image, pt1, pt2, color, thickness) 绘制矩形。
使用 cv2.putText(image, text, position, font, font_scale, color, thickness) 绘制文字。
绘制红色的外轮廓：使用 cv2.drawContours(image, contours, contourIdx, color, thickness) 绘制外轮廓。
绘制红色的 bounding box：使用 cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness) 绘制 bounding box。
(6) 图像处理
图像旋转 35 度：使用 cv2.getRotationMatrix2D(center, angle, scale) 生成旋转矩阵，然后使用 cv2.warpAffine(image, rotation_matrix, (width, height)) 实现旋转。
图像裁剪为左上角 1/4：确定裁剪区域的坐标 [0, 0, width//2, height//2]，然后用切片操作进行裁剪 image[:height//2, :width//2]。
