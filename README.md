# 第三次任务

## 识别 R 标以及锤子，绘制方形和圆形

**cmakelist里面set(SRC ./cc)这行代码有错误，定义了一个变量 SRC，但在后续代码中并没有使用到它。所以直接删除这行代码以避免混淆。**

循环处理每一帧的图像，计算风车旋转的当前状态。

获取风车中心点的位置，根据风车的旋转角度和锤子到风车中心的半径，通过三个关键点计算锤子在每一帧图像中的具体位置，绘制边界框。

“R”标记的检测与绘制

将当前帧转换为灰度图，生成一个匹配结果矩阵，如果找到了匹配度超过设定阈值的“R”标记，绘制“R”标记的边界框。

以两个方框的中心点的连线为直径绘制圆。

在windmill.hpp中，为了方便类外部访问和计算，将start_time 挪进 public，并且引入新的public : getR_center()、getCurrentAngle()、calculatePoint() 来对风车的旋转和位置进行外部计算

修改后的代码见main.cpp和windmill.hpp

<img src="https://github.com/MAKKAPAKKA-DYC/-/blob/main/assets/%E6%96%B9%E5%BD%A2%E5%9C%86%E5%BD%A2.png" alt="风车检测结果" width="500" height="300"/>

## 拟合转速

相关代码和结果已私发
