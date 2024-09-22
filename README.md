# 第二次任务
<h2>1.配置 C++ 开发环境</h2>
	<p style="text-indent:2em">在 Ubuntu 系统中安装vscode编译器和一系列插件。
<h2>2.安装 OpenCV 库</h2>
<p style="text-indent:2em">根据任务书使用编译安装的方式安装 OpenCV 的 C++ 版本。</p>
<h2>3.组织项目结构</h2>
<p style="text-indent:2em">使用cmakelist.txt组织项目结构如下图。</p>
<img src= "https://github.com/MAKKAPAKKA-DYC/-/blob/assets/%E7%AC%AC%E4%BA%8C%E5%91%A8%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%84.png" width="600" height="300">
	<img src="https://github.com/MAKKAPAKKA-DYC/-/blob/main/assets/%E7%AC%AC%E4%BA%8C%E5%91%A8%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%84%E2%80%98%E2%80%99.png" width="600" height="300">

<h2>4.实现基础图像处理操作</h2>
<p style="text-indent:2em">(1)根据chatgpt的提示和调配一系列参数和变量，得到一系列处理后的图片见<strong>resources</strong>文件夹。</p>
<p style="text-indent:2em">(2)以下几点：</p>
<p style="text-indent:2em">在应用均值滤波时，使用多个不同大小的核（如 3x3、5x5、7x7 等）对图像进行均值滤波。找到信噪比最高的滤波结果时，将该图像保存。</p>
<p style="text-indent:2em">在应用高斯滤波时,chatgpt表示寻找合适的核一般使用试探法，较小的核适合保留更多细节，较大的核可以更强地去除噪声。再次寻找信噪比最高的滤波结果并保存图像</p>
<p style="text-indent:2em">在提取红色区域时，首先使用hsv方法对经过高斯滤波处理后的图片进行红色区域提取，调节红色范围，发现提取到的红色区域仍不完全；后使用bgr方法，提取到了更为完全的红色区域。</p>

