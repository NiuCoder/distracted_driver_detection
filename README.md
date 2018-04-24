# distracted_driver_detection（走神司机检测）
本项目是2016年Kaggle上的一个[竞赛](https://www.kaggle.com/c/state-farm-distracted-driver-detection/)，目的在于对连续采集的将近80000张图像进行分类，项目的评判标准是所有图像的logloss。本文提供了一种解决方案，利用vgg16、vgg19、resnet50作为预训练模型，然后进行模型集成，最终在Leader board上的排名为127/1440。
## 运行环境
本项目运行的环境为Intel Core i7-8700K + Nvidia Geforce GTX 1080Ti + 32G内存 + 1T硬盘，项目依赖库详见environment.yaml。
## 其他说明
1. 代码及运行结果见distracted_driver_detection.ipynb。
2. 利用单个预训练模型训练10个分类器的时间约6小时，单个模型的预测时间为1.5小时，三种预训练模型的耗时接近。
3. 数据集的获取可以从Kaggle官网[比赛页](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)获取，注意科学上网，实测在aws上秒下。也可以访问我的[网盘](https://pan.baidu.com/s/1M_Huwrw5_tOM4F5WDuvSdA)， 密码：javp。

