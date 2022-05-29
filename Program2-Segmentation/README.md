# 人工智能实践-第二次项目

## 数据集选择

- stanford

## 模型选择

- deeplabv3(+)

- UNet

## 实验方向记录

- 计算mIOU时去除未知类：mIOU提高一点，但不影响训练过程
- 在Loss的CrossEntropy里忽略未知类，开局好了点但貌似练不动
