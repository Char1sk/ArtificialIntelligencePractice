# 人工智能实践-第二次项目

## 数据集选择

- stanford

## 模型选择

- deeplabv3(+)

- UNet

## 实验方向记录

- 计算mIOU时去除未知类：mIOU提高一点，但不影响训练过程
- 在Loss的CrossEntropy里忽略未知类，没啥用，后面严重过拟合
- LR0.01，迅速50+/36

## 实验设计

| 实验编号  |       模型                    | Epoch      | Batch Size |  Loss Function   | Optimizer | Learning Rate |  output_stride   | Pre-Trained |
| :------: | :---------------------------: | :--------: | :--------: | :--------------: | :-------: | :-----------: | :--------------: | :---------: |
|   (1)    | Deeplabv3(backbone:resnet101) |    100     |     4      | CrossEntropyLoss |    SGD    |     0.05      |        16        |  resnet101  |
|   (2)    | Deeplabv3(backbone:resnet101) |     60     |     4      | CrossEntropyLoss |    SGD    |     0.01      |        16        |  resnet101  |
|   (3)    | Deeplabv3(backbone:resnet101) |     60     |     4      |    Focalloss     |    SGD    |     0.01      |        16        |  resnet101  |
|   (4)    | Deeplabv3(backbone:resnet101) |     60     |     8      |    Focalloss     |    SGD    |     0.01      |        16        |  resnet101  |
|   (5)    | Deeplabv3(backbone:resnet101) |     60     |     12     |    Focalloss     |    SGD    |     0.01      |        16        |  resnet101  |
|   (6)    | Deeplabv3(backbone:resnet101) |     60     |     16     |    Focalloss     |    SGD    |     0.01      |        16        |  resnet101  |
|   (7)    | Deeplabv3(backbone:resnet101) |     60     |     4      |    Focalloss     |    SGD    |     0.01      |        8         |  resnet101  |
|   (8)    | Deeplabv3(backbone:resnet101) |     60     |     4      |    Focalloss     |    SGD    |     0.01      |        32        |  resnet101  |
|   (9)    |    Deeplabv3(backbone:drn)    |     60     |     4      | CrossEntropyLoss |    SGD    |     0.01      |        16        |     drn     |

## 实验结果
| 实验编号  |        结果文件           |
| :------: | :----------------------: |
|   (1)    | result.txt |
|   (2)    | result2.txt |
|   (3)    | result_focalloss.txt |
|   (4)    | result_batchsize8.txt |
|   (5)    | result_batchsize12.txt |
|   (6)    | result_batchsize16.txt |
|   (7)    | result_os8.txt |
|   (8)    | result_os32.txt |
|   (9)    | result_drn.txt |

## 实验总结

- Unet实验
  - LR：0.10, 0.01, 0.001
  - BS：2 4 6
- Unet数据处理
  - 数据集划分：效果拔群
  - 归一操作：resizes randomcrop padcrop scalecrop
  - 增强操作：shuffle flip norm gauss 都没啥用
- DeepLabV3的消融实验：
  - LR：0.10, 0.05, 0.01，越小越好，有质变
  - Loss：CE, FL，提升不小
  - BS：4, 8, 12, 16，越低越好
  - OS：8, 16, 32，越低越好
- DeepLabV3的Backbone：
  - resnet也还好，有0.409
  - drn老版本0.389，新版本0.419
  - mobilenet相当不错，有0.425
  - xception较捞，有概率涨停