# 人工智能实践-第一次项目

## 数据集选择

- CIFAR10
  - 数据集分为10个类别，分别为...
  - 分为训练集50000张图，测试集10000张图
  - 文件含有：batches.meta, data_batch_1, ...
    - batches.meta：包含类别信息...
    - data_batch_x：训练集，每个10000张
    - test_batch：测试集，含有10000张
  - 文件格式：
    - pickle类型文件...
    - 10000x3072规模的npy，每一行...，通道组成...(CIFAR10官网有说)
- 数据预处理
  - 描述怎么预处理的，如如何将其转换为图片BxCxHxW的
- 数据增广
  - 描述怎么数据增广的

## 模型选择

- ResNet50 (on ImageNet)
  - 模型结构：
    - conv1+bn+relu：Bx3x224x224 -> Bx64x112x112
    - maxpool：Bx64x112x112 -> Bx64x56x56
    - conv2：Bx64x56x56 -> Bx256x56x56
      - 3BottleNeck
    - conv3：Bx256x56x56 -> Bx512x28x28
      - 4BottleNeck
    - conv4：Bx512x28x28 -> Bx1024x14x14
      - 6BottleNeck
    - conv5：Bx1024x14x14 -> Bx2048x7x7
      - 3BottleNeck
    - avgpool：Bx2048x7x7 -> Bx2048x1x1
    - flatten+fc：Bx2048x1x1 -> Bx10
  - 输入输出：
    - 接收输入：Bx3x224x224
    - 产生输出：Bx10
  - 残差连接：BottleNeck, identity + projection (Type B)
- ResNet56
  - 模型结构：1 + 2x9 + 2x9 + 2x9 + 1 ...

## 参数设置

- 可以参照实验设计/结果来设置
- 参数初始化
  - 初始化方式1：(如论文中的初始化)
  - 初始化方式2：(如Xavier初始化)
  - ...
- 学习率策略
  - 学习率策略1：(如ReduceLROnPlateau)
  - 学习率策略2：(如某些Epoch上衰减0.1)
  - ...
- ...

## 实验设计

- 模型选择
  - ResNet50 (on ImageNet)
    - 含有BottleNeck结构
    - 适用于大图输入(224x224)
  - ResNet56 (on CIFAR10)
    - 无BottleNeck结构
    - 适用于小图输入(32x32)
  - 两者层数相近，易于比较结构带来的差异
  - 可以考虑再研究一下？
    - ResNet34? (ImageNet 无BottleNeck)
    - ResNet38? (CIFAR10  无BottleNeck)
    - 再次排除BottleNeck因素，且层数相近
- 参数1(如初始化方式)
  - 取值1
  - 取值2
  - ...
- 参数2(如学习率策略)
  - 取值1
  - 取值2
  - ...
- ...

## 实验结果

| 实验编号 |       模型        | Batch Size |  Loss Function   | Optimizer | Learning Rate | Data Augmentation | Pre-Trained |
| :------: | :---------------: | :--------: | :--------------: | :-------: | :-----------: | :---------------: | :---------: |
|   (1)    | ResNet50(224x224) |     16     | CrossEntropyLoss |   Adam    |     0.01      |       None        |    False    |
|   (2)    |  ResNet50(32x32)  |     8      | CrossEntropyLoss |   Adam    |     0.01      |       None        |    False    |
|   (3)    |  ResNet56(6n+2)   |     16     | CrossEntropyLoss |   Adam    |     0.01      |       None        |    False    |

| 实验编号 | Train Acc | Test Acc | Train Loss | Test Loss |
| :------: | :-------: | :------: | :--------: | :-------: |
|   (1)    |  90.16%   |  73.02%  |    0.07    |   1.29    |
|   (2)    |  97.65%   |  78.67%  |    0.07    |   1.29    |
|   (3)    |   忘了    |   忘了   |    忘了    |   忘了    |

注：实验1我重新跑个完整版，实验3我上local系统上查一下再改











