import torch.nn as nn


class BasicBlock(nn.Module):
    r"""BasicBlock
    2 conv, 3x3, 3x3
    """
    def __init__(self, inChannel, outChannel, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inChannel, outChannel, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannel)

        self.conv2 = nn.Conv2d(outChannel, outChannel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannel)

        self.relu = nn.ReLU()

        self.downSample = None
        if stride != 1 or inChannel != outChannel:
            self.downSample = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannel)
            )

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downSample is not None:
            shortcut = self.downSample(x)

        out += shortcut
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    r"""BottleNeck
    3 conv, 1x1, 3x3, 1x1
    """
    def __init__(self, inChannel, outChannel, stride=1):
        super(BottleNeck, self).__init__()

        midChannel = outChannel // 4

        self.conv1 = nn.Conv2d(inChannel, midChannel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(midChannel)

        self.conv2 = nn.Conv2d(midChannel, midChannel, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midChannel)

        self.conv3 = nn.Conv2d(midChannel, outChannel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outChannel)

        self.relu = nn.ReLU()

        self.downSample = None
        if stride != 1 or inChannel != outChannel:
            self.downSample = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannel)
            )

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downSample is not None:
            shortcut = self.downSample(x)

        out += shortcut
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    r"""ResNet50
    1conv, 1pool, convs, 1pool, 1fc
    """
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        # Bx3x224x224 -> Bx64x112x112
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # Bx64x112x112 -> Bx64x56x56
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # BasicBlock Layers, for 18 34
        if block == BasicBlock:
            # Bx64x56x56 -> Bx64x56x56
            # self.conv2 = nn.Sequential(
            #     BasicBlock(64, 64, 1),
            #     BasicBlock(64, 64, 1),
            #     BasicBlock(64, 64, 1)
            # )
            self.conv2 = self._make_layer(block, 64, 64, 1, layers[0])
            # Bx64x56x56 -> Bx128x28x28
            # self.conv3 = nn.Sequential(
            #     BasicBlock(64, 128, 2),
            #     BasicBlock(128, 128, 1),
            #     BasicBlock(128, 128, 1),
            #     BasicBlock(128, 128, 1)
            # )
            self.conv3 = self._make_layer(block, 64, 128, 2, layers[1])
            # Bx128x28x28 -> Bx256x14x14
            # self.conv4 = nn.Sequential(
            #     BasicBlock(128, 256, 2),
            #     BasicBlock(256, 256, 1),
            #     BasicBlock(256, 256, 1),
            #     BasicBlock(256, 256, 1),
            #     BasicBlock(256, 256, 1),
            #     BasicBlock(256, 256, 1)
            # )
            self.conv4 = self._make_layer(block, 128, 256, 2, layers[2])
            # Bx256x14x14 -> Bx512x7x7
            # self.conv5 = nn.Sequential(
            #     BasicBlock(256, 512, 2),
            #     BasicBlock(512, 512, 1),
            #     BasicBlock(512, 512, 1)
            # )
            self.conv5 = self._make_layer(block, 256, 512, 2, layers[3])
        # BasicBlock Layers, for 50 101 152
        elif block == BottleNeck:
            # Bx64x56x56 -> Bx256x56x56
            # self.conv2 = nn.Sequential(
            #     BottleNeck(64, 256, 1),
            #     BottleNeck(256, 256, 1),
            #     BottleNeck(256, 256, 1)
            # )
            self.conv2 = self._make_layer(block, 64, 256, 1, layers[0])
            # Bx256x56x56 -> Bx512x28x28
            # self.conv3 = nn.Sequential(
            #     BottleNeck(256, 512, 2),
            #     BottleNeck(512, 512, 1),
            #     BottleNeck(512, 512, 1),
            #     BottleNeck(512, 512, 1)
            # )
            self.conv3 = self._make_layer(block, 256, 512, 2, layers[1])
            # Bx512x28x28 -> Bx1024x14x14
            # self.conv4 = nn.Sequential(
            #     BottleNeck(512, 1024, 2),
            #     BottleNeck(1024, 1024, 1),
            #     BottleNeck(1024, 1024, 1),
            #     BottleNeck(1024, 1024, 1),
            #     BottleNeck(1024, 1024, 1),
            #     BottleNeck(1024, 1024, 1)
            # )
            self.conv4 = self._make_layer(block, 512, 1024, 2, layers[2])
            # Bx1024x14x14 -> Bx2048x7x7
            # self.conv5 = nn.Sequential(
            #     BottleNeck(1024, 2048, 2),
            #     BottleNeck(2048, 2048, 1),
            #     BottleNeck(2048, 2048, 1)
            # )
            self.conv5 = self._make_layer(block, 1024, 2048, 2, layers[3])
        # Bx512x7x7 -> Bx512x1x1
        # Bx2048x7x7 -> Bx2048x1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Bx512x1x1 -> Bx512
        # Bx2048x1x1 -> Bx2048
        self.flatten = nn.Flatten(1)
        if block == BasicBlock:
            # Bx512 -> Bx10
            self.fc = nn.Linear(512, 10)
        elif block == BottleNeck:
            # Bx2048 -> Bx10
            self.fc = nn.Linear(2048, 10)

    def _make_layer(self, block, inChannel, outChannel, stride, count):
        layers = []
        layers.append(block(inChannel, outChannel, stride))
        for _ in range(1, count):
            layers.append(block(outChannel, outChannel, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])
