import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import numpy as np

from modeling.deeplab import DeepLab
from dataset.custom_dataset import MyDataset
from utils import calMIOU, calPA


def loadData():
    datapath = './iccv09Data'
    data_transforms = transforms.Compose([
        # transforms.RandomCrop(32, padding=4), #随机裁剪
        # transforms.RandomHorizontalFlip(), # 翻转图片
        transforms.ToTensor()
    ])
    train_dataset = MyDataset(datapath, True, data_transforms)
    train_loader = DataLoader(train_dataset, 4)
    test_dataset = MyDataset(datapath, False, data_transforms)
    test_loader = DataLoader(test_dataset, 1)
    return (train_loader, test_loader)


def train(trainLoader, model, lossFunction, optimizer, device):
    model.train()
    tloss, totalBatch = 0, 0
    tiou, totalSize = 0.0, 0
    for batch, (data, label) in enumerate(trainLoader):
        totalBatch += 1
        # convert
        data, label = data.to(device), label.to(device)
        label = label.to(torch.int64).squeeze(dim=1)
        # calculate
        pred = model(data)
        loss = lossFunction(pred, label)
        # print(pred.shape, label.shape, loss)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # stats
        tloss += loss.item()
        totalSize += label.shape[0]
        tiou += calMIOU(pred, label)
        # print
        if (batch+1) % 10 == 0:
            nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f'[{nowTime}]    Batch: {batch+1:>4}, Loss:{loss.item():>7.6f}, mIOU:{tiou/totalSize:>6.4f}')
    return (tloss/totalBatch, tiou/totalSize)


def test(testLoader, model, lossFunction, device):
    model.eval()
    tloss, totalBatch = 0, 0
    tiou = 0.0
    totalCount = len(testLoader)
    with torch.no_grad():
        for batch, (data, label) in enumerate(testLoader):
            totalBatch += 1
            # convert
            data, label = data.to(device), label.to(device)
            label = label.to(torch.int64).squeeze(dim=1)
            # calculate
            pred = model(data)
            loss = lossFunction(pred, label)
            # add
            tloss += loss.item()
            tiou += calMIOU(pred, label)
    return (tloss/totalBatch, tiou/totalCount)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    trainLoader, testLoader = loadData()

    model = DeepLab(backbone='resnet', output_stride=16, num_classes=9).to(device)
    lossFunction = nn.CrossEntropyLoss(ignore_index=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    print("Begin Training")
    for epoch in range(64):
        print(f"Epoch:{epoch+1:>3}, learning rate = {0.1}")

        trainLoss, trainAcc = train(trainLoader, model, lossFunction, optimizer, device)
        testLoss, testAcc = test(testLoader, model, lossFunction, device)
        with open('./result.txt', 'a') as f:
            nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f'[{nowTime}] Epoch{epoch+1:>3d}\n')
            f.write(f'    Train: Loss {trainLoss:>7.6f}, mIOU {trainAcc:>6.4f}\n')
            f.write(f'    Test:  Loss {testLoss :>7.6f}, mIOU {testAcc :>6.4f}\n')
        torch.save(model.state_dict(), './saves/model.pth')
    print("Done")


if __name__ == '__main__':
    main()
