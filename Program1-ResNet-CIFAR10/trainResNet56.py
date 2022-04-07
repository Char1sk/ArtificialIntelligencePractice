import torch
import torch.nn as nn
from torchvision import transforms
import time

import ResNet56
import mydatasets


def loadData():
    cifar10path = '.'
    data_transforms = transforms.Compose([

        transforms.RandomCrop(32, padding=4), #随机裁剪
        transforms.RandomHorizontalFlip(), # 翻转图片
        transforms.ToTensor()
    ])
    train_dataset = mydatasets.MyCifar10(cifar10path, True, data_transforms, True)
    train_loader = mydatasets.MyDataLoader(train_dataset, 128)
    test_dataset = mydatasets.MyCifar10(cifar10path, False, data_transforms)
    test_loader = mydatasets.MyDataLoader(test_dataset, 128)
    return (train_loader, test_loader)


def train(trainLoader, model, lossFunction, optimizer, device):
    model.train()
    tloss, totalBatch = 0, 0
    correct, totalSize = 0, 0
    for batch, (data, label) in enumerate(trainLoader):
        totalBatch += 1
        # convert
        data, label = data.to(device), label.to(device)
        label = label.to(torch.int64)
        # calculate
        pred = model(data)
        loss = lossFunction(pred, label)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # stats
        tloss += loss.item()
        totalSize += label.shape[0]
        correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        # print
        if (batch+1) % 10 == 0:
            nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f'[{nowTime}]    Batch: {batch+1:>4}, Loss:{loss.item():>7.6f}, AvgAcc:{100*correct/totalSize:>6.4f}%')
    return (tloss/totalBatch, correct/totalSize)


def test(testLoader, model, lossFunction, device):
    model.eval()
    tloss, totalBatch = 0, 0
    correct = 0
    totalCount = len(testLoader)
    with torch.no_grad():
        for batch, (data, label) in enumerate(testLoader):
            totalBatch += 1
            # convert
            data, label = data.to(device), label.to(device)
            label = label.to(torch.int64)
            # calculate
            pred = model(data)
            loss = lossFunction(pred, label)
            # add
            tloss += loss.item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    return (tloss/totalBatch, correct/totalCount)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    trainLoader, testLoader = loadData()

    model = ResNet56.resnet56().to(device)
    lossFunction = nn.CrossEntropyLoss()
    lr = 1e-1
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)

    print("Begin Training")
    for epoch in range(64):
        if epoch == 32:
            lr/=10
        elif epoch == 48:
            lr/=10

        print(f"Epoch:{epoch+1:>3}, learning rate = {lr}")
        trainLoss, trainAcc = train(trainLoader, model, lossFunction, optimizer, device)
        testLoss, testAcc = test(testLoader, model, lossFunction, device)
        with open('./result.txt', 'a') as f:
            nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f'[{nowTime}] Epoch{epoch+1:>3d}\n')
            f.write(f'    Train: Loss {trainLoss:>7.6f}, Acc {trainAcc:>6.4f}%\n')
            f.write(f'    Test:  Loss {testLoss :>7.6f}, Acc {testAcc :>6.4f}%\n')
        torch.save(model.state_dict(), './saves/model.pth')
    print("Done")


if __name__ == '__main__':
    main()
