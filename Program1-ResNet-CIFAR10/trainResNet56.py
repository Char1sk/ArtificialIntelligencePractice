import torch
import torch.nn as nn
from torchvision import transforms

import ResNet56
import mydatasets


def loadData():
    cifar10path = '.'
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = mydatasets.MyCifar10(cifar10path, True, data_transforms)
    train_loader = mydatasets.MyDataLoader(train_dataset, 16)
    test_dataset = mydatasets.MyCifar10(cifar10path, False, data_transforms)
    test_loader = mydatasets.MyDataLoader(test_dataset, 16)
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
            print(f'    Batch: {batch+1:>4}, Loss:{loss.item():>7.6f}, AvgAcc:{100*correct/totalSize:>6.4f}%')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Begin Training")
    for epoch in range(100):
        print(f"Epoch:{epoch+1:>3}, learning rate = {0.01}")
        trainLoss, trainAcc = train(trainLoader, model, lossFunction, optimizer, device)
        testLoss, testAcc = test(testLoader, model, lossFunction, device)
        with open('./result.txt', 'a') as f:
            f.write(f'Epoch{epoch+1:>3d}\n')
            f.write(f'    Train: Loss {trainLoss:>7.6f}, Acc {trainAcc:>6.4f}%\n')
            f.write(f'    Test:  Loss {testLoss :>7.6f}, Acc {testAcc :>6.4f}%\n')
        torch.save(model.state_dict(), './saves/model.pth')
    print("Done")


if __name__ == '__main__':
    main()
