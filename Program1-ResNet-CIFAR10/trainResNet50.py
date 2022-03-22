import torch
import torch.nn as nn
from torchvision import transforms

import ResNet50
import mydatasets


def loadData():
    cifar10path = '.'
    data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    train_dataset = mydatasets.MyCifar10(cifar10path, True, data_transforms)
    train_loader = mydatasets.MyDataLoader(train_dataset, 16)
    test_dataset = mydatasets.MyCifar10(cifar10path, False, data_transforms)
    test_loader = mydatasets.MyDataLoader(test_dataset, 16)
    return (train_loader, test_loader)


def train(trainLoader, model, lossFunction, optimizer, device):
    model.train()
    correct, totalSize = 0, 0
    for batch, (data, label) in enumerate(trainLoader):
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
        totalSize += label.shape[0]
        correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        # print
        if (batch+1) % 10 == 0:
            print(f'    Batch: {batch+1:>4}, Loss:{loss.item():7.6f}, AvgAcc:{100*correct/totalSize:6.4f}%')


def test(testLoader, model, lossFunction, device):
    model.eval()
    with torch.no_grad():
        tloss, correct = 0, 0
        batchCount = 0
        totalCount = len(testLoader)
        for batch, (data, label) in enumerate(testLoader):
            batchCount += 1
            # convert
            data, label = data.to(device), label.to(device)
            label = label.to(torch.int64)
            # calculate
            pred = model(data)
            loss = lossFunction(pred, label)
            # add
            tloss += loss.item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    tloss /= batchCount
    correct /= totalCount
    # print
    print(f'Test: Acc: {correct}, Loss:{tloss}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    trainLoader, testLoader = loadData()

    model = ResNet50.resnet50().to(device)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Begin Training")
    for epoch in range(100):
        print(f"Epoch:{epoch+1:>3}, learning rate = {0.01}")
        train(trainLoader, model, lossFunction, optimizer, device)
        test(testLoader, model, lossFunction, device)
        torch.save(model.state_dict(), './saves/model.pth')
    print("Done")

    # print('iter1')
    # for i, (d, l) in enumerate(testLoader):
    #     print(i)
    # print('iter2')
    # for i, (d, l) in enumerate(testLoader):
    #     print(i)


if __name__ == '__main__':
    main()
    # lossfun = nn.CrossEntropyLoss()
    # pred = torch.tensor([
    #     [1., 0.],
    #     [0., 1.]
    # ])
    # # label = torch.tensor([
    # #     [1., 0.],
    # #     [0., 1.]
    # # ])
    # label = torch.tensor([
    #     0, 1
    # ])
    # print(lossfun(pred, label))
