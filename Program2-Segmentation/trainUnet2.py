import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import numpy as np
from torch.nn import functional as F
#from FocalLoss import *
from modeling.unet import Unet
from dataset.custom_dataset import MyDataset

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, 
                                  reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def calMIOU(pred, label):
    nc = pred.shape[1]
    _, index = pred.max(dim=1)
    # print(index.shape, label.shape)
    index, label = index.cpu().numpy(), label.cpu().numpy()
    miou = 0.0
    for i, l in zip(index, label):
        # print(i.shape, l.shape)
        hist = np.zeros((nc, nc), dtype=int)
        for ri, rl in zip(i, l):
            # print(nc*rl.astype(int)+ri)
            hist += np.bincount(nc*rl.astype(int)+ri, minlength=nc**2).reshape(nc, nc)
        # print('hist:', hist)
        # print(np.diag(hist))
        # print((hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))
        miou += (np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1) ).mean()
        # print('miou:', miou)
    return miou


def loadData():
    datapath = './iccv09Data'
    data_transforms = transforms.Compose([
        # transforms.RandomCrop(32, padding=4), #随机裁剪
        #transforms.RandomHorizontalFlip(p=0.5), # 翻转图片
        #transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])
    train_dataset = MyDataset(datapath, True, data_transforms)
    train_loader = DataLoader(train_dataset, 6)
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

        optimizer.zero_grad()
        # calculate
        pred = model(data)
        loss = lossFunction(pred, label)
        # print(pred.shape, label.shape, loss)
        # optimize
        
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

    exp_name = 'U-net(bs:6)'

    trainLoader, testLoader = loadData()

    model = Unet(3, 9).to(device)
    #lossFunction = nn.BCELoss()
    #optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    milestones=[100, 150]
    
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    print("Begin Training")
    for epoch in range(60):
        print(f"Epoch:{epoch+1:>3}, learning rate = {optimizer.param_groups[0]['lr']}")
        # if (epoch==31):
        #     lr = lr /10
        #     optimizer.param_groups[0]["lr"]=lr
        # if (epoch==47):
        #     lr = lr /10
        #     optimizer.param_groups[0]["lr"]=lr 

        trainLoss, trainAcc = train(trainLoader, model, lossFunction, optimizer, device)
        testLoss, testAcc = test(testLoader, model, lossFunction, device)
        #scheduler.step()
        print('Train: Loss {:>7.6f}, mIOU {:>6.4f}    Test: Loss {:>7.6f}, mIOU {:>6.4f}\n'.format(trainLoss, trainAcc, testLoss, testAcc))
        with open('./result1.txt', 'a') as f:
            nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())            
            f.write("#{}# [{}] Epoch{:>3d}\n".format(exp_name, nowTime, epoch+1))
            f.write('#{}#    Train: Loss {:>7.6f}, mIOU {:>6.4f}\n'.format(exp_name, trainLoss, trainAcc))
            f.write('#{}#    Test: Loss {:>7.6f}, mIOU {:>6.4f}\n'.format(exp_name, testLoss, testAcc))
        #torch.save(model.state_dict(), './saves/model.pth')
    print("Done")


if __name__ == '__main__':
    main()

