from pickletools import optimize
from tkinter import N
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import yaml
from modeling.deeplab import DeepLab
from modeling.loss import FocalLossV1
from dataset.custom_dataset import MyDataset
from utils import calMIOU, calPA


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


# load config.yml globally
with open("./config.yml", "r") as f:
    config = yaml.load(f.read(), Loader=yaml.CLoader)

def loadData():
    datapath = './iccv09Data'
    data_transforms = transforms.Compose([
        transforms.RandomCrop(32), #随机裁剪
        transforms.RandomHorizontalFlip(), # 翻转图片
        transforms.RandomVerticalFlip(),
        # transforms.RandomPerspective(),
        transforms.ToTensor()
    ])
    train_dataset = MyDataset(datapath, True, data_transforms)
    train_loader = DataLoader(train_dataset, config['TRAIN_BATCH_SIZE'])
    test_dataset = MyDataset(datapath, False, data_transforms)
    test_loader = DataLoader(test_dataset, config['TEST_BATCH_SIZE'])
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
        if config['DEBUG_MODE']:
            print(pred.shape, label.shape, loss)
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
    nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f'[{nowTime}]    Loss:{tloss/totalBatch:>7.6f}, mIOU:{tiou/totalCount:>6.4f}')
    return (tloss/totalBatch, tiou/totalCount)

def init_dir(config):
    if not os.path.exists(config['SAVE_MODEL_DIR']):
        os.mkdir(config['SAVE_MODEL_DIR'])
    if not os.path.exists(config['RESULT_DIR']):
        os.mkdir(config['RESULT_DIR'])

def get_result_file_name(config):
    n = 1
    name = "DeepLab-" + config['BACKBONE'] + '-' + str(n) + '.txt'
    while os.path.exists(config['RESULT_DIR'] + os.sep + name):
        n += 1
        name = "DeepLab-" + config['BACKBONE'] + '-' + str(n) + '.txt'
    return config['RESULT_DIR'] + os.sep + name

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    trainLoader, testLoader = loadData()

    model = DeepLab(backbone=config['BACKBONE'], output_stride=16, num_classes=9).to(device)
    if config['LOSS'] == 'CE':
        lossFunction = nn.CrossEntropyLoss()
    # lossFunction = FocalLossV1()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.8)

    init_dir(config)
    print("Begin Training")
    init_msg = ">> DeepLabv3\n".format(config['BACKBONE'])
    result_file_name = get_result_file_name(config)
    with open(result_file_name, "w") as f:
        f.write(init_msg)
        f.write('---------------config.yml-------------------\n')
        for key in config:
            f.write(str(key)+':'+str(config[key])+'\n')
        f.write('--------------------------------------------\n\n')

    for epoch in range(config['NUM_EPOCH']):
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f"Epoch:{epoch+1:>3}, learning rate = {now_lr}")

        trainLoss, trainAcc = train(trainLoader, model, lossFunction, optimizer, device)
        testLoss, testAcc = test(testLoader, model, lossFunction, device)
            
        nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(result_file_name, "a") as f:
            f.write(f'[{nowTime}] Epoch{epoch+1:>3d}, lr = {now_lr}\n')
            f.write(f'    Train: Loss {trainLoss:>7.6f}, mIOU {trainAcc:>6.4f}\n')
            f.write(f'    Test:  Loss {testLoss :>7.6f}, mIOU {testAcc :>6.4f}\n')

        torch.save(model.state_dict(), './saves/model.pth')
        lr_scheduler.step(trainLoss)

    print("Done")


if __name__ == '__main__':
    main()
