################ CHANGED ##################
# Move test into main function
# Move loadimage loadmask into class
# Make transform be a parameter
# Make transform at 320x240

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import numpy as np
import random

class MyDataset(Dataset):
    
    def __init__(self, path, isTrain, imageTransform, maskTransform, flipProb=0):
        self.path = path
        self.isTrain = isTrain
        self.imageTransform = imageTransform
        self.maskTransform = maskTransform
        self.flipProb = flipProb
        
        # self.horizonPath = 'horizons.txt'
        self.listPath = 'trainList.txt' if self.isTrain else 'testList.txt'
        # self.listPath1 = 'trainList.txt'
        # self.listPath2 = 'testList.txt'
        self.imagesPath = 'images'
        self.labelsPath = 'labels'
        
        self.names = []
        with open(os.path.join(self.path, self.listPath), 'r') as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                self.names.append(line)
        # with open(os.path.join(self.path, self.listPath1), 'r') as f:
        #     while True:
        #         line = f.readline().strip()
        #         if not line:
        #             break
        #         self.names.append(line)
        # with open(os.path.join(self.path, self.listPath2), 'r') as f:
        #     while True:
        #         line = f.readline().strip()
        #         if not line:
        #             break
        #         self.names.append(line)
        # with open(os.path.join(self.path, self.horizonPath)) as f:
        #     while True:
        #         line = f.readline()
        #         if not line:
        #             break
        #         m = [i for i in line.split(' ')]
        #         isStandard = int(m[1]) == 320 and int(m[2]) == 240
        #         if (self.isTrain and isStandard) or \
        #                 (not self.isTrain and not isStandard):
        #             self.names.append(m[0])
        
    def __getitem__(self, index):
        fn = self.names[index]
        if random.random() < self.flipProb:
            # print('!!!!!!!!!!!!!!!!!!!!!!!')
            doTrans = True
        else:
            doTrans = False
        image = self.loadimage(fn, doTrans)
        mask = self.loadmask(fn, doTrans)
        # return {'img': image, 'mask': mask,}
        # print(image.shape, mask.shape)
        return (image, mask)
    
    def __len__(self):
        return len(self.names)
    
    def loadimage(self, path, doTrans):
        img_pil = Image.open(os.path.join(self.path, self.imagesPath, f'{path}.jpg'))
        if doTrans:
            # print('!!!!!!!!!!!!!!!!')
            img_pil = transforms.RandomHorizontalFlip(1)(img_pil)
        img_pil = self.imageTransform(img_pil)
        return img_pil
    
    def loadmask(self, path, doTrans):
        mask=[]
        with open(os.path.join(self.path, self.labelsPath, f'{path}.regions.txt')) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                m = [int(i) if len(i)==1 else 8 for i in line.split(' ')]
                # m = [int(i) for i in line.split(' ')]
                mask.append(m)
        # mask = Image.fromarray(np.array(mask))
        mask = Image.fromarray(np.int8(np.array(mask)))
        if doTrans:
            # print('!!!!!!!!!!!!!!!!')
            mask = transforms.RandomHorizontalFlip(1)(mask)
        mask = self.maskTransform(mask)
        return mask
    

if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    fpath = './iccv09Data/'
    
    train_set = MyDataset(fpath, True, transform, transform)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    test_set = MyDataset(fpath, False, transform, transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    count = 0
    for ii, sample in enumerate(train_loader):
        count = count+1
    print(count)
    
    count = 0
    for ii, sample in enumerate(test_loader):
        count = count+1
    print(count)
