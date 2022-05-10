import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import numpy as np

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

def loadimage(path):
    img_pil = Image.open('./iccv09Data/images/'+path+'.jpg').convert('L')
    img_pil = transform(img_pil)
    return img_pil
def loadmask(path):
    mask=[]
    with open('./iccv09Data/labels/'+path+'.regions.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            m = [int(i) for i in line.split(' ')]
            mask.append(m)
    mask=Image.fromarray(np.array(mask))
    mask=transform(mask)
    return mask


names=[]
with open('.\iccv09Data\horizons.txt') as f:
    while True:
        line = f.readline()
        if not line:
            break
        m = [i for i in line.split(' ')]
        #if m[1] == 320 and m[2] == 240:
        names.append(m)

class MyDataset(Dataset):
    def __init__(self,data,loadimage,loadmask):
        self.images = data
        self.loadimage = loadimage
        self.loadmask = loadmask
    def __getitem__(self, index):
        fn = self.images[index][0]
        image = self.loadimage(fn)
        mask = self.loadmask(fn)
        #print(image.shape)
        #print(mask.shape)
        return {'img': image, 'mask':mask,}
    def __len__(self):
        return len(self.images)


train_size = 200
train_set = MyDataset(data=names[:train_size],loadimage=loadimage,loadmask=loadmask)
train_loader = DataLoader(train_set, batch_size=4,shuffle=True)

test_set = MyDataset(data=names[train_size:],loadimage=loadimage,loadmask=loadmask)
test_loader = DataLoader(test_set, batch_size=4,shuffle=True)

count = 0
for ii, sample in enumerate(train_loader):
    #print(sample['img'].shape)
    count = count+1
print(count)