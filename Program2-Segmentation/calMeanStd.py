import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset.custom_dataset import MyDataset


datapath = './iccv09Data'
image_transforms = transforms.Compose([
    transforms.ToTensor()
])
mask_transforms = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = MyDataset(datapath, False, image_transforms, mask_transforms)
train_loader = DataLoader(train_dataset, 1)

def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0,0,0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    print(num_batches)
    print(channels_sum)
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2) **0.5

    return mean,std

mean,std = get_mean_std(train_loader)

print(mean)
print(std)

# 715
# tensor([344.4247, 350.7574, 339.5501])
# tensor([0.4817, 0.4906, 0.4749])
# tensor([0.2500, 0.2495, 0.2752])
