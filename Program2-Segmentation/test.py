from modeling.deeplab import DeepLab
from modeling.unet import Unet
import torch

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())

    model = Unet(in_ch=3, out_ch=1)
    input = torch.rand(2, 3, 256, 256)
    output = model(input)
    print(output.size())