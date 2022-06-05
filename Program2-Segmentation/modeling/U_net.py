import torch
import torch.nn as nn
import torch.nn.functional as F

#class DoubleConv(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(DoubleConv, self).__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(in_ch, out_ch, 3, padding=1),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(out_ch, out_ch, 3, padding=1),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True)
#        )

#    def forward(self, input):
#        return self.conv(input)


#class Unet(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(Unet, self).__init__()

#        self.conv1 = DoubleConv(in_ch, 32)
#        self.pool1 = nn.MaxPool2d(2)
#        self.conv2 = DoubleConv(32, 64)
#        self.pool2 = nn.MaxPool2d(2)
#        self.conv3 = DoubleConv(64, 128)
#        self.pool3 = nn.MaxPool2d(2)
#        self.conv4 = DoubleConv(128, 256)
#        self.pool4 = nn.MaxPool2d(2)
#        self.conv5 = DoubleConv(256, 512)
#        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#        self.conv6 = DoubleConv(512, 256)
#        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#        self.conv7 = DoubleConv(256, 128)
#        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#        self.conv8 = DoubleConv(128, 64)
#        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#        self.conv9 = DoubleConv(64, 32)
#        self.conv10 = nn.Conv2d(32, out_ch, 1)

#    def forward(self, x):
#        c1 = self.conv1(x)
#        p1 = self.pool1(c1)
#        c2 = self.conv2(p1)
#        p2 = self.pool2(c2)
#        c3 = self.conv3(p2)
#        p3 = self.pool3(c3)
#        c4 = self.conv4(p3)
#        p4 = self.pool4(c4)
#        c5 = self.conv5(p4)
#        up_6 = self.up6(c5)
#        merge6 = torch.cat([up_6, c4], dim=1)
#        c6 = self.conv6(merge6)
#        up_7 = self.up7(c6)
#        merge7 = torch.cat([up_7, c3], dim=1)
#        c7 = self.conv7(merge7)
#        up_8 = self.up8(c7)
#        merge8 = torch.cat([up_8, c2], dim=1)
#        c8 = self.conv8(merge8)
#        up_9 = self.up9(c8)
#        merge9 = torch.cat([up_9, c1], dim=1)
#        c9 = self.conv9(merge9)
#        c10 = self.conv10(c9)
#        out = nn.Sigmoid()(c10)
#        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)
 
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
 
class Up(nn.Module):
    """Upscaling then double conv"""
 
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
 
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
 
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
 
 
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
 
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
 
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256)
    net = Unet(3,1)
    print(net(a).shape)
