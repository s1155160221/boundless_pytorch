import time
import numpy as np
import torch
from torchvision.models import inception_v3

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class GatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,):
        super(GatedConv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.sigmoid = torch.nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = torch.nn.functional.elu(x) * self.gated(mask)
        x = torch.nn.functional.instance_norm(x)
        return x

#Generator
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gconv1 = GatedConv(4, 32, 5, 1, padding=2) #257x257
        self.gconv2 = GatedConv(32, 64, 3, 2, padding=1) #129x129
        self.gconv3 = GatedConv(64, 64, 3, 1, padding=1) #129x129
        self.gconv4 = GatedConv(64, 128, 3, 2, padding=1) #65x65
        self.gconv5 = GatedConv(128, 128, 3, 1, padding=1) #65x65
        self.gconv6 = GatedConv(128, 128, 3, 1, padding=1) #65x65

        self.gconv7 = GatedConv(128, 128, 3, 1, dilation=2, padding=2)  #65x65
        self.gconv8 = GatedConv(128, 128, 3, 1, dilation=4, padding=4)  #65x65
        self.gconv9 = GatedConv(128, 128, 3, 1, dilation=8, padding=8)  #65x65
        self.gconv10 = GatedConv(128, 128, 3, 1, dilation=16, padding=16)  #65x65

        self.gconv11 = GatedConv(256, 128, 3, 1, padding=1) #65x65
        self.gconv12 = GatedConv(256, 128, 3, 1, padding=1) #65x65
        self.gconv13 = GatedConv(256, 64, 3, 1, padding=1) #129x129
        self.gconv14 = GatedConv(128, 64, 3, 1, padding=1) #129x129
        self.gconv15 = GatedConv(128, 32, 3, 1, padding=1) #257x257
        self.gconv16 = GatedConv(64, 16, 3, 1, padding=1) #257x257

        self.conv17 = torch.nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = self.gconv1(x)
        out2 = self.gconv2(out1)
        out3 = self.gconv3(out2)
        out4 = self.gconv4(out3)
        out5 = self.gconv5(out4)
        out6 = self.gconv6(out5)

        out = self.gconv7(out6)
        out = self.gconv8(out)
        out = self.gconv9(out)
        out = self.gconv10(out)

        out = torch.cat([out, out6], dim=1) #skip connection 
        out = self.gconv11(out)
        out = torch.cat([out, out5], dim=1) #skip connection 
        out = self.gconv12(out)
        out = torch.cat([out, out4], dim=1) #skip connection 
        out = torch.nn.functional.interpolate(out, (129, 129), mode='bilinear', align_corners=False) #upsample
        out = self.gconv13(out)
        out = torch.cat([out, out3], dim=1) #skip connection 
        out = self.gconv14(out)
        out = torch.cat([out, out2], dim=1) #skip connection 
        out = torch.nn.functional.interpolate(out, (257, 257), mode='bilinear', align_corners=False) #upsample
        out = self.gconv15(out)
        out = torch.cat([out, out1], dim=1) #skip connection 
        out = self.gconv16(out)

        out = self.conv17(out)
        return torch.clamp(out, min=-1, max=1)

#Discriminator
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2)),
            torch.nn.LeakyReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)),
            torch.nn.LeakyReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)),
            torch.nn.LeakyReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            torch.nn.LeakyReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            torch.nn.LeakyReLU(),
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            torch.nn.LeakyReLU(),
        )
        self.layer7 = torch.nn.utils.spectral_norm(torch.nn.Linear(6400, 256, bias=False))

        self.linear_phi = torch.nn.utils.spectral_norm(torch.nn.Linear(256, 1, bias=False))
        self.linear_c = torch.nn.utils.spectral_norm(torch.nn.Linear(2048, 256, bias=False))
        
        self.flatten = Flatten() 

    def forward(self, x, m, c):
        out = torch.cat([x, m], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        phi = self.layer7(self.flatten(out))

        d1 = self.linear_phi(phi)
        c = self.linear_c(c)
        d2 = (phi * c).sum(1, keepdim=True)
        return torch.add(d1, d2)
    
#Extractor
class InceptionExtractor(torch.nn.Module):
    def __init__(self):
        super(InceptionExtractor, self).__init__()
        self.inception = inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1', transform_input=True, aux_logits=True)
        self.inception.fc = torch.nn.Identity()

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, (299, 299), mode='bilinear', align_corners=False)
        x = self.inception(x)
        x = torch.nn.functional.normalize(x)
        return x