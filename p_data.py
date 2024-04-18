import time
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

std = np.array([0.5, 0.5, 0.5])
mean = np.array([1, 1, 1])

def denormalize(tensors):
    for c in range(3):
        tensors[:, c].add_(mean[c]).mul_(std[c])
    return torch.clamp(tensors, 0, 255)

class ImageDataset(Dataset):
    def __init__(self, root, ratio):
        self.hr_shape = 257
        self.ratio = ratio
        self.transform = transforms.Compose([transforms.Resize((self.hr_shape, self.hr_shape))])
        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        x = Image.open(self.files[index % len(self.files)])
        
        #transforms
        x = np.asarray(x).astype("f")
        x = x.transpose(2, 0, 1) #from (hr_shape, hr_shape, 3) to (3, hr_shape, hr_shape)
        x = x / 127.5 - 1.0 #normalize
        x = torch.from_numpy(x)
        x = self.transform(x) #resize
        
        #mask
        edge = int(self.hr_shape * self.ratio) + random.randint(-4, 4)
        M = np.zeros((self.hr_shape, self.hr_shape))
        M[:, -edge:] = 1
        M = np.transpose(M) #extent right -> extent bottom
        M = M[np.newaxis, :, :]
        z = x * (1 - M) 

        M = torch.from_numpy(M)
        z_M = torch.cat([z, M])

        return {"real_img": x, "masked_img": z, 'mask': M, 'g_input': z_M}

    def __len__(self):
        return len(self.files)
