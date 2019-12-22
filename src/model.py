import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import scipy.io
import torchvision.models as models
import torchvision.transforms as transforms


class ToTensor():
    def __call__(self, sample):
        return torch.from_numpy(sample)

class Dataset_load(data.Dataset):
    
    def __init__(self,samples,labels,transform = None):
        self.samples = samples
        self.labels = labels
        self.transform = transform()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        if self.transform is not None:
            return self.transform(self.samples[idx]), self.transform(self.labels[idx])

class VGG16Conv4(nn.Module):
            def __init__(self):
                super(AlexNetConv4, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(original_model.features.children())[:-3]
                )
            def forward(self, x):
                x = self.features(x)
                return x


min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

img = io.imread('000000.bin.png')
img = transform_pipeline(img)
vgg16 = models.vgg16(pretrained = True)
vgg16_fe = list(vgg16.children())[:-1][0]
prediction = vgg16_fe(img)
print(prediction.shape)


