import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F

#data

#image 2d data
def transfrom(reshape=None, transpose=None):
    return reshape,transpose

def read_data(data_dir, label_dir, transfrom): 
        np_data = np.load(data_dir)
        np_label = np.load(label_dir).reshape(-1)
        reshape, transpose = transfrom
        if transpose:
            np_data = np_data.transpose(*transpose)
        if reshape:
            num = np_data.shape[0]
            np_data = np_data.reshape((num,*reshape))
        data = torch.from_numpy(np_data).float()
        label = torch.from_numpy(np_label).int()
        return data,label 
#dataset
class npdata(Dataset):
    def __init__(self, img, label):
        super(Dataset)
        self.img = img
        self.label = label
        assert len(self.img) == len(self.label)
    def __getitem__(self,idx):
        return self.img[idx], self.label[idx]

    def __len__(self):
        return len(self.img)

class npdata_3(Dataset):
    def __init__(self, data1, data2, label):
        super(Dataset)
        self.data1 = data1
        self.data2 = data2
        self.label = label
        assert len(self.data1) == len(self.label)
    def __getitem__(self,idx):
        return self.data1[idx],self.data2[idx], self.label[idx]
    def __len__(self):
        return len(self.data1)

#Model

#resnet
class Residual18(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
def resnet_block18(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual18(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual18(num_channels, num_channels))
    return blk
class Residual50(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=1, padding=1)
        if use_1x1conv:
            self.conv0 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv0 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv0:
            X = self.conv0(X)
        Y += X
        return F.relu(Y)
def resnet_block50(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual50(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual50(num_channels, num_channels))
    return blk

#1-d SPP-net
class spatial_pyramid_pooling(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size
    def forward(self, X):
        num = X.shape[0]
        for i in self.pool_size:
            max_pool = nn.AdaptiveMaxPool1d(i)
            interx = max_pool(X)
            if i == 1:
                spp = interx.view(num, -1)
            else:
                spp = torch.cat((spp, interx.view(num, -1)), 1)
        return spp

#LSTM
