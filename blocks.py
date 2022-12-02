import torch
import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super.__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding=kernel_size//2)
        self.b_norm1 = nn.BatchNorm2d(num_features=out_c)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
        self.one_conv = nn.Sequential(self.conv1, self.b_norm1, self.lrelu)

    def forward(self, x):
        return self.one_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None, kernel_size=3):
        super.__init__()
        if mid_c is None:
            mid_c = out_c
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=kernel_size, padding=kernel_size//2)
        self.b_norm1 = nn.BatchNorm2d(num_features=mid_c)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=mid_c, out_channels=out_c, kernel_size=kernel_size, padding=kernel_size//2)
        self.b_norm2 = nn.BatchNorm2d(num_features=out_c)
        self.dbl_conv = nn.Sequential(self.conv1, self.b_norm1, self.lrelu, self.conv2, self.b_norm2, self.lrelu)

    def forward(self, x):
        return self.dbl_conv(x)

class DownSample(nn.Module):
    # Use maxpool then <nb_conv> layers (either dbl or simple)
    def __init__(self, in_c, out_c, mid_c=None, kernel_size=3, nb_conv=2):
        super.__init__()
        if nb_conv not in [1, 2]:
            nb_conv = 2
        if nb_conv == 1:
            self.downsample = nn.Sequential(nn.MaxPool2d(kernel_size=2), 
                                            SimpleConv(in_c=in_c, out_c=out_c, kernel_size=kernel_size))
        else:
            self.downsample = nn.Sequential(nn.MaxPool2d(kernel_size=2), 
                                            DoubleConv(in_c=in_c, out_c=out_c, mid_c=mid_c, kernel_size=kernel_size))
    def forward(self, x):
        return self.downsample(x)

class AtmosClassifier(nn.Module):
    # given a laten layer representation, gives prediction for atmospheric labels
    def __init__(self, in_f):
        super.__init__()
        self.linear = nn.Linear(in_features=in_f, out_features=4)
        self.softmax = nn.Softmax()
        self.atmos_classifier = nn.Sequential(self.linear, self.softmax)

    def forward(self, x):
        x = x.flatten()
        return self.atmos_classifier(x)

class GroundClassifier(nn.Module):
    # given a laten layer representation, gives prediction for ground labels
    def __init__(self, in_f):
        super.__init__()
        self.linear = nn.Linear(in_features=in_f, out_features=13)
        self.ground_classifier = nn.Sequential(self.linear)

    def forward(self, x):
        x = x.flatten()
        return self.ground_classifier(x)

class Classfier(nn.Module):
    # given a laten layer representation, gives prediction for atmospheric labels
    def __init__(self, in_f):
        super.__init__()
        self.ground_classifier = GroundClassifier(in_f)
        self.atmos_classifier = AtmosClassifier(in_f)

    def forward(self, x):
        x = x.flatten()
        return torch.cat((self.atmos_classifier(x), self.ground_classifier(x)))


