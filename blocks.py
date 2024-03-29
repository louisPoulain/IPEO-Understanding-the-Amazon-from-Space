import torch
import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding=kernel_size//2)
        self.b_norm1 = nn.BatchNorm2d(num_features=out_c, momentum=0.1)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
        self.one_conv = nn.Sequential(self.conv1, self.b_norm1, self.lrelu)

    def forward(self, x):
        return self.one_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None, kernel_size=3):
        super().__init__()
        if mid_c is None:
            mid_c = out_c
        conv1 = nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=kernel_size, padding=kernel_size//2)
        b_norm1 = nn.BatchNorm2d(num_features=mid_c, momentum=0.1)
        lrelu = nn.LeakyReLU(inplace=True)
        conv2 = nn.Conv2d(in_channels=mid_c, out_channels=out_c, kernel_size=kernel_size, padding=kernel_size//2)
        b_norm2 = nn.BatchNorm2d(num_features=out_c, momentum=0.1)
        self.dbl_conv = nn.Sequential(conv1, b_norm1, lrelu,conv2,b_norm2, lrelu)

    def forward(self, x):
        return self.dbl_conv(x)

class DownSample(nn.Module):
    # Use maxpool then <nb_conv> layers (either dbl or simple)
    def __init__(self, in_c, out_c, mid_c=None, kernel_size=3, nb_conv=2):
        super().__init__()
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
        super().__init__()
        self.linear = nn.Linear(in_features=in_f, out_features=4)
        self.softmax = nn.Softmax(dim=-1)
        self.atmos_classifier = nn.Sequential(self.linear, self.softmax)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.atmos_classifier(x)

class GroundClassifier(nn.Module):
    # given a laten layer representation, gives prediction for ground labels
    def __init__(self, in_f):
        super().__init__()
        linear = nn.Linear(in_features=in_f, out_features=13)
        sigmoid = nn.Sigmoid()
        self.ground_classifier = nn.Sequential(linear, sigmoid)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.ground_classifier(x)

class Classifier(nn.Module):
    # given a laten layer representation, gives prediction for atmospheric labels
    def __init__(self, in_f):
        super().__init__()
        self.ground_classifier = GroundClassifier(in_f)
        self.atmos_classifier = AtmosClassifier(in_f)

    def forward(self, x):
        return torch.cat((self.atmos_classifier(x), self.ground_classifier(x)), dim=1)


