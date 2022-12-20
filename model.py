import pytorch_lightning as pl
#from pytorch_lightning import loggers as pl_loggers
#from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
import torchvision
import blocks as blk
import time
import numpy as np
from accuracy_metrics import Hamming_distance, overall_acc


class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss1 = torch.nn.CrossEntropyLoss()
        self.loss2 = torch.nn.MultiLabelSoftMarginLoss()
    def forward(self, y_hat, y):
        y = y.float()
        loss_atmos = self.loss1(y_hat[:, :4], y[:, :4])
        loss_ground = self.loss2(y_hat[:, 4:], y[:, 4:])
        return loss_atmos+loss_ground

class PlanetModel(pl.LightningModule):
    def __init__(self, model = None):
        super().__init__()
        self.model = model
        self.loss = custom_loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)  ## A VOIR SI ON VEUT CA COMME LOSS
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat, y

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", overall_acc(y_hat, y, threshold=3))
        self.log("hamming_dist", Hamming_distance(y_hat, y, threshold=3))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
        
class testModel(nn.Module):
    def __init__(self, max_channels=1024):
        super().__init__()
        layer1 = blk.DoubleConv(in_c=4, out_c=8, mid_c=6, kernel_size=3) # 256x256
        layer2 = blk.DownSample(in_c=8, out_c=12, kernel_size=3, nb_conv=2)    # 128x128
        layer3 = blk.DownSample(in_c=12, out_c=16, kernel_size=3, nb_conv=2)    # 64x64
        layer4 = blk.DownSample(in_c=16, out_c=32, kernel_size=3, nb_conv=2)    #32x32
        layer5 = blk.DownSample(in_c=32, out_c=64, kernel_size=3, nb_conv=2)    #16x16
        layer6 = blk.DownSample(in_c=64, out_c=128, kernel_size=3, nb_conv=2)    #8x8
        layer7 = blk.DownSample(in_c=128, out_c=256, kernel_size=3, nb_conv=2)    #4x4
        layer8 = blk.DownSample(in_c=256, out_c=max_channels, kernel_size=3, nb_conv=2)    #2x2
        layer9 = blk.DownSample(in_c=max_channels, out_c=max_channels, kernel_size=3, nb_conv=2)    #1x1
        classifier = blk.Classifier(in_f=max_channels)
        self.model = nn.Sequential(layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, classifier)

    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module):
    def __init__(self, depth=18):
        super().__init__()
        if depth==18:
            backbone = torchvision.models.resnet18(weights='DEFAULT')
        elif depth==34:
            backbone = torchvision.models.resnet34(weights='DEFAULT')
        elif depth==50:
            backbone = torchvision.models.resnet50(weights='DEFAULT')
        elif depth==101:
            backbone = torchvision.models.resnet101(weights='DEFAULT')
        elif depth==152:
            backbone = torchvision.models.resnet152(weights='DEFAULT')
        else:
            print("Use a value in [18, 34, 50, 101, 152]. Using 18 right now...")
            backbone = torchvision.models.resnet18(weights='DEFAULT')
        
        classifier = blk.Classifier(in_f=512)
        self.model = nn.Sequential(*list(backbone.children())[:-1], classifier)
    
    def forward(self, x):
        return self.model(x)
