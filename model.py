import pytorch_lightning as pl
#from pytorch_lightning import loggers as pl_loggers
#from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.functional as F
import torch.nn as nn
import torchvision
import blocks as blk


class PlanetModel(pl.LightningModule):
    def __init__(self, model = None):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.MultiLabelSoftMarginLoss(y_hat, y)  ## A VOIR SI ON VEUT CA COMME LOSS
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat, y

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.MultiLabelSoftMarginLoss(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", (y_hat.argmax(1) == y).float().mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
        
class testModel(nn.Module):
    def __init__(self, pretrained:bool=False):
        self.pretrained = pretrained
        if self.pretrained:
            self.model = torchvision.models.AlexNet() ## A CHANGER
        else:
            layer1 = blk.DoubleConv(in_c=3, out_c=8, mid_c=4, kernel_size=3) # 256x256
            layer2 = blk.DownSample(in_c=8, out_c=16, kernel_size=3, nb_conv=1)    # 128x128
            layer3 = blk.DownSample(in_c=8, out_c=16, kernel_size=3, nb_conv=1)    # 64x64
            layer4 = blk.DownSample(in_c=16, out_c=32, kernel_size=3, nb_conv=1)    #32x32
            layer5 = blk.DownSample(in_c=32, out_c=64, kernel_size=3, nb_conv=1)    #16x16
            layer6 = blk.DownSample(in_c=128, out_c=256, kernel_size=3, nb_conv=1)    #8x8
            layer7 = blk.DownSample(in_c=256, out_c=512, kernel_size=3, nb_conv=1)    #4x4
            layer8 = blk.DownSample(in_c=512, out_c=1024, kernel_size=3, nb_conv=1)    #2x2
            layer9 = blk.DownSample(in_c=1024, out_c=1024, kernel_size=3, nb_conv=1)    #1x1
            classifier = blk.Classfier(in_f=1024)
            self.model = nn.Sequential(layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, classifier)


    def forward(self, x):
        return self.model(x)