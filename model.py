import pytorch_lightning as pl
#from pytorch_lightning import loggers as pl_loggers
#from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.functional as F
import torch.nn as nn
import torchvision


class PlanetModel(pl.LightningModule):
    def __init__(self, model = None):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)  ## A VOIR SI ON VEUT CA COMME LOSS
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat, y

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
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
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) #256
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3) # 126
            self.maxpool = nn.MaxPool2d(kernel_size=2) # 128
            self.enc = nn.Seqential(self.conv1, self.maxpool, self.conv2)
            self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3) # 128
            self.upsample = nn.Upsample(scale_factor=2) # 256
            self.dec = nn.Sequential(self.deconv1, self.upsample)

    def forward(self, x):
        if self.pretrained:
            x = self.model(x)
            return x
        else:
            None