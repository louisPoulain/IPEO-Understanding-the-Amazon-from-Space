import pytorch_lightning as pl
#from pytorch_lightning import loggers as pl_loggers
#from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.functional as F
import torchvision


class PlanetModel(pl.LightningModule):
    def __init__(self, pretrained = False):
        super().__init__()
        if pretrained:
            self.model = torchvision.models.resnet18(num_classes=21) ## TO BE CHANGED
        else:
            self.model = None ## TO BE CHANGED

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
        
