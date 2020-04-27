#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18

import pytorch_lightning as pl


# In[11]:


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        # not the best model...
        self.model = resnet18(pretrained=True)

    def forward(self, x):
        # called with self(x)
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.AdamW(self.parameters(), lr=0.01)

    def train_dataloader(self):
        t = transforms.Compose([
            transforms.Pad(224),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return DataLoader(ImageFolder("./dataset/train", transform=t), batch_size = 64, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        t = transforms.Compose([
                transforms.Pad(224),
                transforms.RandomSizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])
        return DataLoader(ImageFolder("./dataset/test", transform=t), batch_size = 128, drop_last=True, pin_memory=True, num_workers=4)


# In[12]:


mnist_model = Model()

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)    
trainer.fit(mnist_model)   


# In[13]:


trainer.test()


# In[ ]:




