#!/usr/bin/env python
# coding: utf-8

__constant__ = ["SwishJitAutoFn"]

import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from torchvision.models import resnet18, resnet34

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams

        self.model = torch.hub.load(
            'rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)

        for i, param in enumerate(self.model.parameters()):
            print(i)
            if i < 20:  # 18
                param.requires_grad = False
        self.model.classifier = nn.Linear(1280, 25)
        self.model.classifier.weight.requires_grad = True
        self.model.classifier.bias.requires_grad = True

    def prepare_data(self):
        super().prepare_data()

        train_transforms = transforms.Compose([
            transforms.Pad(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=.12, contrast=.03, saturation=.03),
            transforms.RandomAffine(5, scale=(0.98, 1.08), shear=(5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.train_dataset = ImageFolder(
            "./dataset/train", transform=train_transforms)

        val_transforms = transforms.Compose([
            transforms.Pad(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.val_dataset = ImageFolder(
            "./dataset/val", transform=val_transforms)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'Train/loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        # implement your own
        out = self(x)
        loss = F.cross_entropy(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # all optional...
        # return whatever you need for the collation function test_epoch_end
        output = OrderedDict({
            'val_loss': loss,
            'val_acc': torch.tensor(val_acc),  # everything must be a tensor
        })

        # return an optional dict
        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        log = {'Val/loss': avg_loss, 'Val/acc': avg_acc}

        return OrderedDict({
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'progress_bar': log,
            "log": log})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["max_lr"], weight_decay=self.hparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.hparams["max_lr"], steps_per_epoch=len(self.train_dataset), epochs=5, cycle_momentum=False)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, drop_last=True, num_workers=4)


if __name__ == "__main__":
    f = "efficient_net_b0_v3_new_dataset.ckpt"
    model = Model.load_from_checkpoint(f"./checkpoints/{f}")

    model = model.model

    model = nn.Sequential(
        model,
        nn.Softmax(1)
    )

    model.eval()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("android/model.pt")

    print(model(example).shape)
