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


from main import Model


if __name__ == "__main__":
    f = "epoch=9.ckpt"
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
