# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:13:18 2024

@author: Mostafa
"""
# %% importing tensorboard's summarywriter and torch
import torch
from torch.utils.tensorboard import SummaryWriter

# %% begining of training and logging
writer = SummaryWriter(log_dir="./runs/file1/")
# %% creating fake data
x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())
# %% train loop definition
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


train_model(10)
writer.flush()

# %% closing summarywriter
writer.close()
