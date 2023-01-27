import pathlib
import typing as t

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.datasets import tiny_imagenet, TinyImageNetDataset
from src.classifiers.models import DenseNet, ResNet, VGG

"""
    We use models from models.py with frozen backbones that we fine tune on the tiny imagenet dataset.
    Normally we should train on tiny imagenet training set and evaluate on validation set.
    We evaluate the accuracy of the fine tuned models on the corrupted datasets.
    We then look at how corruptions coming from different models affect this accuracy.
"""


def train(model, dataset, optimizer, epochs):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(dataset.loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target.argmax(dim=1))
            loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataset.loader.dataset)} ({100. * batch_idx / len(dataset.loader):.0f}%)]\tLoss: {loss.item():.6f}')

def evaluate(model, dataset):
    model.eval()
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data, target in dataset.loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  
        correct += pred.eq(target.argmax(dim=1, keepdim=True).view_as(pred)).sum().item()
    return correct / len(dataset.dataset)

if __name__ == '__main__':  
    # Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = ResNet().to(device)
    train(resnet, tiny_imagenet, torch.optim.Adam(resnet.parameters(), lr=0.001), 3)
    print(evaluate(resnet, tiny_imagenet))
    #Now evaluate on corruptions