import torch.nn as nn
import torch.nn.functional as F
import torchvision

"""
    Models with possibly frozen backbones
"""


class DenseNet(nn.Module):
    def __init__(self, freeze_backbone = False):
        super(DenseNet, self).__init__()
        self.densenet = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.DEFAULT)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.densenet.parameters():
                param.requires_grad = False
        self.densenet.classifier = nn.Sequential(
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 200),
        )
    
    def forward(self, x):
        return self.densenet(x)
    
    def get_features(self, x):
        return self.densenet.features(x)
    

class VGG(nn.Module):
    def __init__(self, freeze_backbone = False):
        super(VGG, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.vgg.parameters():
                param.requires_grad = False
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 200),
        )
    
    def forward(self, x):
        return self.vgg(x)
    
    def get_features(self, x):
        return self.vgg.features(x)


class ResNet(nn.Module):
    def __init__(self, freeze_backbone = False):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 200),
        )
    
    def forward(self, x):
        return self.resnet(x)
    
    def get_features(self, x):
        return self.resnet.conv1(x)
