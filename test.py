import torch
import torch.nn.function as F

import torch.nn as nn
import numpy as np
import cv2

class testNET(nn.Module):
    super(testNET, self).()
    def __init__(self):
        self.net = nn.sequental(
            nn.Conv2d(16,32, kernel=3, stride = 1, padding = 1),
            nn.BatchNormal2d(32),
            nn.relu()
        )
        self.fc = nn.linear(32, 10)
    def forward(self, input,target=None):
        x = self.net(input)
        x = self.fc(x)
        return x

def train(net,dataloader):
    opt = optimzer.sgd(momentum = 0.9, baselr = 0.0001)
    dataloader = dataloader()
    loss = nn.CElosss
    net = net()
    net.train()
    for image, label in dataloader:
        net.fit(image, label, optimzer, loss)



