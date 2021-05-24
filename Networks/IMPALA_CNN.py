import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = self.relu(x_in)
        x = F.relu(self.conv1(x))
        x_out = self.conv2(x)
        return x_in + x_out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, block_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, block_channels, 3, 1, 1)
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.res1 = ResBlock(block_channels)
        self.res2 = ResBlock(block_channels)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.max_pool(x)
        x = self.res1(x)
        return self.res2(x)


class ImpalaCnn64(nn.Module):
    def __init__(self, in_channels, num_outputs, base_channels=16):
        super(ImpalaCnn64, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, base_channels)
        self.conv_block2 = ConvBlock(base_channels, 2*base_channels)
        self.conv_block3 = ConvBlock(2*base_channels, 2*base_channels)

        self.fc1 = nn.Linear(8*8*2*base_channels, 256)
        self.Actor = nn.Linear(256, num_outputs)
        self.Critic = nn.Linear(256, 1)
        self.SM = nn.Softmax(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.reshape(x.size(0), -1)
        self.state = self.relu(self.fc1(x))

        value = self.Critic(self.state)
        self.prob = self.SM(self.Actor(self.state))
        dist = Categorical(probs=self.prob)
        return dist, value