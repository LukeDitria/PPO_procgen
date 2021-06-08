import torch.nn as nn
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

class FcResBlock(nn.Module):
    def __init__(self, first_block, **kwargs):
        super(FcResBlock, self).__init__()
        self.fc1 = nn.Linear(kwargs["state_size"], int(kwargs["state_size"] / kwargs["scale_down"]))
        self.fc2 = nn.Linear(int(kwargs["state_size"] / kwargs["scale_down"]), kwargs["state_size"])
        self.first_block = first_block
        self.relu = nn.ReLU()
        self.do = nn.Dropout(0.5)

    def forward(self, x_in):
        if not self.first_block:
            x = self.relu(x_in)
        else:
            x = x_in
        x = self.do(self.relu(self.fc1(x)))
        x = self.fc2(x) + x_in
        return x

class FcBottleNeckBlock(nn.Module):
    def __init__(self, first_block, **kwargs):
        super(FcBottleNeckBlock, self).__init__()
        self.fc1 = nn.Linear(kwargs["state_size"], int(kwargs["state_size"] / kwargs["scale_down"]))
        self.fc2 = nn.Linear(int(kwargs["state_size"] / kwargs["scale_down"]), kwargs["state_size"])
        self.first_block = first_block
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = self.relu(self.fc1(x_in))
        return self.relu(self.fc2(x))

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

class ImpalaCnn64Invert(nn.Module):
    def __init__(self, in_channels, num_outputs, base_channels=16):
        super(ImpalaCnn64Invert, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, 2*base_channels)
        self.conv_block2 = ConvBlock(2*base_channels, base_channels)
        self.conv_block3 = ConvBlock(base_channels, base_channels)

        self.fc1 = nn.Linear(8*8*base_channels, 256)
        self.Actor = nn.Linear(256, num_outputs)
        self.Critic = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, train=False):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.reshape(x.size(0), -1)
        self.state = self.relu(self.fc1(x))

        value = self.Critic(self.state)
        logits = self.Actor(self.state)
        dist = Categorical(logits=logits)
        return dist, value


class ShallowNet(nn.Module):
    def __init__(self, in_channels, num_outputs, base_channels=16):
        super(ShallowNet, self).__init__()
        print("ShallowNet")
        self.conv1 = nn.Conv2d(in_channels, 16*base_channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(16*base_channels, 64, 3, 2, 1)
        self.do = nn.Dropout(0.5)

        self.fc1 = nn.Linear(16*16*64, 256)
        self.Actor = nn.Linear(256, num_outputs)
        self.Critic = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, train=False):
        x = self.do(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        self.state = self.relu(self.fc1(x))

        value = self.Critic(self.state)
        logits = self.Actor(self.state)
        dist = Categorical(logits=logits)
        return dist, value


class SmallThickNet(nn.Module):
    def __init__(self, in_channels, num_outputs, base_channels=16):
        super(SmallThickNet, self).__init__()
        print("SmallThickNet")
        self.conv1 = nn.Conv2d(in_channels, 32*base_channels, 5, 4, 2)
        self.conv2 = nn.Conv2d(32*base_channels, 64, 5, 4, 2)
        self.do = nn.Dropout(0.5)

        self.fc1 = nn.Linear(4*4*64, 256)
        self.Actor = nn.Linear(256, num_outputs)
        self.Critic = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, train=False):
        x = self.do(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        self.state = self.relu(self.fc1(x))

        value = self.Critic(self.state)
        logits = self.Actor(self.state)
        dist = Categorical(logits=logits)
        return dist, value


class WideTopNet(nn.Module):
    def __init__(self, in_channels, num_outputs, base_channels=16):
        super(WideTopNet, self).__init__()
        print("WideTopNet")
        self.conv1 = nn.Conv2d(in_channels, 8*base_channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(8*base_channels, 64, 3, 2, 1)
        self.do = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16*16*64, 256)
        self.res_1 = self.create_layers(state_size=256, num_layers=8, scale_down=8, layer_type=FcResBlock)

        self.Actor = nn.Linear(256, num_outputs)
        self.Critic = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def create_layers(self, **kwargs):

        layers = []
        for i in range(kwargs["num_layers"]):
            layers.append(kwargs["layer_type"](i == 0, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        x = self.do(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        self.state = self.relu(self.res_1(x))

        value = self.Critic(self.state)
        logits = self.Actor(self.state)
        dist = Categorical(logits=logits)
        return dist, value

class ImpalaCnn64Value(nn.Module):
    def __init__(self, in_channels, base_channels=16):
        super(ImpalaCnn64Value, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, base_channels)
        self.conv_block2 = ConvBlock(base_channels, 2*base_channels)
        self.conv_block3 = ConvBlock(2*base_channels, 2*base_channels)

        self.fc1 = nn.Linear(8*8*2*base_channels, 256)
        self.Critic = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, train=False):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.reshape(x.size(0), -1)
        self.state = self.relu(self.fc1(x))
        value = self.Critic(self.state)
        return value

class DeepActor(nn.Module):
    def __init__(self, in_channels, num_outputs, base_channels=16):
        super(DeepActor, self).__init__()
        print("DeepActor")
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(base_channels, 64, 3, 2, 1)
        self.do = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16*16*64, 256)
        self.res_1 = self.create_layers(state_size=256, num_layers=8, scale_down=8, layer_type=FcBottleNeckBlock)

        self.Actor = nn.Linear(256, num_outputs)
        self.relu = nn.ReLU()

    def create_layers(self, **kwargs):

        layers = []
        for i in range(kwargs["num_layers"]):
            layers.append(kwargs["layer_type"](i == 0, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        x = self.do(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        self.state = self.relu(self.res_1(x))

        logits = self.Actor(self.state)
        dist = Categorical(logits=logits)
        return dist

class Split_Net(nn.Module):
    def __init__(self, in_channels, num_outputs, base_channels=16):
        super(Split_Net, self).__init__()
        self.actor = DeepActor(in_channels, num_outputs, base_channels)
        self.critic = ImpalaCnn64Value(in_channels, base_channels)

    def forward(self, x, train=False):
        dist = self.actor(x)
        value = self.critic(x)
        return dist, value


