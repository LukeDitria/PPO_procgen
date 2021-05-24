import torch.nn as nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
import torch
import torchvision.models as models

def rms(x):
#     return x
    return x/x.pow(2).mean(1, keepdim=True).sqrt()

class FcResBlock(nn.Module):
    def __init__(self, first_block, **kwargs):
        super(FcResBlock, self).__init__()
        self.fc1 = nn.Linear(kwargs["state_size"], int(kwargs["state_size"] / kwargs["scale_down"]))
        self.fc2 = nn.Linear(int(kwargs["state_size"] / kwargs["scale_down"]), kwargs["state_size"])
        self.first_block = first_block
        self.relu = nn.ReLU()

    def forward(self, x_in):
        if not self.first_block:
            x_in = self.relu(x_in)
        x = self.relu(self.fc1(x_in))
        x = self.fc2(x) + x_in
        return x

class FcResBlock2(nn.Module):
    def __init__(self, first_block, **kwargs):
        super(FcResBlock2, self).__init__()
        self.fc1 = nn.Linear(kwargs["state_size"], int(kwargs["state_size"] / kwargs["scale_down"]))
        self.fc2 = nn.Linear(int(kwargs["state_size"] / kwargs["scale_down"]), kwargs["state_size"])
        self.fc3 = nn.Linear(kwargs["state_size"], kwargs["state_size"])

        self.first_block = first_block
        self.relu = nn.ReLU()

    def forward(self, x_in):
        if not self.first_block:
            x_in = self.relu(x_in)
        x = self.relu(self.fc1(x_in))
        x = self.relu(self.fc2(x) + x_in)
        return self.fc3(x)

class FcResActionBlock(nn.Module):
    def __init__(self, first_block, **kwargs):
        super(FcResActionBlock, self).__init__()
        self.fc1 = nn.Linear(kwargs["state_size"] + kwargs["action_size"],
                             int(kwargs["state_size"] / kwargs["scale_down"]))

        self.fc2 = nn.Linear(int(kwargs["state_size"] / kwargs["scale_down"]),
                             kwargs["state_size"])
        self.first_block = first_block
        self.relu = nn.ReLU()

    def forward(self, x_in):
        state, action = x_in
        x_cat = torch.cat((state, action), 1)

        if not self.first_block:
            x_cat = self.relu(x_cat)
        x = self.relu(self.fc1(x_cat))
        x = self.fc2(x) + state
        return (x, action)

class FcResActionBlock2(nn.Module):
    def __init__(self, first_block, **kwargs):
        super(FcResActionBlock2, self).__init__()
        self.fc1 = nn.Linear(kwargs["state_size"] + kwargs["action_size"],
                             int(kwargs["state_size"] / kwargs["scale_down"]))

        self.fc2 = nn.Linear(int(kwargs["state_size"] / kwargs["scale_down"]),
                             kwargs["state_size"])

        self.fc3 = nn.Linear(kwargs["state_size"], kwargs["state_size"])

        self.first_block = first_block
        self.relu = nn.ReLU()

    def forward(self, x_in):
        state, action = x_in
        x_cat = torch.cat((state, action), 1)

        if not self.first_block:
            x_cat = self.relu(x_cat)
        x = self.relu(self.fc1(x_cat))
        x = self.relu(self.fc2(x) + state)
        return (self.fc3(x), action)

class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2, acti_out=True):
        super(Res_down, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale,scale)

        self.acti_out = acti_out

    def forward(self, x):
        skip = self.AvePool(self.conv3(x))

        x = F.relu(self.conv1(x))
        x = self.AvePool(x)
        x = self.conv2(x)

        if self.acti_out:
            return F.relu(x + skip)
        else:
            x = x + skip
        return x

class Res_down_IN(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2, acti_out=True):
        super(Res_down_IN, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(channel_out//2, affine=True)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        self.in2 = nn.InstanceNorm2d(channel_out, affine=True)

        self.AvePool = nn.AvgPool2d(scale, scale)

        self.acti_out = acti_out

    def forward(self, x):
        skip = self.AvePool(self.conv3(x))

        x = F.relu(self.in1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.conv2(x)

        if self.acti_out:
            return F.relu(self.in2(x + skip))
        else:
            x = self.in2(x + skip)
        return x


class ResEncoder(nn.Module):
    def __init__(self, channels, ch=16, hidden1=512):
        super(ResEncoder, self).__init__()
        self.conv1 = Res_down(channels, ch)
        self.conv2 = Res_down(ch, ch*2)
        self.conv3 = Res_down(ch*2, ch*4)
        self.conv4 = Res_down(ch*4, ch*8)
        self.fc1 = nn.Linear(ch*8*4*4, hidden1)
        self.do = nn.Dropout(0.5)

    def forward(self, x, train=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.size(0), -1)
        if train:
            x = self.do(x)
        return self.fc1(x)


class ResEncoderIN(nn.Module):
    def __init__(self, channels, ch=16, hidden1=512):
        super(ResEncoderIN, self).__init__()
        self.conv1 = Res_down_IN(channels, ch)
        self.conv2 = Res_down_IN(ch, ch*2)
        self.conv3 = Res_down_IN(ch*2, ch*4)
        self.conv4 = Res_down_IN(ch*4, ch*8)
        self.fc1 = nn.Linear(ch*8*4*4, hidden1)
        self.do = nn.Dropout(0.5)

    def forward(self, x, train=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.size(0), -1)
        if train:
            x = self.do(x)
        return self.fc1(x)

class ResEncoderAttention(nn.Module):
    def __init__(self, channels, ch=16, hidden1=512):
        super(ResEncoderAttention, self).__init__()
        self.conv1 = Res_down_IN(channels, ch)
        self.conv2 = Res_down_IN(ch, ch*2)
        self.conv3 = Res_down_IN(ch*2, ch*4)

        self.conv_key = Res_down(ch*4, ch*4, acti_out=False)
        self.query = nn.Linear(ch*4*8*8, ch*4)
        self.conv_value = Res_down(ch*4, ch*8)

        self.fc1 = nn.Linear(ch*8*4*4, hidden1)
        self.do = nn.Dropout(0.5)

    def get_attention(self, x):
        key = self.conv_key(x)
        query = self.query(x.reshape(x.shape[0], -1)).unsqueeze(2)
        key_query = (key.reshape(x.shape[0], x.shape[1], -1) * query).sum(1)#BSxH*W
        return F.softmax(key_query, 1).reshape(x.shape[0], 1, key.shape[2], key.shape[3])#BSx1xHxW

    def forward(self, x, train=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        value = self.conv_value(x)
        self.attention_map = self.get_attention(x)
        value_out = value*self.attention_map

        x_1 = value_out.reshape(value_out.size(0), -1)
        if train:
            x_1 = self.do(x_1)
        return self.fc1(x_1)

class BasicEncoder(nn.Module):
    def __init__(self, channels, ch, hidden1):
        super(BasicEncoder, self).__init__()
        self.conv1 = nn.Conv2d(channels, ch, 3, 2, 1)
        self.conv2 = nn.Conv2d(ch, ch*2, 3, 2, 1)
        self.conv3 = nn.Conv2d(ch*2, ch*4, 3, 2, 1)
        self.conv4 = nn.Conv2d(ch*4, ch*8, 3, 2, 1)
        self.fc1 = nn.Linear(ch*8*4*4, hidden1)
        # self.do = nn.Dropout(0.5)

    def forward(self, x, train=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        # if train:
        #     x = self.do(x)
        return F.relu(self.fc1(x))

class ResRewardEncoderRMS(nn.Module):
    def __init__(self, channels, ch=16, hidden1=512):
        super(ResRewardEncoderRMS, self).__init__()
        self.encoder = ResEncoder(channels, ch=ch, hidden1=hidden1)

        self.fc1_rew = nn.Linear(hidden1, hidden1//2)
        self.fc2_rew = nn.Linear(hidden1//2, 1)

    def reward_pred(self, x, train):
        state = self.forward(x, train)
        x1 = F.relu(self.fc1_rew(state))
        rew_out = self.fc2_rew(x1)
        return state, rew_out

    def forward(self, x, train):
        return rms(self.encoder(x, train))

class BasicRewardEncoderRMS(nn.Module):
    def __init__(self, channels, ch=16, hidden1=512):
        super(BasicRewardEncoderRMS, self).__init__()
        self.encoder = ResEncoder(channels, ch=ch, hidden1=hidden1)

        self.fc1_rew = nn.Linear(hidden1, hidden1//2)
        self.fc2_rew = nn.Linear(hidden1//2, 1)

    def reward_pred(self, x, train):
        state = self.forward(x, train)
        x1 = F.relu(self.fc1_rew(state))
        rew_out = self.fc2_rew(x1)
        return state, rew_out

    def forward(self, x, train):
        return rms(self.encoder(x, train))

class StateEncoderDeltaPredictorRMS(nn.Module):
    """
    Takes in a 64x64 image and predicts a delta state rep
    """
    def __init__(self, channels, ch=16, input_size=512, pred_depth=1, num_actions=15):
        self.num_actions = num_actions
        super(StateEncoderDeltaPredictorRMS, self).__init__()
        self.res_encoder = ResEncoder(channels, ch=ch, hidden1=input_size)
        self.res_predictor = self.create_layers(state_size=input_size, num_layers=pred_depth, action_size=num_actions,
                                                scale_down=8, layer_type=FcResActionBlock2)

    def create_layers(self, **kwargs):
        layers = []
        for i in range(kwargs["num_layers"]):
            layers.append(kwargs["layer_type"](i == 0, **kwargs))
        return nn.Sequential(*layers)

    def state_pred(self, data_in, action, state_0, train):
        action_vector = F.one_hot(action, self.num_actions).type(torch.cuda.FloatTensor)
        state_1 = self.forward(data_in, train)
        state_delta, _ = self.res_predictor((state_1, action_vector))
        return state_1, state_delta + state_0

    def forward(self, x, train):
        return rms(self.res_encoder(x, train))
    
class StateEncoderMultiLayer(nn.Module):
    """"
    Each block takes in the original input data and predicts the previous blocks next state
    """
    def __init__(self, channels, ch, hidden1, pred_depth):
        super(StateEncoderMultiLayer, self).__init__()
        self.encoder_rew = ResRewardEncoderRMS(channels, ch, hidden1)
        self.predictor_encoder = StateEncoderDeltaPredictorRMS(channels, ch, hidden1, pred_depth)

    def get_states(self, obs, train=False):
        return self.encoder_rew(obs, train)

    def train_step(self, obs, action, train=False):
        state_0, rew_out = self.encoder_rew.reward_pred(obs, train)
        state_n, next_state_pred = self.predictor_encoder.state_pred(obs, action, state_0.detach(), train)
        return state_0, state_n, rew_out, next_state_pred

    def reward_pred(self, obs, train=False):
        state_0, rew_out = self.encoder_rew.reward_pred(obs, train)
        state_n = self.predictor_encoder(obs, train)
        return state_0, state_n, rew_out

    def forward(self, obs, train=False):
        state_0 = self.encoder_rew(obs, train)
        state_n = self.predictor_encoder(obs, train)
        return state_0, state_n

class PPO_agent_res_split(nn.Module):
    def __init__(self, input_size, num_outputs, num_layers=1, stack_size=0):
        super(PPO_agent_res_split, self).__init__()
        state_size = input_size*(stack_size + 1)

        self.res_a = self.create_layers(state_size=state_size, num_layers=1, scale_down=2, layer_type=FcResBlock)
        self.res_c = self.create_layers(state_size=state_size, num_layers=1, scale_down=2, layer_type=FcResBlock)

        self.fc_a = nn.Linear(state_size, 64)
        self.fc_c = nn.Linear(state_size, 64)

        self.Actor = nn.Linear(64, num_outputs)
        self.Critic = nn.Linear(64, 1)

    def create_layers(self, **kwargs):

        layers = []
        for i in range(kwargs["num_layers"]):
            layers.append(kwargs["layer_type"](i == 0, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, critic_in, actor_in):
        actor_in = actor_in.reshape(actor_in.shape[0], -1)
        critic_in = critic_in.reshape(critic_in.shape[0], -1)

        xa = F.relu(self.res_a(actor_in))
        xc = F.relu(self.res_c(critic_in))
        xa = F.relu(self.fc_a(xa))
        xc = F.relu(self.fc_c(xc))

        logits = self.Actor(xa)
        dist = Categorical(logits=logits)

        value = self.Critic(xc)
        return dist, value

class PPO_agent_res_split_counter(nn.Module):
    def __init__(self, input_size, num_outputs, num_layers=1, stack_size=0):
        super(PPO_agent_res_split_counter, self).__init__()
        state_size = input_size * (stack_size + 1)
        self.fc_a_1 = nn.Linear(state_size + 256, state_size)
        self.fc_c_1 = nn.Linear(state_size + 256, state_size)

        self.res_a = self.create_layers(state_size=state_size, num_layers=num_layers,
                                        scale_down=2, layer_type=FcResBlock)
        self.res_c = self.create_layers(state_size=state_size, num_layers=num_layers,
                                        scale_down=2, layer_type=FcResBlock)

        self.fc_a_2 = nn.Linear(state_size, 64)
        self.fc_c_2 = nn.Linear(state_size, 64)

        self.Actor = nn.Linear(64, num_outputs)
        self.Critic = nn.Linear(64, 1)

    def create_layers(self, **kwargs):
        layers = []
        for i in range(kwargs["num_layers"]):
            layers.append(kwargs["layer_type"](i == 0, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, critic_in, actor_in, step_counter):
        actor_in = actor_in.reshape(actor_in.shape[0], -1)
        critic_in = critic_in.reshape(critic_in.shape[0], -1)

        actor_in = torch.cat((actor_in, step_counter), 1)
        critic_in = torch.cat((critic_in, step_counter), 1)

        xa = F.relu(self.fc_a_1(actor_in))
        xc = F.relu(self.fc_c_1(critic_in))

        xa = F.relu(self.res_a(xa))
        xc = F.relu(self.res_c(xc))

        xa = F.relu(self.fc_a_2(xa))
        xc = F.relu(self.fc_c_2(xc))

        logits = self.Actor(xa)
        dist = Categorical(logits=logits)

        value = self.Critic(xc)
        return dist, value