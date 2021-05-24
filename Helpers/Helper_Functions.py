import numpy as np
from collections import deque
import torch


class ReplayBuffer:
    def __init__(self, data_names, buffer_size, mini_batch_size, output_device):
        self.data_keys = data_names
        self.data_dict = {}
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.output_device = output_device
        self.reset()

    def reset(self):
        # Create a deque for each data type with set max length
        for name in self.data_keys:
            self.data_dict[name] = deque(maxlen=self.buffer_size)

    def buffer_full(self):
        return len(self.data_dict[self.data_keys[0]]) == self.buffer_size

    def data_log(self, data_name, data):
        # split tensor along batch into a list of individual datapoints
        data = data.cpu().split(1)
        # Extend the deque for data type, deque will handle popping old data to maintain buffer size
        self.data_dict[data_name].extend(data)

    def __iter__(self):
        batch_size = len(self.data_dict[self.data_keys[0]])
        batch_size = batch_size - batch_size % self.mini_batch_size

        ids = np.random.permutation(batch_size)
        ids = np.split(ids, batch_size // self.mini_batch_size)
        for i in range(len(ids)):
            batch_dict = {}
            for name in self.data_keys:
                c = [self.data_dict[name][j] for j in ids[i]]
                batch_dict[name] = torch.cat(c).to(self.output_device)
            batch_dict["batch_size"] = len(ids[i])
            yield batch_dict

    def __len__(self):
        return len(self.data_dict[self.data_keys[0]])


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values.append(next_value)
    gae = 0
    returns = deque()

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])
    values.pop()
    return returns


def lr_linear(frame_max, frame_no, lr, optimizer):
    lr_adj = ((frame_max - frame_no) / frame_max) * lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_adj
    return optimizer


def state_to_tensor(state_dict, device):
    if type(state_dict) is dict:
        state = torch.FloatTensor(state_dict['rgb'].transpose((0, 3, 1, 2))).to(device)
    else:
        state = torch.FloatTensor(state_dict.transpose((2, 0, 1))).unsqueeze(0).to(device)

    return state / 256  # Put tensor on gpu before divide - much faster



