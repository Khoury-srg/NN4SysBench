import math

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
S_INFO = 6
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
cuda = torch.cuda.is_available()
RAND_RANGE = 1000
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = torch.FloatTensor([60.0 * MILLISECONDS_IN_SECOND])  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes


class ActorNetwork_mid(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.conv1 = nn.Conv1d(1, 128, 4)
        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(1, 128)

        self.linear3 = nn.Linear(2048, 128)
        self.linear4 = nn.Linear(128, self.a_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.to(torch.float32)
        x = x.view([-1, self.s_dim[0], self.s_dim[1]])
        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 5:6, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        out = self.softmax(x)

        return out


class ActorNetwork_small(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(8, 128)
        self.linear3 = nn.Linear(8, 128)
        self.linear4 = nn.Linear(6, 128)
        self.linear5 = nn.Linear(1, 128)

        self.linear6 = nn.Linear(768, 128)
        self.linear7 = nn.Linear(128, self.a_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.to(torch.float32)

        x = x.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 5:6, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)
        out = self.softmax(x)

        return out


class ActorNetwork_big(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.conv1 = nn.Conv1d(1, 128, 4)
        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(1, 128)

        self.linear3 = nn.Linear(2048, 256)
        self.linear4 = nn.Linear(256, self.a_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.to(torch.float32)

        x = x.view([-1, self.s_dim[0], self.s_dim[1]])
        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 5:6, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        out = self.softmax(x)

        return out


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        self.conv1 = nn.Conv1d(1, 128, 4)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(96256, 128)
        self.linear3 = nn.Linear(128, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view([-1, self.s_dim[0], self.s_dim[1]])

        x = x.to(torch.float32)

        split_0 = self.linear1(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)
        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear1(x[:, 5:6, -1])

        split_0 = torch.flatten(split_0)
        split_1 = torch.flatten(split_1)
        split_2 = torch.flatten(split_2)
        split_3 = torch.flatten(split_3)
        split_4 = torch.flatten(split_4)
        split_5 = torch.flatten(split_5)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5)).view(1, -1)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def compute_loss(s_batch, a_batch, r_batch, terminal, actor, critic, criterion=nn.MSELoss()):
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    # v_batch = critic.forward(s_batch)
    critic_input = torch.from_numpy(s_batch)
    if cuda:
        critic_input = critic_input.cuda()
    v_batch = critic.forward(critic_input)

    if cuda:
        v_batch = v_batch.cpu()
    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    R_batch = torch.from_numpy(R_batch)
    td_batch = R_batch - v_batch

    a_batch = torch.from_numpy(a_batch)

    input = torch.from_numpy(s_batch)
    if cuda:
        input = input.cuda()
    probability = actor.forward(input)
    if cuda:
        probability = probability.cpu()
    m_probs = Categorical(probability)
    log_probs = m_probs.log_prob(a_batch.squeeze())
    log_probs = log_probs.view((-1, 1))
    actor_loss = torch.sum(log_probs * (-td_batch))
    entropy_loss = -ENTROPY_WEIGHT * torch.sum(m_probs.entropy())
    actor_loss = actor_loss + entropy_loss

    input = torch.from_numpy(s_batch)
    if cuda:
        input = input.cuda()
    critic_output = critic.forward(input)
    if cuda:
        critic_output = critic_output.cpu()
    critic_loss = criterion(critic_output, R_batch.to(torch.float32))

    return actor_loss, critic_loss, td_batch
