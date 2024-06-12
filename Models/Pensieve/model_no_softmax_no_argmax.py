import math

import torch
import torch.nn as nn
import numpy as np

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


class ActorNetwork_mid_marabou(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.conv1 = nn.Conv1d(1, 128, 4, padding=0)
        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(1, 128)

        self.linear3 = nn.Linear(2048, 128)
        self.linear4 = nn.Linear(128, self.a_dim)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([self.s_dim[0], self.s_dim[1]])
        split_0, split_1, split_2, split_3, split_4, split_5 = torch.split(x, [1, 1, 1, 1, 1, 1], dim=0)
        a, b, c, d, e, f, g, split_0 = torch.split(split_0, [1, 1, 1, 1, 1, 1, 1, 1], dim=1)
        split_0 = split_0.view(split_0.shape[0], -1)

        split_0 = self.linear0(split_0)
        split_0 = self.relu(split_0)

        a, b, c, d, e, f, g, split_1 = torch.split(split_1, [1, 1, 1, 1, 1, 1, 1, 1], dim=1)

        split_1 = self.linear1(split_1)
        split_1 = self.relu(split_1)

        split_2 = self.conv1(split_2)

        split_2 = self.relu(split_2)
        split_3 = self.conv1(split_3)
        split_3 = self.relu(split_3)

        split_4, a = torch.split(split_4, [A_DIM, 2], dim=1)
        split_4 = self.conv1(split_4)
        split_4 = self.relu(split_4)
        a, split_5 = split_5.split(split_4, [7, 1], dim=1)
        split_5 = self.linear2(split_5)

        split_2 = split_2.view(1, -1)
        split_3 = split_3.view(1, -1)
        split_4 = split_4.view(1, -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        return x


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

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
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

        return x





class ActorNetwork_mid_parallel(nn.Module):
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
        self.video_sizes = torch.tensor([10, 20, 40, 80, 160, 320]).to(torch.float32).reshape(-1, 1)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([-1, 2 * self.s_dim[0], self.s_dim[1]])
        x1, x2 = torch.split(x, 6, dim=1)

        split_0 = self.linear0(x1[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x1[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x1[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x1[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x1[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x1[:, 5:6, -1])

        split_2 = split_2.flatten(1)
        split_3 = split_3.flatten(1)
        split_4 = split_4.flatten(1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        x = self.relu(x)
        sq = torch.square(x)
        deno = torch.sum(sq, 1, keepdim=True)
        distribution = sq / deno
        bit_rate1 = torch.matmul(distribution, self.video_sizes)

        split_0 = self.linear0(x2[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x2[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x2[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x2[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x2[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x2[:, 5:6, -1])

        split_2 = split_2.flatten(1)
        split_3 = split_3.flatten(1)
        split_4 = split_4.flatten(1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        sq = torch.square(x)
        deno = torch.sum(sq, 1, keepdim=True)
        distribution = sq / deno
        bit_rate2 = torch.matmul(distribution, self.video_sizes)
        return bit_rate1 - bit_rate2


class ActorNetwork_small_marabou(nn.Module):
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

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))

        x = x.view([self.s_dim[0], self.s_dim[1]])
        split_0, split_1, split_2, split_3, split_4, split_5 = torch.split(x, [1, 1, 1, 1, 1, 1], dim=0)
        a, b, c, d, e, f, g, split_0 = torch.split(split_0, [1, 1, 1, 1, 1, 1, 1, 1], dim=1)
        split_0 = split_0.view(split_0.shape[0], -1)

        split_0 = self.linear0(split_0)
        split_0 = self.relu(split_0)

        a, b, c, d, e, f, g, split_1 = torch.split(split_1, [1, 1, 1, 1, 1, 1, 1, 1], dim=1)
        split_1 = split_1.view(split_1.shape[0], -1)

        split_1 = self.linear1(split_1)
        split_1 = self.relu(split_1)

        split_2 = self.linear2(split_2)
        split_2 = self.relu(split_2)
        split_3 = self.linear3(split_3)
        split_3 = self.relu(split_3)
        split_4, a = torch.split(split_4, [A_DIM, 2], dim=1)
        a, split_5 = torch.split(split_5, [7, 1], dim=1)

        split_4 = self.linear4(split_4)
        split_4 = self.relu(split_4)
        split_5 = self.linear5(split_5)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        return x


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

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
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

        return x





class ActorNetwork_small_parallel(nn.Module):
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
        self.video_sizes = torch.tensor([10, 20, 40, 80, 160, 320]).to(torch.float32).reshape(-1, 1)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([-1, self.s_dim[0] * 2, self.s_dim[1]])
        x1, x2 = torch.split(x, 6, dim=1)
        split_0 = self.linear0(x1[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x1[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x1[:, 2, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x1[:, 3, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x1[:, 4, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x1[:, 5:6, -1])

        split_2 = split_2.flatten(1)
        split_3 = split_3.flatten(1)
        split_4 = split_4.flatten(1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        x = self.relu(x)
        sq = torch.pow(x, 3)
        deno = torch.sum(sq, 1, keepdim=True)
        distribution = sq / deno
        bit_rate1 = torch.matmul(distribution, self.video_sizes)

        split_0 = self.linear0(x2[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x2[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x2[:, 2, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x2[:, 3, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x2[:, 4, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x2[:, 5:6, -1])

        split_2 = split_2.flatten(1)
        split_3 = split_3.flatten(1)
        split_4 = split_4.flatten(1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        x = self.relu(x)
        sq = torch.pow(x, 3)
        deno = torch.sum(sq, 1, keepdim=True)
        distribution = sq / deno
        bit_rate2 = torch.matmul(distribution, self.video_sizes)
        return bit_rate1 - bit_rate2

class ActorNetwork_big_marabou(nn.Module):
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

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([self.s_dim[0], self.s_dim[1]])
        split_0, split_1, split_2, split_3, split_4_5, a = torch.split(x, [1, 1, 1, 1, 1, 1], dim=0)
        a, b, c, d, e, f, g, split_0 = torch.split(split_0, [1, 1, 1, 1, 1, 1, 1, 1], dim=1)


        split_0 = self.linear0(split_0)
        split_0 = self.relu(split_0)

        a, b, c, d, e, f, g, split_1 = torch.split(split_1, [1, 1, 1, 1, 1, 1, 1, 1], dim=1)

        split_1 = self.linear1(split_1)
        split_1 = self.relu(split_1)


        split_2 = split_2.reshape(1, 1, -1)

        split_2 = self.conv1(split_2)

        split_2 = self.relu(split_2)
        split_3 = split_3.reshape(1, 1, -1)
        split_3 = self.conv1(split_3)
        split_3 = self.relu(split_3)

        split_4, a, split_5 = torch.split(split_4_5, [A_DIM, 1, 1], dim=1)
        split_4 = split_4.reshape(1, 1, -1)
        split_4 = self.conv1(split_4)
        split_4 = self.relu(split_4)
        split_5 = self.linear2(split_5)

        print(split_2.shape)

        split_2 = split_2.reshape(-1)
        split_3 = split_3.reshape(-1)
        split_4 = split_4.reshape(-1)
        split_0 = split_0.reshape(-1)
        split_1 = split_1.reshape(-1)
        split_5 = split_5.reshape(-1)




        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 0)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        return x
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

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
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

        return x


class ActorNetwork_big_parallel(nn.Module):
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
        self.video_sizes = torch.tensor([10, 20, 40, 80, 160, 320]).to(torch.float32).reshape(-1, 1)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))

        x = x.view([-1, 2 * self.s_dim[0], self.s_dim[1]])
        x1, x2 = torch.split(x, 6, dim=1)

        split_0 = self.linear0(x1[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x1[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x1[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x1[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x1[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x1[:, 5:6, -1])

        split_2 = split_2.flatten(1)
        split_3 = split_3.flatten(1)
        split_4 = split_4.flatten(1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        x = self.relu(x)
        sq = torch.pow(x, 3)
        deno = torch.sum(sq, 1, keepdim=True)
        distribution = sq / deno
        bit_rate1 = torch.matmul(distribution, self.video_sizes)

        split_0 = self.linear0(x2[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x2[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x2[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x2[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x2[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x2[:, 5:6, -1])

        split_2 = split_2.flatten(1)
        split_3 = split_3.flatten(1)
        split_4 = split_4.flatten(1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        x = self.relu(x)
        sq = torch.pow(x, 3)
        deno = torch.sum(sq, 1, keepdim=True)
        distribution = sq / deno
        bit_rate2 = torch.matmul(distribution, self.video_sizes)
        return bit_rate1 - bit_rate2
