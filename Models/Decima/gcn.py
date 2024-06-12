"""
Graph Convolutional Network

Propergate node features among neighbors
via parameterized message passing scheme
"""

import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from tf_op import glorot
from graph_convolution import GraphLayer


# According to paper, all use 2 layer hidden layers
class GraphCNN(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim, max_depth):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dims
        self.max_depth = max_depth

        self.act_fn = torch.nn.LeakyReLU()

        # initialize message passing transformation parameters
        # h: x -> x'
        self.h_gc1 = GraphLayer(input_dim, hid_dims[0])
        self.h_gc2 = GraphLayer(hid_dims[0], hid_dims[1])
        self.h_gc3 = GraphLayer(hid_dims[1], output_dim)

        # f: x' -> e
        self.f_gc1 = GraphLayer(output_dim, hid_dims[0])
        self.f_gc2 = GraphLayer(hid_dims[0], hid_dims[1])
        self.f_gc3 = GraphLayer(hid_dims[1], output_dim)

        # g: e -> e
        self.g_gc1 = GraphLayer(output_dim, hid_dims[0])
        self.g_gc2 = GraphLayer(hid_dims[0], hid_dims[1])
        self.g_gc3 = GraphLayer(hid_dims[1], output_dim)

    def forward(self, adj_mats, masks, node_inputs):
        # message passing among nodes
        # the information is flowing from leaves to roots

        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        for d in range(self.max_depth):
            # work flow: index_select -> f -> masked assemble via adj_mat -> g
            y = x

            # process the features on the nodes
            y = self.f_gc1(y)
            y = self.act_fn(y)
            y = self.f_gc2(y)
            y = self.act_fn(y)
            y = self.f_gc3(y)
            y = self.act_fn(y)

            # message passing
            y = torch.spmm(adj_mats[d], y)

            # aggregate child features
            y = self.f_gc1(y)
            y = self.act_fn(y)
            y = self.f_gc2(y)
            y = self.act_fn(y)
            y = self.f_gc3(y)
            y = self.act_fn(y)

            # remove the artifact from the bias term in g
            y = y * masks[d]

            # assemble neighboring information
            x = x + y

        return x
