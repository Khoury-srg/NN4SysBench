"""
Graph Summarization Network

Summarize node features globally
via parameterized aggregation scheme
"""

import copy
import numpy as np
from tf_op import glorot
import torch
import torch.nn as nn
from graph_convolution import GraphLayer


class GraphSNN(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim):
        # on each transformation, input_dim -> (multiple) hid_dims -> output_dim
        # the global level summarization will use output from DAG level summarizaiton
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dims = hid_dims

        self.act_fn = torch.nn.LeakyReLU()

        # initialize summarization parameters for each hierarchy
        self.dag_gc1 = GraphLayer(input_dim, hid_dims[0])
        self.dag_gc2 = GraphLayer(hid_dims[0], hid_dims[1])
        self.dag_gc3 = GraphLayer(hid_dims[1], output_dim)

        self.global_gc1 = GraphLayer(output_dim, hid_dims[0])
        self.global_gc2 = GraphLayer(hid_dims[0], hid_dims[1])
        self.global_gc3 = GraphLayer(hid_dims[1], output_dim)


    def summarize(self, summ_mats, running_dags_mat, inputs):
        # summarize information in each hierarchy
        # e.g., first level summarize each individual DAG
        # second level globally summarize all DAGs
        x = inputs

        summaries = []

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)



        s = torch.sparse.mm(summ_mats.to(torch.float32), s)
        summaries.append(s)

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)

        s = torch.sparse.mm(running_dags_mat.to(torch.float32), s)
        summaries.append(s)

        return summaries
