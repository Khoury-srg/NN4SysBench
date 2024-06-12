import torch.nn
import torch.nn as nn

from graph_convolution import GraphLayer, GraphLayer_marabou
from torch.nn.parameter import Parameter

Max_Node = 20


class model_benchmark(nn.Module):
    def __init__(self):
        super().__init__()
        # gcn
        self.output_dim = 8
        self.max_depth = 8

        self.act_fn = torch.nn.LeakyReLU()

        # initialize message passing transformation parameters
        # h: x -> x'
        self.h_gc1 = GraphLayer(5, 16)
        self.h_gc2 = GraphLayer(16, 8)
        self.h_gc3 = GraphLayer(8, 8)

        # f: x' -> e
        self.f_gc1 = GraphLayer(8, 16)
        self.f_gc2 = GraphLayer(16, 8)
        self.f_gc3 = GraphLayer(8, 8)

        # g: e -> e
        self.g_gc1 = GraphLayer(8, 16)
        self.g_gc2 = GraphLayer(16, 8)
        self.g_gc3 = GraphLayer(8, 8)

        # gsn

        # initialize summarization parameters for each hierarchy
        self.dag_gc1 = GraphLayer(5 + 8, 16)
        self.dag_gc2 = GraphLayer(16, 8)
        self.dag_gc3 = GraphLayer(8, 8)

        self.global_gc1 = GraphLayer(8, 16)
        self.global_gc2 = GraphLayer(16, 8)
        self.global_gc3 = GraphLayer(8, 8)

        # actor network
        self.node_input_dim = 5
        self.job_input_dim = 3
        self.output_dim = 8
        self.executor_levels = range(1, 16)

        self.fc1 = nn.Linear(29, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, input):
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400], dim=1)
        node_inputs = node_inputs.view([-1, Max_Node, 5])
        node_valid_mask = input[:, 100:120].view([-1, 1, Max_Node])
        gcn_mats = input[:, 120:3320].view([-1, 8, Max_Node, Max_Node])
        gcn_masks = input[:, 3320:3480].view([-1, 8, Max_Node, 1])
        summ_mats = input[:, 3480:3880].view([-1, Max_Node, Max_Node])
        running_dags_mat = input[:, 3880:3900].view([-1, 1, Max_Node])
        dag_summ_backward_map = input[:, 3900:4300].view([-1, Max_Node, Max_Node])


        # gcn
        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
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
        y = torch.matmul(gcn_mats[:, 0], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        y = y * gcn_masks[:, 0]

        # assemble neighboring information
        x = x + y

        # -------------------------2------------------------
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
        y = torch.matmul(gcn_mats[:, 1], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 1]

        # assemble neighboring information
        x = x + y

        # -------------------------3------------------------
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
        y = torch.matmul(gcn_mats[:, 2], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 2]

        # assemble neighboring information
        x = x + y

        # -------------------------4------------------------
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
        y = torch.matmul(gcn_mats[:, 3], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 3]

        # assemble neighboring information
        x = x + y

        # -------------------------5------------------------
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
        y = torch.matmul(gcn_mats[:, 4], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 4]

        # assemble neighboring information
        x = x + y

        # -------------------------6------------------------
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
        y = torch.matmul(gcn_mats[:, 5], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 5]

        # assemble neighboring information
        x = x + y

        # -------------------------7------------------------
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
        y = torch.matmul(gcn_mats[:, 6], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 6]

        # assemble neighboring information
        x = x + y

        # -------------------------8------------------------
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
        y = torch.matmul(gcn_mats[:, 7], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 7]

        # assemble neighboring information
        x = x + y

        gcn_output = x

        # gsn
        x = torch.concat([node_inputs, gcn_output], dim=2)

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(summ_mats, s)

        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)

        gsn_global_summary = torch.matmul(running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(dag_summ_backward_map, gsn_dag_summary)

        gsn_global_summ_extend_node = torch.concat([
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary], dim=1)
        merge_node = torch.concat([
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node], dim=2)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        node_outputs = node_outputs.view([-1, 1, Max_Node])

        # valid mask on node
        node_valid_mask = (node_valid_mask - 1) * 10000.0

        # apply mask
        node_outputs = node_outputs + node_valid_mask

        return torch.flatten(node_outputs, start_dim=1)


class model_benchmark_marabou(nn.Module):
    def __init__(self, input):
        super().__init__()
        # gcn
        # initialize message passing transformation parameters
        # h: x -> x'
        self.h_gc1 = GraphLayer_marabou(5, 16)
        self.h_gc2 = GraphLayer_marabou(16, 8)
        self.h_gc3 = GraphLayer_marabou(8, 8)

        # f: x' -> e
        self.f_gc1 = GraphLayer_marabou(8, 16)
        self.f_gc2 = GraphLayer_marabou(16, 8)
        self.f_gc3 = GraphLayer_marabou(8, 8)

        # g: e -> e
        self.g_gc1 = GraphLayer_marabou(8, 16)
        self.g_gc2 = GraphLayer_marabou(16, 8)
        self.g_gc3 = GraphLayer_marabou(8, 8)

        # gsn
        self.act_fn = torch.nn.ReLU()

        # initialize summarization parameters for each hierarchy
        self.dag_gc1 = GraphLayer_marabou(5 + 8, 16)
        self.dag_gc2 = GraphLayer_marabou(16, 8)
        self.dag_gc3 = GraphLayer_marabou(8, 8)

        self.global_gc1 = GraphLayer_marabou(8, 16)
        self.global_gc2 = GraphLayer_marabou(16, 8)
        self.global_gc3 = GraphLayer_marabou(8, 8)

        # actor network

        self.fc1 = nn.Linear(29, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        '''
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400], dim=1)
        self.gcn_mats = gcn_mats.reshape([8, Max_Node, Max_Node])
        self.gcn_masks = gcn_masks.reshape([8, Max_Node, 1])
        gcn_mats0, gcn_mats1, gcn_mats2, gcn_mats3, gcn_mats4, gcn_mats5, gcn_mats6, gcn_mats7 = torch.split(gcn_mats,
                                                                                                             [1, 1, 1,
                                                                                                              1, 1, 1,
                                                                                                              1, 1],
                                                                                                             dim=0)
        gcn_masks0, gcn_masks1, gcn_masks2, gcn_masks3, gcn_masks4, gcn_masks5, gcn_masks6, gcn_masks7 = torch.split(
            gcn_masks,
            [1, 1, 1,
             1, 1, 1,
             1, 1],
            dim=0)
        self.summ_mats = summ_mats.reshape([-1, Max_Node, Max_Node])
        self.running_dags_mat = running_dags_mat.reshape([-1, 1, Max_Node])
        self.dag_summ_backward_map = dag_summ_backward_map.reshape([-1, Max_Node, Max_Node])
        '''
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400], dim=0)
        self.gcn_mats = Parameter(gcn_mats.reshape([8, Max_Node, Max_Node]))
        self.gcn_masks = Parameter(gcn_masks.reshape([8, Max_Node, 1]))
        self.summ_mats = Parameter(summ_mats.reshape([Max_Node, Max_Node]))
        self.running_dags_mat = Parameter(running_dags_mat.reshape([1, Max_Node]))
        self.dag_summ_backward_map = Parameter(dag_summ_backward_map.reshape([Max_Node, Max_Node]))

    def forward(self, input):
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400], dim=0)
        node_inputs = node_inputs.reshape([Max_Node, 5])

        # gcn
        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
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
        y = torch.matmul(self.gcn_mats[0], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[0]

        # assemble neighboring information
        y = x + y

        # -------------------------2------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[1], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[1]

        # assemble neighboring information
        y = x + y

        # -------------------------3------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[2], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[2]

        # assemble neighboring information
        y = x + y

        # -------------------------4------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[3], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[3]

        # assemble neighboring information
        y = x + y

        # -------------------------5------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[4], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[4]

        # assemble neighboring information
        y = x + y

        # -------------------------6------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[5], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        # y = y * self.gcn_masks[5]

        # assemble neighboring information
        y = x + y

        # -------------------------7------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[6], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        # y = y * self.gcn_masks[5]

        # assemble neighboring information
        y = x + y

        # -------------------------8------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[7], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[7]

        # assemble neighboring information
        y = x + y
        gcn_output = y

        # gsn

        x = torch.cat((node_inputs, gcn_output), dim=1)

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(self.summ_mats, s)

        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)

        gsn_global_summary = torch.matmul(self.running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(self.dag_summ_backward_map, gsn_dag_summary)

        gsn_global_summ_extend_node = torch.cat((
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary), dim=0)

        # node_inputs = node_inputs.view([1, 20, 5])
        # gcn_output = gcn_output.view([1, 20, 8])

        merge_node = torch.cat((
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node), dim=1)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        # valid mask on node
        node_valid_mask = node_valid_mask * 10000.0
        node_outputs = node_outputs.reshape(1, 20)

        # apply mask
        node_outputs = node_outputs + node_valid_mask

        return node_outputs
