import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model architecture

class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units, max_num_sample, max_num_join, max_num_predicate):
        super(SetConv, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)
        self.max_num_sample = max_num_sample
        self.max_num_predicate = max_num_predicate
        self.max_num_join = max_num_join
        self.sample_feats = sample_feats
        self.predicate_feats = predicate_feats
        self.join_feats = join_feats

    def forward(self, inputs):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]


        samples_aggr = inputs.index_select(1, torch.arange(self.max_num_sample)).index_select(-1, torch.arange(self.sample_feats+1))
        samples, sample_mask = torch.split(samples_aggr, self.sample_feats, dim=-1)
        predicates_aggr = inputs.index_select(1, torch.arange(self.max_num_sample, self.max_num_sample+self.max_num_predicate)).index_select(-1, torch.arange(
            self.predicate_feats + 1))
        predicates, predicate_mask = torch.split(predicates_aggr, self.predicate_feats, dim=-1)
        joins_aggr = inputs.index_select(1, torch.arange(self.max_num_sample+self.max_num_predicate, self.max_num_sample+self.max_num_predicate+self.max_num_join)).index_select(-1, torch.arange(
            self.join_feats + 1))
        joins, join_mask = torch.split(joins_aggr, self.join_feats, dim=-1)


        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        hid_join = hid_join / join_norm

        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out
