import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from basic.util import exclusive_combine


class MeanAggregator(nn.Module):

    def __init__(self, features, cur_device, gcn=False):

        super(MeanAggregator, self).__init__()
        self.features = features
        self.device = cur_device
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs):
        samp_neighs = [samp_neigh + [nodes[i]] for i, samp_neigh in enumerate(to_neighs)]

        unique_nodes_list = exclusive_combine(samp_neighs)
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        # The mask for aggregation
        mask = torch.zeros(len(samp_neighs), len(unique_nodes), requires_grad=False, device=self.device)
        # The connections
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # Normalize
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        embed_matrix = self.features(unique_nodes_list)
        to_feats = mask.mm(embed_matrix)
        return to_feats  # node_num * fea_dim


class EgoEncoder(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, graph, aggregator, base_model=None):
        super(EgoEncoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.graph = graph
        self.aggregator = aggregator
        if base_model is not None:
            self.base_model = base_model

        self.weight = nn.Parameter(torch.FloatTensor(self.feat_dim, embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        to_neighs = [self.graph.neighbor_dict[node] for node in nodes]
        neigh_feats = self.aggregator.forward(nodes, to_neighs)
        combined = neigh_feats
        combined.mm(self.weight)
        combined = torch.tanh(combined)
        return combined  # node_num * emb_dim


class ContentCNN(nn.Module):
    def __init__(self, word_num, word_emb_dim, conv_dim, kernel_num, kernel_sizes, dropout, cur_device):
        super(ContentCNN, self).__init__()
        self.word_embeddings = nn.Embedding(word_num, word_emb_dim)
        # self.word_embeddings.weight = nn.Parameter(torch.FloatTensor(word_num, word_emb_dim))
        # self.word_embeddings.cuda(cur_device)

        # CNN with different kernel sizes
        self.conv_list = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, word_emb_dim)) for K in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(len(kernel_sizes) * kernel_num, conv_dim)
        self.weight = nn.Parameter(torch.FloatTensor(len(kernel_sizes) * kernel_num, conv_dim))
        self.device = cur_device

        init.xavier_uniform(self.word_embeddings.weight)
        init.xavier_uniform(self.weight)

    def conv_and_pool(self, x, conv):
        x_conv = conv(x)
        x_act = F.relu(x_conv).squeeze(3)  # (N, Co, W)
        x_pool = F.max_pool1d(x_act, x_act.size(2)).squeeze(2)
        return x_pool

    def forward(self, node_batch):
        query = torch.LongTensor(node_batch).cuda(self.device)
        x = self.word_embeddings(query)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_list]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        # logit = self.fc(x)  # (N, C)
        logit = x.mm(self.weight)
        logit = torch.tanh(logit)
        return logit