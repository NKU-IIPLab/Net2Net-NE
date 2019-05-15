import time

import numpy as np
import torch
from torch import nn
from torch.nn import init

from basic.classify import read_node_label
from basic.graph import MyGraph
from basic.util import read_word_code, node_classification, fetch
from model.models import MeanAggregator, EgoEncoder, ContentCNN


class Net2Net(nn.Module):
    def __init__(self, global_graph, features, encoder):
        super(Net2Net, self).__init__()
        self.graph = global_graph
        self.node_num = self.graph.node_num
        self.embed_dim = encoder.embed_dim
        self.features = features
        self.encoder = encoder
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, self.node_num))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.encoder(nodes)
        scores = embeds.mm(self.weight)
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

    def evaluate(self, b_list, lab, ratio):
        self.eval()
        hidden = []
        idx = []
        for bat in b_list:
            h = self.encoder(bat)
            hidden.extend(h.detach().cpu().numpy())
            idx.extend(bat)

        f1 = []
        for r in ratio:
            f1.append(node_classification(hidden, np.arange(len(lab)), [lab[i] for i in idx], r))
        return f1


def main():
    data_dir = 'data/Citeseer/'
    adj_file = 'edges.txt'
    label_file = 'labels.txt'
    con_file = 'title.txt'
    voca_file = 'voc.txt'
    word_num = 5523
    max_doc_len = 34

    # data_dir = 'data/DBLP/'
    # adj_file = 'edges.txt'
    # label_file = 'labels.txt'
    # con_file = 'title.txt'
    # voca_file = 'voc.txt'
    # word_num = 8501
    # max_doc_len = 27

    # data_dir = 'data/Cora'
    # adj_file = 'edges.txt'
    # label_file = 'labels.txt'
    # con_file = 'abstract.txt'
    # voca_file = 'voc.txt'
    # word_num = 12619
    # max_doc_len = 100

    word_emb_dim = 500
    conv_dim = 500
    kernel_num = 200
    kernel_sizes = [1, 2, 3, 4, 5]
    conv_drop = 0.2
    enc_dim = 500
    batch_size = 32
    epoch_num = 100
    l_rate = 1e-4
    class_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]

    gpu_id = 2
    gpu = torch.device('cuda', gpu_id)

    start = time.time()
    graph = MyGraph(data_dir + adj_file)

    _, labels = read_node_label(data_dir + label_file)

    node_content, pad_code = read_word_code(data_dir + con_file, data_dir + voca_file)

    features = ContentCNN(word_num, word_emb_dim, conv_dim, kernel_num, kernel_sizes, conv_drop, gpu)

    agg1 = MeanAggregator(lambda nodes: features(fetch(node_content, nodes, max_doc_len, pad_code)), gpu)
    enc1 = EgoEncoder(lambda nodes: features(fetch(node_content, nodes, max_doc_len, pad_code)), conv_dim, enc_dim,
                      graph, agg1)

    agg2 = MeanAggregator(lambda nodes: enc1(nodes), gpu)
    enc2 = EgoEncoder(lambda nodes: enc1(nodes), enc1.embed_dim, enc_dim, graph, agg2, base_model=enc1)

    c2n = Net2Net(graph, features, enc2)
    c2n.cuda(gpu)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, c2n.parameters()), lr=l_rate)

    for e in range(epoch_num):
        avg_loss = []
        c2n.train()
        batch_list = graph.get_batches(batch_size)
        for batch in batch_list:
            optimizer.zero_grad()
            loss = c2n.loss(batch, torch.tensor(batch, dtype=torch.int64, device=gpu))
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        # node classification results
        f1_micro = c2n.evaluate(batch_list, labels, class_ratio)
        minute = np.around((time.time() - start) / 60)
        ls = np.mean(avg_loss)
        print('Epoch:', e, 'loss:', ls, 'mi-F1:', np.around(f1_micro, 3), 'time:', minute, 'mins.')
        avg_loss.clear()

if __name__ == "__main__":
    main()
