import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from basic.classify import Classifier


# Read node features from file
def read_node_fea(feature_path):
    fea = []
    fin = open(feature_path, 'r')
    for l in fin.readlines():
        vec = l.split()
        fea.append(np.array([float(x) for x in vec[1:]]))
    fin.close()
    return np.array(fea, dtype='float32')


def read_word_code(text_path, voca_path):
    words = []
    fin = open(voca_path, 'r')
    for l in fin.readlines():
        words.append(l.strip())
    fin.close()
    word_map = {words[i]: i for i in range(len(words))}
    pad_code = word_map['<eos>']

    content_code = []
    fin = open(text_path, 'r')
    for l in fin.readlines():
        info = l.strip().split(' ')
        doc_code = [word_map[w] for w in info]
        # if len(doc_code) > max_len:
        #     doc_code = doc_code[0: max_len]
        # else:
        #     doc_code.extend([pad_code for _ in range(max_len - len(doc_code))])
        content_code.append(doc_code)
    return content_code, pad_code
    # return np.array(content_code, dtype='int')


def fetch(content_code, ids, max_len, pad_code):
    code = []
    for id in ids:
        doc_code = content_code[id]
        if len(doc_code) > max_len:
            doc_code = doc_code[0: max_len]
        else:
            doc_code.extend([pad_code for _ in range(max_len - len(doc_code))])
        code.append(doc_code)

    return code


def node_classification(hidden, idx, label, ratio):
    lr = Classifier(vectors=hidden, clf=LogisticRegression())
    f1_mi = lr.split_train_evaluate(idx, label, ratio)
    return f1_mi


def exclusive_combine(*in_list):
    res = set()
    in_list = list(*in_list)
    for n_l in in_list:
        for i in n_l:
            res.add(i)
    return list(res)


def identity_map(n_list):
    id_dict = {}
    for i in range(len(n_list)):
        id_dict[n_list[i]] = i
    return id_dict


def agg_mean(M, id_dict, keys):
    idList = []
    for id in keys:
        idList.append(id_dict[id])

    return torch.mean(M[idList, :], 0, True)


def agg_max(M, id_dict, keys):
    idList = []
    for id in keys:
        idList.append(id_dict[id])
    res, _ = torch.max(M[idList, :], 0, True)
    return res