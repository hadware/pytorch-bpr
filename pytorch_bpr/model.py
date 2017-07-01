from typing import List, Tuple

import numpy
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def to_variable(input):
    return Variable(torch.from_numpy(numpy.array(input)))


class MFModel(nn.Module):

    def __init__(self, nb_users : int, nb_items : int, embedding_dim : int):
        super().__init__()
        self.users_mat = nn.Embedding(nb_users, embedding_dim)
        self.items_mat = nn.Embedding(nb_items, embedding_dim)


class Scorer:
    pass


class DotProductScorer(Scorer):

    def __init__(self, users_mat : nn.Embedding, items_mat : nn.Embedding):
        self.users_mat = users_mat
        self.items_mat = items_mat
        self._rank = items_mat.embedding_dim
        self.bias_mat = nn.Embedding(self.items_mat.num_embeddings, 1) # num embeddings is the items count

    def __call__(self, batch_u: Variable, batch_i: Variable):
        users_features = self.users_mat(batch_u)
        items_features = self.items_mat(batch_i)
        return (users_features * items_features).sum(1) + self.bias_mat(batch_i)


class BPRLossFunctional:

    def __init__(self, scorer : DotProductScorer):
        self.scorer = scorer

    def step(self, triplets_list : List[Tuple[int,int,int]]):
        batch_u, batch_i, batch_j = (to_variable(batch) for batch in zip(*triplets_list))
        xui = self.scorer(batch_u, batch_i)
        xuj = self.scorer(batch_u, batch_j)
        return -F.logsigmoid(xui - xuj).sum()


class BPR:

    def __init__(self, scorer : Scorer, model : torch.Module):
        self.scorer = scorer
        self.model = model