import numpy as np
import torch
import torch.nn as nn


class BaseProxiesLearner(nn.Module):
    def __init__(self, num_ranks, dim):
        super(BaseProxiesLearner, self).__init__()

        self.proxies = nn.Parameter(torch.empty((num_ranks, dim)), requires_grad=True)
        nn.init.xavier_normal_(self.proxies)

    def forward(self):
        return self.proxies


class LinearProxiesLearner(nn.Module):
    def __init__(self, num_ranks, dim):
        super(LinearProxiesLearner, self).__init__()

        self.rank_ids = nn.Parameter(torch.arange(num_ranks)[:, None].float(), requires_grad=False)

        self.v0 = nn.Parameter(torch.empty((1, dim)), requires_grad=True)
        self.v1 = nn.Parameter(torch.empty((1, dim)), requires_grad=True)
        nn.init.xavier_normal_(self.v0)
        nn.init.xavier_normal_(self.v1)

    def forward(self):
        proxies = self.rank_ids * self.v0 + self.v1
        return proxies


class SemicircularProxiesLearner(nn.Module):
    def __init__(self, num_ranks, dim):
        super(SemicircularProxiesLearner, self).__init__()

        self.num_ranks = num_ranks
        self.rank_ids = nn.Parameter(torch.arange(num_ranks)[:, None].float(), requires_grad=False)

        self.v0 = nn.Parameter(torch.empty((1, dim)), requires_grad=True)
        self.v1 = nn.Parameter(torch.empty((1, dim)), requires_grad=True)
        nn.init.xavier_normal_(self.v0)
        nn.init.xavier_normal_(self.v1)

    def forward(self):
        theta = self.rank_ids * np.pi / (self.num_ranks - 1)
        gamma = torch.cosine_similarity(self.v0, self.v1).arccos()
        norm_v0 = self.v0 / torch.linalg.norm(self.v0, dim=-1)
        norm_v1 = self.v1 / torch.linalg.norm(self.v1, dim=-1)
        proxies = (gamma - theta).sin() / gamma.sin() * norm_v0 + theta.sin() / gamma.sin() * norm_v1
        return proxies
