import torch
import torch.nn as nn


class CosineMetric(nn.Module):
    def __init__(self, scale):
        super(CosineMetric, self).__init__()
        self.scale = scale

    def forward(self, x1, x2):
        return self.scale * torch.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=-1)


class EuclideanMetric(nn.Module):
    @staticmethod
    def forward(x1, x2):
        return -torch.log(1 + torch.cdist(x1, x2) ** 2)
