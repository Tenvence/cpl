import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as cv_models


class OrdinalModel(nn.Module):
    def __init__(self, num_clusters):
        super(OrdinalModel, self).__init__()
        self.num_clusters = num_clusters

        base_backbone = cv_models.vgg16_bn(pretrained=True)
        feature_dim = 512

        self.feature_extractor = nn.Sequential(*list(base_backbone.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        self.cluster_ids = nn.Parameter(torch.arange(self.num_clusters)[:, None].float(), requires_grad=False)

        # Linear-Layout Constrained Proxies Learner
        self.margin_vec = nn.Parameter(torch.empty((1, feature_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.margin_vec)

        # semi-circle centroids
        self.base_vec = nn.Parameter(torch.empty((1, feature_dim)), requires_grad=True)
        self.indicate_vec = nn.Parameter(torch.empty((1, feature_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.base_vec)
        nn.init.xavier_normal_(self.indicate_vec)

    def _get_linear_centroids(self):
        centroids = self.cluster_ids * self.margin_vec
        return centroids

    def _get_semi_circle_centroids(self):
        base_angle = np.pi / (self.num_clusters - 1)
        theta = self.cluster_ids * base_angle
        gamma = torch.cosine_similarity(self.base_vec, self.indicate_vec).arccos()
        norm_base_vec = self.base_vec / torch.linalg.norm(self.base_vec, dim=-1)
        norm_indicate_vec = self.indicate_vec / torch.linalg.norm(self.indicate_vec, dim=-1)
        centroids = (gamma - theta).sin() / gamma.sin() * norm_base_vec + theta.sin() / gamma.sin() * norm_indicate_vec
        return centroids

    @staticmethod
    def _get_kl_div(pred, target):
        loss = target * (target.log() - pred.log())
        loss = loss.sum(dim=-1).mean()
        return loss

    def forward(self, x, gt=None):
        feature = self.feature_extractor(x)

        # # linear centroids
        # centroids = self._get_linear_centroids()
        # feature_centroid_dist = torch.cdist(feature, centroids)
        # assign_logist = -(1. + feature_centroid_dist ** 2).log()
        # assign_distribution = func.softmax(assign_logist, dim=-1)

        scale = 10

        # semi-circle centroids
        centroids = self._get_semi_circle_centroids()
        feature_centroid_dist = torch.cosine_similarity(feature[:, None, :], centroids[None, :, :], dim=-1)  # [B, C]
        assign_logist = scale * feature_centroid_dist
        assign_distribution = func.softmax(assign_logist, dim=-1)

        if not self.training:
            return assign_distribution
        else:
            # # linear centroids
            # centroids_dist = torch.cdist(centroids, centroids).detach()  # [C, C]
            # selected_centroids_dist = centroids_dist[gt, :]  # [B, C]
            # ordinal_logist = -(1. + selected_centroids_dist ** 2).log()
            # ordinal_distribution = func.softmax(ordinal_logist, dim=-1)
            # ordinal_loss = self._get_kl_div(assign_distribution, ordinal_distribution)
            # return ordinal_loss

            # semi-circle centroids
            centroids_dist = torch.cosine_similarity(centroids[:, None, :], centroids[None, :, :], dim=-1).detach()  # [C, C]
            selected_centroids_dist = centroids_dist[gt, :]  # [B, C]
            ordinal_logist = scale * selected_centroids_dist
            ordinal_distribution = func.softmax(ordinal_logist, dim=-1)
            ordinal_loss = self._get_kl_div(assign_distribution, ordinal_distribution)
            return ordinal_loss
