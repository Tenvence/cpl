import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as cv_models

import cpl.proxies_learner as pl


class CplModel(nn.Module):
    def __init__(self, num_ranks, dim, cosine_scale, poisson_tau, constraint):
        super(CplModel, self).__init__()
        self.num_ranks = num_ranks
        self.cosine_scale = cosine_scale
        self.poisson_tau = poisson_tau
        self.constraint = constraint

        self.feature_extractor = nn.Sequential(
            *list(cv_models.vgg16_bn(pretrained=True).children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1), nn.Linear(512, dim, bias=False)
        )

        self.proxies_learner = None
        if constraint in {'U-P', 'U-B'}:
            self.proxies_learner = pl.BaseProxiesLearner(num_ranks, dim)
        if constraint == 'L-L':
            self.proxies_learner = pl.LinearLayoutConstrainedProxiesLearner(num_ranks, dim)
        if constraint == 'L-S':
            self.proxies_learner = pl.SemicircularLayoutConstrainedProxiesLearner(num_ranks, dim)

    @staticmethod
    def _get_kl_div(pred, target):
        # loss = target * (target.log() - pred.log())
        loss = -target * pred.log()
        loss = loss.sum(dim=-1).mean()
        return loss

    def forward(self, x, gt=None):
        feature = self.feature_extractor(x)
        proxies = self.proxies_learner()

        assign_metric = None
        if self.constraint in {'U-P', 'U-B', 'L-L'}:
            assign_metric = -torch.cdist(feature, proxies).log()
        if self.constraint == 'L-S':
            assign_metric = self.cosine_scale * torch.cosine_similarity(feature[:, None, :], proxies[None, :, :], dim=-1)  # [B, C]
        assign_distribution = func.softmax(assign_metric, dim=-1)

        if not self.training:
            return assign_distribution
        else:
            target_distribution = None
            if self.constraint == 'U-P':
                rank_ids = torch.arange(self.num_ranks)[None, :].float()  # [1, C]
                factorial = torch.tensor(scipy.special.factorial(rank_ids))[:, None]  # [1, C]
                rank_ids = rank_ids.to(gt.device)
                factorial = factorial.to(gt.device)
                lam = gt[:, None] + 0.5  # [B, 1]
                tef = rank_ids * torch.log(lam) - lam - torch.log(factorial)
                target_distribution = func.softmax(tef / self.poisson_tau, dim=-1)
            if self.constraint == 'U-B':
                pass
            if self.constraint == 'L-L':
                proxies_metric = -torch.cdist(proxies, proxies).log().detach()  # [C, C]
                selected_proxies_metric = proxies_metric[gt, :]  # [B, C]
                target_distribution = func.softmax(selected_proxies_metric, dim=-1)
            if self.constraint == 'L-S':
                proxies_metric = self.cosine_scale * torch.cosine_similarity(proxies[:, None, :], proxies[None, :, :], dim=-1).detach()  # [C, C]
                selected_proxies_metric = proxies_metric[gt, :]  # [B, C]
                target_distribution = func.softmax(selected_proxies_metric, dim=-1)

            loss = self._get_kl_div(assign_distribution, target_distribution)
            return loss
