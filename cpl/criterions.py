import torch.nn as nn
import torch
import scipy.special
import torch.nn.functional as func


class Criterion(nn.Module):
    def forward(self, assign_distribution, gt, proxies_metric):
        pass


class SoftCplPoissonLoss(Criterion):
    def __init__(self, num_ranks, tau):
        super(SoftCplPoissonLoss, self).__init__()
        self.num_ranks = num_ranks
        self.tau = tau

    def forward(self, assign_distribution, gt, proxies_metric):
        rank_ids = torch.arange(self.num_ranks)[None, :].float()  # [1, C]
        factorial = torch.tensor(scipy.special.factorial(rank_ids))  # [1, C]
        rank_ids = rank_ids.to(gt.device)
        factorial = factorial.to(gt.device)
        lam = gt[:, None] + 0.5  # [B, 1]
        ordinal_smoothing_func = rank_ids * torch.log(lam) - lam - torch.log(factorial)
        target_distribution = func.softmax(ordinal_smoothing_func / self.tau, dim=-1)

        loss = func.kl_div(assign_distribution.log(), target_distribution, reduction='batchmean')
        return loss


class SoftCplBinomialLoss(Criterion):
    def __init__(self, num_ranks, tau):
        super(SoftCplBinomialLoss, self).__init__()
        self.num_ranks = num_ranks
        self.tau = tau

    def forward(self, assign_distribution, gt, proxies_metric):
        rank_ids = torch.arange(self.num_ranks)[None, :].float()  # [1, C]
        binom = scipy.special.binom(self.num_ranks - 1, rank_ids)  # [1, C]
        rank_ids = rank_ids.to(gt.device)
        binom = binom.to(gt.device)
        p = (2 * gt[:, None] + 1) / (2 * self.num_ranks)
        ordinal_smoothing_func = binom.log() + rank_ids * p.log() + (self.num_ranks - 1 - rank_ids) * (1 - p).log()
        target_distribution = func.softmax(ordinal_smoothing_func / self.tau, dim=-1)

        loss = func.kl_div(assign_distribution.log(), target_distribution, reduction='batchmean')
        return loss


class HardCplLoss(Criterion):
    def forward(self, assign_distribution, gt, proxies_metric):
        selected_proxies_metric = proxies_metric[gt, :].detach()  # [B, C]
        target_distribution = func.softmax(selected_proxies_metric, dim=-1)
        loss = func.kl_div(assign_distribution.log(), target_distribution, reduction='batchmean')
        return loss
