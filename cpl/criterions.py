import torch.nn as nn
import torch
import scipy.special
import torch.nn.functional as func


class SoftCplPoissonLoss(nn.Module):
    def __init__(self, num_ranks, tau, loss_lam):
        super(SoftCplPoissonLoss, self).__init__()
        self.num_ranks = num_ranks
        self.tau = tau
        self.loss_lam = loss_lam

    def forward(self, assign_metric, gt, proxies_metric):
        loss = func.cross_entropy(assign_metric, gt)

        rank_ids = torch.arange(self.num_ranks)[None, :].float()  # [1, C]
        factorial = torch.tensor(scipy.special.factorial(rank_ids))  # [1, C]
        rank_ids = rank_ids.to(gt.device)
        factorial = factorial.to(gt.device)

        lam = torch.arange(self.num_ranks)[:, None].float() + 0.5  # [B, 1]
        lam = lam.to(gt.device)
        ordinal_smoothing_func = rank_ids * torch.log(lam) - lam - torch.log(factorial)
        target_distribution = func.softmax(ordinal_smoothing_func / self.tau, dim=-1)

        loss += self.loss_lam * func.kl_div(func.softmax(proxies_metric, dim=-1).log(), target_distribution, reduction='batchmean')
        return loss


class SoftCplBinomialLoss(nn.Module):
    def __init__(self, num_ranks, tau, loss_lam):
        super(SoftCplBinomialLoss, self).__init__()
        self.num_ranks = num_ranks
        self.tau = tau
        self.loss_lam = loss_lam

    def forward(self, assign_metric, gt, proxies_metric):
        loss = func.cross_entropy(assign_metric, gt)

        rank_ids = torch.arange(self.num_ranks)[None, :].float()  # [1, C]
        binom = scipy.special.binom(self.num_ranks - 1, rank_ids)  # [1, C]
        p = (2 * torch.arange(self.num_ranks)[:, None].float() + 1) / (2 * self.num_ranks)

        rank_ids = rank_ids.to(gt.device)
        binom = binom.to(gt.device)
        p = p.to(gt.device)
        ordinal_smoothing_func = binom.log() + rank_ids * p.log() + (self.num_ranks - 1 - rank_ids) * (1 - p).log()
        target_distribution = func.softmax(ordinal_smoothing_func / self.tau, dim=-1)

        loss += self.loss_lam * func.kl_div(func.softmax(proxies_metric, dim=-1).log(), target_distribution, reduction='batchmean')
        return loss


class HardCplLoss(nn.Module):
    @staticmethod
    def forward(assign_metric, gt, proxies_metric):
        selected_proxies_metric = proxies_metric[gt, :].detach()  # [B, C]
        loss = func.kl_div(func.log_softmax(assign_metric, dim=-1), func.softmax(selected_proxies_metric, dim=-1), reduction='batchmean')
        return loss
