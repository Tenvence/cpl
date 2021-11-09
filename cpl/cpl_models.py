import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as func


class CplModel(nn.Module):
    def __init__(self, feature_extractor, proxies_learner, metric_method):
        super(CplModel, self).__init__()

        self.feature_extractor = feature_extractor
        self.metric_method = metric_method
        self.proxies_learner = proxies_learner

    @amp.autocast()
    def forward(self, x):
        feature = self.feature_extractor(x)
        proxies = self.proxies_learner()

        assign_metric = self.metric_method(feature, proxies)
        assign_distribution = func.softmax(assign_metric, dim=-1)

        proxies_metric = self.metric_method(proxies, proxies)

        return assign_distribution, proxies_metric
