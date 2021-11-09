import torch.nn as nn
import torchvision.models as cv_models


class Vgg16(nn.Module):
    def __init__(self, feature_dim):
        super(Vgg16, self).__init__()
        self.feature_extractor = nn.Sequential(
            *list(cv_models.vgg16_bn(pretrained=True).children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1), nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        return self.feature_extractor(x)


class ResNet18(nn.Module):
    def __init__(self, feature_dim):
        super(ResNet18, self).__init__()
        self.feature_extractor = nn.Sequential(*list(cv_models.resnet18(pretrained=True).children())[:-1], nn.Flatten(1), nn.Linear(512, feature_dim))

    def forward(self, x):
        return self.feature_extractor(x)


class ResNet101(nn.Module):
    def __init__(self, feature_dim):
        super(ResNet101, self).__init__()
        self.feature_extractor = nn.Sequential(*list(cv_models.resnet101(pretrained=True).children())[:-1], nn.Flatten(1), nn.Linear(2048, feature_dim))

    def forward(self, x):
        return self.feature_extractor(x)
