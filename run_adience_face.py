import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import datasets.adience_face as af_dataset
import engine
import lr_lambda as ll
from cpl.cpl_models import CplModel
from cpl.criterions import SoftCplPoissonLoss, SoftCplBinomialLoss, HardCplLoss
from cpl.feature_extractors import Vgg16, ResNet18, ResNet101
from cpl.metric_methods import EuclideanMetric, CosineMetric
from cpl.proxies_learner import BaseProxiesLearner, LinearProxiesLearner, SemicircularProxiesLearner


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=970423, type=int)
    parser.add_argument('--root', default='../../DataSet/AdienceFace', type=str)
    parser.add_argument('--scheduler', default=4, type=int)

    parser.add_argument('--constraint', default='S-P', type=str, help='{S-P, S-B, H-L, H-S}')
    parser.add_argument('--feature_extractor', default='vgg16', type=str, help='{vgg16, resnet50, resnet101}')
    parser.add_argument('--metric_method', default='euclidean', type=str, help='{euclidean, cosine}')
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--cosine_scale', default=7., type=float)
    parser.add_argument('--tau', default=0.1, type=float)

    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--base_epochs', default=12, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--anchor_lr_mul', default=10, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--warm_up_epoch', default=1, type=int)
    parser.add_argument('--warm_up_ratio', default=0.333, type=float)
    parser.add_argument('--base_milestones', default=[9, 11], type=list)
    parser.add_argument('--step_gamma', default=0.1, type=float)

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_fold(fold_idx, args):
    num_ranks = 8
    train_data_loader, val_data_loader, test_data_loader = af_dataset.get_data_loader(args.root, fold_idx, args.batch_size)

    feature_extractor = None
    if args.feature_extractor == 'vgg16':
        feature_extractor = Vgg16(args.feature_dim)
    if args.feature_extractor == 'resnet18':
        feature_extractor = ResNet18(args.feature_dim)
    if args.feature_extractor == 'resnet101':
        feature_extractor = ResNet101(args.feature_dim)

    metric_method = None
    if args.metric_method == 'euclidean':
        metric_method = EuclideanMetric()
    if args.metric_method == 'cosine':
        metric_method = CosineMetric(args.cosine_scale)

    proxies_learner = criterion = None
    if args.constraint == 'S-P':
        proxies_learner = BaseProxiesLearner(num_ranks, args.feature_dim)
        criterion = SoftCplPoissonLoss(num_ranks, args.tau)
    if args.constraint == 'S-B':
        proxies_learner = BaseProxiesLearner(num_ranks, args.feature_dim)
        criterion = SoftCplBinomialLoss(num_ranks, args.tau)
    if args.constraint == 'H-L':
        metric_method = EuclideanMetric()
        proxies_learner = LinearProxiesLearner(num_ranks, args.feature_dim)
        criterion = HardCplLoss()
    if args.constraint == 'H-S':
        metric_method = CosineMetric(args.cosine_scale)
        proxies_learner = SemicircularProxiesLearner(num_ranks, args.feature_dim)
        criterion = HardCplLoss()

    model = CplModel(feature_extractor, proxies_learner, metric_method)
    model = nn.DataParallel(model).cuda()

    optim_parameters = [
        {'params': [p for n, p in model.module.named_parameters() if n.startswith('feature_extractor') and p.requires_grad]},
        {'params': [p for n, p in model.module.named_parameters() if not n.startswith('feature_extractor') and p.requires_grad], 'lr': args.lr * args.anchor_lr_mul}
    ]
    optimizer = optim.SGD(optim_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    milestones = [args.scheduler * ms for ms in args.base_milestones]
    lr_lambda = ll.get_warm_up_multi_step_lr_lambda(len(train_data_loader), args.warm_up_epoch, args.warm_up_ratio, milestones, args.step_gamma)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc, best_val_mae = 100, 100
    test_acc, test_mae = 0, 0
    epochs = args.scheduler * args.base_epochs
    for epoch_idx in range(epochs):
        print(f'Fold: {fold_idx}; Epoch: {epoch_idx + 1:>2} / {epochs};')

        epoch_loss, t_train = engine.train(model, criterion, optimizer, lr_scheduler, train_data_loader)
        val_acc, val_mae, t_val = engine.val(model, val_data_loader)
        print(f'    Train: [Loss={epoch_loss:.4f}; Time={t_train:.2f}s]; Val: [ACC={val_acc:.3f}; MAE={val_mae:.3f}; Time={t_val:.2f}s];')

        if val_mae < best_val_mae:
            best_val_acc = val_acc
            best_val_mae = val_mae
            test_acc, test_mae, test_t = engine.val(model, test_data_loader)
            print(f'    Get best model! Test: [ACC={test_acc:.3f}; MAE={test_mae:.3f}; Time={test_t:.2f}s].\n')
        else:
            print(f'    Do not get best model! Best Test: [ACC={test_acc:.3f}; MAE={test_mae:.3f}].\n')

    print(f'Fold: {fold_idx}; Best Val: [ACC={best_val_acc:.3f}; MAE={best_val_mae:.3f}]; Best Test: [ACC={test_acc:.3f}; MAE={test_mae:.3f}].\n')

    return best_val_acc, best_val_mae, test_acc, test_mae


def main():
    args = get_args_parser()
    set_random_seed(args.random_seed)

    val_acc_list, val_mae_list = [], []
    test_acc_list, test_mae_list = [], []
    for fold_idx in range(5):
        val_acc, val_mae, test_acc, test_mae = run_fold(fold_idx, args)
        val_acc_list.append(val_acc)
        val_mae_list.append(val_mae)
        test_acc_list.append(test_acc)
        test_mae_list.append(test_mae)

    val_acc_mean = np.array(val_acc_list).mean()
    val_acc_std = np.array(val_acc_list).std()

    val_mae_mean = np.array(val_mae_list).mean()
    val_mae_std = np.array(val_mae_list).std()

    test_acc_mean = np.array(test_acc_list).mean()
    test_acc_std = np.array(test_acc_list).std()

    test_mae_mean = np.array(test_mae_list).mean()
    test_mae_std = np.array(test_mae_list).std()

    print(f'Final Val : [ACC: {val_acc_mean:.3f} ± {val_acc_std:.3f}; MAE: {val_mae_mean:.3f} ± {val_mae_std:.3f}];')
    print(f'Final Test: [ACC: {test_acc_mean:.3f} ± {test_acc_std:.3f}; MAE: {test_mae_mean:.3f} ± {test_mae_std:.3f}].')


if __name__ == '__main__':
    main()
