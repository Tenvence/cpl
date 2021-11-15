import argparse
import random

import numpy as np
import torch.optim as optim
import torch.utils.data
import tqdm
import torch.utils.data as data

import datasets.historical_color as hc_dataset
import engine
import lr_lambda as ll
from cpl.cpl_model import CplModel
from cpl.criterions import SoftCplPoissonLoss, SoftCplBinomialLoss, HardCplLoss
from cpl.feature_extractors import Vgg16, ResNet18, ResNet101
from cpl.metric_methods import EuclideanMetric, CosineMetric
from cpl.proxies_learner import BaseProxiesLearner, LinearProxiesLearner, SemicircularProxiesLearner


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=970423, type=int)
    parser.add_argument('--root', default='../../DataSet/HistoricalColor', type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--scheduler', default=4, type=int)

    parser.add_argument('--constraint', default='S-P', type=str, help='{S-P, S-B, H-L, H-S}')
    parser.add_argument('--feature_extractor', default='vgg16', type=str, help='{vgg16, resnet50, resnet101}')
    parser.add_argument('--metric_method', default='euclidean', type=str, help='{euclidean, cosine}')
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--cosine_scale', default=7., type=float)
    parser.add_argument('--tau', default=0.1, type=float)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--base_epochs', default=12, type=int)
    parser.add_argument('--lr', default=0.2, type=float)
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
    num_ranks = 5

    set_random_seed(args.random_seed + fold_idx)
    train_dataset, val_dataset, test_dataset = hc_dataset.get_data_loaders(args.root)
    train_dataloader = data.DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_dataloader = data.DataLoader(val_dataset, args.batch_size, pin_memory=True)
    test_dataloader = data.DataLoader(test_dataset, args.batch_size, pin_memory=True)

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

    model = CplModel(feature_extractor, proxies_learner, metric_method).cuda()

    model_name = f'model_historical_color_{args.constraint}_{args.feature_extractor}_{args.metric_method}_{args.feature_dim}_{args.cosine_scale}_{args.tau}.pkl'

    optim_parameters = [
        {'params': [p for n, p in model.named_parameters() if n.startswith('feature_extractor') and p.requires_grad]},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('feature_extractor') and p.requires_grad], 'lr': args.lr * args.anchor_lr_mul}
    ]
    optimizer = optim.SGD(optim_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    milestones = [args.scheduler * ms for ms in args.base_milestones]
    lr_lambda = ll.get_warm_up_multi_step_lr_lambda(len(train_dataloader), args.warm_up_epoch, args.warm_up_ratio, milestones, args.step_gamma)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_mae = 1000
    test_acc, test_mae = 0, 0
    epochs = args.scheduler * args.base_epochs
    # processor = tqdm.tqdm(range(epochs))
    for epoch_idx in range(epochs):
        epoch_loss, t_train = engine.train(model, criterion, optimizer, lr_scheduler, train_dataloader)
        _, val_mae, t_val = engine.val(model, val_dataloader)
        val_mae = val_mae.item()

        print_str = f'Fold:{fold_idx};Epoch:{epoch_idx + 1}/{epochs};Train:[Loss={epoch_loss:.4f};Time={t_train:.2f}s];Val:[MAE={val_mae:.3f};Time={t_val:.2f}s];'
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.module.state_dict(), model_name)
            print_str += ':-)'
        print(print_str)
    test_acc, test_mae, test_t = engine.val(model, test_dataloader)
    test_acc = test_acc.item()
    test_mae = test_mae.item()
    print(f'ACC: {test_acc:.3f}, MAE: {test_mae:.3f}; Time: {test_t:.2f}.\n')
    return test_acc, test_mae


def main():
    args = get_args_parser()
    torch.cuda.set_device(args.device_id)

    acc_list, mae_list = [], []
    for fold_idx in range(10):
        acc, mae = run_fold(fold_idx, args)
        acc_list.append(acc)
        mae_list.append(mae)

    acc_mean = np.array(acc_list).mean()
    acc_std = np.array(acc_list).std()

    mae_mean = np.array(mae_list).mean()
    mae_std = np.array(mae_list).std()

    print(f'ACC: {acc_mean:.3f} ± {acc_std:.3f}')
    print(f'MAE: {mae_mean:.3f} ± {mae_std:.3f}')


if __name__ == '__main__':
    main()
