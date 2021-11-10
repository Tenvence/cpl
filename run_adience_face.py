import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import time
import torch.distributed as dist
import torch.nn.parallel as parallel

import datasets.adience_face as af_dataset
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
    parser.add_argument('--root', default='../../DataSet/AdienceFace', type=str)
    parser.add_argument('--scheduler', default=4, type=int)

    parser.add_argument('--constraint', default='S-P', type=str, help='{S-P, S-B, H-L, H-S}')
    parser.add_argument('--feature_extractor', default='vgg16', type=str, help='{vgg16, resnet50, resnet101}')
    parser.add_argument('--metric_method', default='euclidean', type=str, help='{euclidean, cosine}')
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--cosine_scale', default=7., type=float)
    parser.add_argument('--tau', default=0.1, type=float)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--base_epochs', default=12, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--anchor_lr_mul', default=10, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--warm_up_epoch', default=1, type=int)
    parser.add_argument('--warm_up_ratio', default=0.333, type=float)
    parser.add_argument('--base_milestones', default=[9, 11], type=list)
    parser.add_argument('--step_gamma', default=0.1, type=float)

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--master_rank', default=0, type=int)

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_fold(fold_idx, args):
    num_ranks = 8
    train_dataset, val_dataset, test_dataset = af_dataset.get_datasets(args.root, fold_idx)

    train_dist_sampler = data.distributed.DistributedSampler(train_dataset)
    val_dist_sampler = data.distributed.DistributedSampler(val_dataset)
    test_dist_sampler = data.distributed.DistributedSampler(test_dataset)

    train_data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=4, sampler=train_dist_sampler, pin_memory=True, drop_last=True)
    val_data_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=4, sampler=val_dist_sampler, pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, args.batch_size, num_workers=4, sampler=test_dist_sampler, pin_memory=True)

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
    model = parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)

    model_name = f'model_adience_face_{args.constraint}_{args.feature_extractor}_{args.metric_method}_{args.feature_dim}_{args.cosine_scale}_{args.tau}.pkl'

    optim_parameters = [
        {'params': [p for n, p in model.module.named_parameters() if n.startswith('feature_extractor') and p.requires_grad]},
        {'params': [p for n, p in model.module.named_parameters() if not n.startswith('feature_extractor') and p.requires_grad], 'lr': args.lr * args.anchor_lr_mul}
    ]
    optimizer = optim.SGD(optim_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    milestones = [args.scheduler * ms for ms in args.base_milestones]
    lr_lambda = ll.get_warm_up_multi_step_lr_lambda(len(train_data_loader), args.warm_up_epoch, args.warm_up_ratio, milestones, args.step_gamma)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_mae = 100
    epochs = args.scheduler * args.base_epochs
    for epoch_idx in range(epochs):
        train_dist_sampler.set_epoch(epoch_idx)

        epoch_loss, t_train = engine.train(model, criterion, optimizer, lr_scheduler, train_data_loader)
        dist.reduce(epoch_loss, dst=args.master_rank, op=dist.ReduceOp.SUM)

        _, val_mae, val_t = engine.val(model, val_data_loader)
        dist.reduce(val_mae, dst=args.master_rank, op=dist.ReduceOp.SUM)

        if dist.get_rank() == args.master_rank:
            epoch_loss = epoch_loss.item() / dist.get_world_size()
            val_mae = val_mae.item() / dist.get_world_size()
            print_str = f'Fold:{fold_idx};Epoch:{epoch_idx + 1}/{epochs};Train:[Loss={epoch_loss:.4f};Time={t_train:.2f}s];Val:[MAE={val_mae:.3f};Time={val_t:.2f}s];'
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.module.state_dict(), model_name)
                print_str += ':-)'
            print(print_str)

    time.sleep(1)
    model.module.load_state_dict(torch.load(model_name))
    test_acc, test_mae, test_t = engine.val(model, test_data_loader)
    print(f'device: {dist.get_rank()}; ACC: {test_acc:.3f}, MAE: {test_mae:.3f}; Time: {test_t:.2f}.')

    time.sleep(1)
    dist.reduce(test_acc, dst=args.master_rank, op=dist.ReduceOp.SUM)
    time.sleep(1)
    dist.reduce(test_mae, dst=args.master_rank, op=dist.ReduceOp.SUM)

    if dist.get_rank() == args.master_rank:
        test_acc = test_acc.item() / dist.get_world_size()
        test_mae = test_mae.item() / dist.get_world_size()
        print(f'Test: [ACC: {test_acc:.3f}; MAE: {test_mae:.3f}].\n')

    return best_val_mae, test_acc, test_mae


def main():
    args = get_args_parser()
    dist.init_process_group(backend='nccl')
    set_random_seed(args.random_seed + dist.get_rank())
    torch.cuda.set_device(torch.device(f'cuda:{dist.get_rank()}'))

    val_mae_list, test_acc_list, test_mae_list = [], [], []
    for fold_idx in range(5):
        val_mae, test_acc, test_mae = run_fold(fold_idx, args)
        val_mae_list.append(val_mae)
        test_acc_list.append(test_acc)
        test_mae_list.append(test_mae)

    if dist.get_rank() == args.master_rank:
        val_mae_mean = np.array(val_mae_list).mean()

        test_acc_mean = np.array(test_acc_list).mean()
        test_acc_std = np.array(test_acc_list).std()

        test_mae_mean = np.array(test_mae_list).mean()
        test_mae_std = np.array(test_mae_list).std()

        print(f'Final Val MAE: {val_mae_mean:.3f}; Final Test: [ACC: {test_acc_mean:.3f} ± {test_acc_std:.3f}; MAE: {test_mae_mean:.3f} ± {test_mae_std:.3f}].')


if __name__ == '__main__':
    main()
