import argparse
import random

import numpy as np
import torch.optim as optim
import torch.utils.data
import tqdm

import datasets.historical_color as hc_dataset
import engine
import lr_lambda as ll
from cpl import CplModel


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=970423, type=int)
    parser.add_argument('--root', default='../../DataSet/HistoricalColor', type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--scheduler', default=4, type=int)

    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--cosine_scale', default=5., type=float)
    parser.add_argument('--poisson_tau', default=0.01, type=float)
    parser.add_argument('--binomial_tau', default=0.01, type=float)
    parser.add_argument('--constraint', default='S-P', type=str, help='{S-P, S-B, H-L, H-S}')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--base_epochs', default=12, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--anchor_lr_mul', default=10., type=float)

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
    set_random_seed(args.random_seed + fold_idx)
    train_data_loader, val_data_loader, test_data_loader = hc_dataset.get_data_loaders(args.root, args.batch_size)

    model = CplModel(5, args.feature_dim, args.cosine_scale, args.poisson_tau, args.binomial_tau, args.constraint).cuda()

    optim_parameters = [
        {'params': [p for n, p in model.named_parameters() if n.startswith('feature_extractor') and p.requires_grad]},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('feature_extractor') and p.requires_grad], 'lr': args.lr * args.anchor_lr_mul}
    ]
    optimizer = optim.SGD(optim_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    milestones = [args.scheduler * ms for ms in args.base_milestones]
    lr_lambda = ll.get_warm_up_multi_step_lr_lambda(len(train_data_loader), args.warm_up_epoch, args.warm_up_ratio, milestones, args.step_gamma)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    min_val_score = 1000
    test_acc, test_mae = 0, 0
    epochs = args.scheduler * args.base_epochs
    # processor = tqdm.tqdm(range(epochs))
    for epoch_idx in range(epochs):
        epoch_loss = engine.train(model, optimizer, lr_scheduler, train_data_loader)
        val_acc, val_mae = engine.val(model, val_data_loader)

        print(f'Fold: {fold_idx + 1}; Epoch: {epoch_idx + 1:>2} / {epochs}; Train: [Loss: {epoch_loss:.4f}]; Val: [ACC: {val_acc:.3f}; MAE: {val_mae:.3f}]')
        if val_mae <= min_val_score:
            min_val_score = val_mae
            test_acc, test_mae = engine.val(model, test_data_loader)
            print(f'Got best val results! New results on test set are [ACC: {test_acc:.3f}; MAE: {test_mae:.3f}]\n')
        else:
            print(f'Not get best val results! Current results on test set are [ACC: {test_acc:.3f}; MAE: {test_mae:.3f}]\n')
    print(f'Fold: {fold_idx + 1}; Test: [ACC: {test_acc:.3f}; MAE: {test_mae:.3f}].\n')
    return test_acc.item(), test_mae.item()


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
