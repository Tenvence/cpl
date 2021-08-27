import argparse
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data
import time

import datasets.adience_face as af_dataset
import engine
import lr_lambda as ll
from cpl import CplModel


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=970423, type=int)
    parser.add_argument('--root', default='../../DataSet/AdienceFace', type=str)
    parser.add_argument('--scheduler', default=4, type=int)

    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--cosine_scale', default=7., type=float)
    parser.add_argument('--poisson_tau', default=0.0001, type=float)
    parser.add_argument('--constraint', default='U-P', type=str, help='{U-P, U-B, L-L, L-S}')

    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--base_epochs', default=12, type=int)
    parser.add_argument('--lr', default=0.2, type=float)
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
    train_data_loader, val_data_loader, test_data_loader, train_sampler = af_dataset.get_data_loader(args.root, fold_idx, args.train_batch_size, args.eval_batch_size)

    model = CplModel(num_ranks=8, dim=args.feature_dim, cosine_scale=args.cosine_scale, poisson_tau=args.poisson_tau, constraint=args.constraint).cuda()
    model = parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)

    optim_parameters = [
        {'params': [p for n, p in model.module.named_parameters() if n.startswith('feature_extractor') and p.requires_grad]},
        {'params': [p for n, p in model.module.named_parameters() if not n.startswith('feature_extractor') and p.requires_grad], 'lr': args.lr * args.anchor_lr_mul}
    ]
    optimizer = optim.SGD(optim_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    milestones = [args.scheduler * ms for ms in args.base_milestones]
    lr_lambda = ll.get_warm_up_multi_step_lr_lambda(len(train_data_loader), args.warm_up_epoch, args.warm_up_ratio, milestones, args.step_gamma)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    min_val_score = 1000
    epochs = args.scheduler * args.base_epochs
    for epoch_idx in range(epochs):
        st = time.time()

        train_sampler.set_epoch(epoch_idx)
        epoch_loss = engine.train(model, optimizer, lr_scheduler, train_data_loader)
        dist.reduce(epoch_loss, dst=args.master_rank, op=dist.ReduceOp.SUM)

        acc, mae = engine.val(model, val_data_loader)
        dist.reduce(acc, dst=args.master_rank, op=dist.ReduceOp.SUM)
        dist.reduce(mae, dst=args.master_rank, op=dist.ReduceOp.SUM)

        if dist.get_rank() == args.master_rank:
            epoch_loss = epoch_loss.item() / dist.get_world_size()
            t = time.time() - st

            acc = acc.item() / dist.get_world_size()
            mae = mae.item() / dist.get_world_size()

            print(f'Fold: {fold_idx}; Epoch: {epoch_idx + 1:>2} / {epochs}; Train: [Loss: {epoch_loss:.4f}]; Val: [ACC: {acc:.1f}; MAE: {mae:.2f}]; Time: {t:.3f}s.')

            if mae <= min_val_score:
                min_val_score = mae
                torch.save(model.module.state_dict(), f'model_adience_face_{fold_idx}.pkl')

    time.sleep(1)

    model.module.load_state_dict(torch.load(f'model_adience_face_{fold_idx}.pkl'))
    acc, mae = engine.val(model, test_data_loader)

    time.sleep(1)

    dist.reduce(acc, dst=args.master_rank, op=dist.ReduceOp.SUM)
    dist.reduce(mae, dst=args.master_rank, op=dist.ReduceOp.SUM)

    if dist.get_rank() == args.master_rank:
        acc = acc.item() / dist.get_world_size()
        mae = mae.item() / dist.get_world_size()
        print(f'Test: [ACC: {acc:.1f}; MAE: {mae:.2f}].\n')

    return acc, mae


def main():
    args = get_args_parser()
    dist.init_process_group(backend='nccl')
    set_random_seed(args.random_seed + dist.get_rank())
    torch.cuda.set_device(torch.device(f'cuda:{dist.get_rank()}'))

    acc_list, mae_list = [], []
    for fold_idx in range(5):
        acc, mae = run_fold(fold_idx, args)
        acc_list.append(acc)
        mae_list.append(mae)

    if dist.get_rank() == args.master_rank:
        acc_mean = np.array(acc_list).mean()
        acc_std = np.array(acc_list).std()

        mae_mean = np.array(mae_list).mean()
        mae_std = np.array(mae_list).std()

        print(f'ACC: [{acc_list[0]:.1f}, {acc_list[1]:.1f}, {acc_list[2]:.1f}, {acc_list[3]:.1f}, {acc_list[4]:.1f}]; {acc_mean:.1f} ± {acc_std:.1f}.')
        print(f'MAE: [{mae_list[0]:.2f}, {mae_list[1]:.2f}, {mae_list[2]:.2f}, {mae_list[3]:.2f}, {mae_list[4]:.2f}]; {mae_mean:.2f} ± {mae_std:.2f}.')


if __name__ == '__main__':
    main()
