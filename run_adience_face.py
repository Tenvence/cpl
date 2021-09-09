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
    parser.add_argument('--poisson_tau', default=0.1, type=float)
    parser.add_argument('--binomial_tau', default=0.1, type=float)
    parser.add_argument('--constraint', default='S-P', type=str, help='{S-P, S-B, H-L, H-S}')

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

    model = CplModel(8, args.feature_dim, args.cosine_scale, args.poisson_tau, args.binomial_tau, args.constraint).cuda()
    model = parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)

    optim_parameters = [
        {'params': [p for n, p in model.module.named_parameters() if n.startswith('feature_extractor') and p.requires_grad]},
        {'params': [p for n, p in model.module.named_parameters() if not n.startswith('feature_extractor') and p.requires_grad], 'lr': args.lr * args.anchor_lr_mul}
    ]
    optimizer = optim.SGD(optim_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    milestones = [args.scheduler * ms for ms in args.base_milestones]
    lr_lambda = ll.get_warm_up_multi_step_lr_lambda(len(train_data_loader), args.warm_up_epoch, args.warm_up_ratio, milestones, args.step_gamma)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0
    best_mae = 1000
    epochs = args.scheduler * args.base_epochs
    for epoch_idx in range(epochs):
        st = time.time()

        train_sampler.set_epoch(epoch_idx)
        epoch_loss = engine.train(model, optimizer, lr_scheduler, train_data_loader)
        dist.reduce(epoch_loss, dst=args.master_rank, op=dist.ReduceOp.SUM)

        val_acc, val_mae = engine.val(model, val_data_loader)
        dist.reduce(val_acc, dst=args.master_rank, op=dist.ReduceOp.SUM)
        dist.reduce(val_mae, dst=args.master_rank, op=dist.ReduceOp.SUM)

        if dist.get_rank() == args.master_rank:
            epoch_loss = epoch_loss.item() / dist.get_world_size()
            t = time.time() - st

            val_acc = val_acc.item() / dist.get_world_size()
            val_mae = val_mae.item() / dist.get_world_size()

            print(f'Fold: {fold_idx}; Epoch: {epoch_idx + 1:>2} / {epochs}; Train: [Loss: {epoch_loss:.4f}]; Val: [ACC: {val_acc:.1f}; MAE: {val_mae:.2f}]; Time: {t:.3f}s.')

            if val_mae <= best_mae:
                best_mae = val_mae
                best_acc = val_acc
                torch.save(model.module.state_dict(), 'model_adience_face.pkl')

    time.sleep(1)

    model.module.load_state_dict(torch.load('model_adience_face.pkl'))
    test_acc, test_mae = engine.val(model, test_data_loader)

    print(f'device: {dist.get_rank()}; ACC: {test_acc:.3f}, MAE: {test_mae:.4f}.')
    time.sleep(1)

    dist.reduce(test_acc, dst=args.master_rank, op=dist.ReduceOp.SUM)
    time.sleep(1)
    dist.reduce(test_mae, dst=args.master_rank, op=dist.ReduceOp.SUM)

    if dist.get_rank() == args.master_rank:
        test_acc = test_acc.item() / dist.get_world_size()
        test_mae = test_mae.item() / dist.get_world_size()
        print(f'Test: [ACC: {test_acc:.3f}; MAE: {test_mae:.4f}].\n')

    return best_acc, best_mae, test_acc, test_mae


def main():
    args = get_args_parser()
    dist.init_process_group(backend='nccl')
    set_random_seed(args.random_seed + dist.get_rank())
    torch.cuda.set_device(torch.device(f'cuda:{dist.get_rank()}'))

    val_acc_list, val_mae_list = [], []
    test_acc_list, test_mae_list = [], []
    for fold_idx in range(5):
        val_acc, val_mae, test_acc, test_mae = run_fold(fold_idx, args)
        val_acc_list.append(val_acc)
        val_mae_list.append(val_mae)
        test_acc_list.append(test_acc)
        test_mae_list.append(test_mae)

    if dist.get_rank() == args.master_rank:
        val_acc_mean = np.array(val_acc_list).mean()
        val_acc_std = np.array(val_acc_list).std()

        val_mae_mean = np.array(val_mae_list).mean()
        val_mae_std = np.array(val_mae_list).std()

        print(f'Val ACC: {val_acc_mean:.3f} ± {val_acc_std:.3f}.')
        print(f'Val MAE: {val_mae_mean:.4f} ± {val_mae_std:.4f}.')

        test_acc_mean = np.array(test_acc_list).mean()
        test_acc_std = np.array(test_acc_list).std()

        test_mae_mean = np.array(test_mae_list).mean()
        test_mae_std = np.array(test_mae_list).std()

        print(f'Test ACC: {test_acc_mean:.3f} ± {test_acc_std:.3f}.')
        print(f'Test MAE: {test_mae_mean:.4f} ± {test_mae_std:.4f}.')


if __name__ == '__main__':
    main()
