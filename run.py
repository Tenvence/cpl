import argparse
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data as data

import engine
import utils


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=970423, type=int)
    parser.add_argument('--scheduler', default=4, type=int)

    parser.add_argument('--dataset', default='AF', help='{AF, HC, IA, MI}')
    parser.add_argument('--af_root', default='../../DataSet/AdienceFace', type=str)

    parser.add_argument('--constraint', default='S-P', type=str, help='{UPL, S-P, S-B, H-L, H-S}')
    parser.add_argument('--feature_extractor', default='V16', type=str, help='{V16, R50, R101}')
    parser.add_argument('--metric_method', default='E', type=str, help='{E, C}')
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--cosine_scale', default=7., type=float)
    parser.add_argument('--tau', default=0.1, type=float)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--master_rank', default=0, type=int)

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_fold(args, fold_idx):
    train_dataset, val_dataset, test_dataset, num_ranks = utils.get_train_val_test_datasets(args, fold_idx)

    train_dist_sampler = data.distributed.DistributedSampler(train_dataset)
    val_dist_sampler = data.distributed.DistributedSampler(val_dataset)

    train_data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=8, sampler=train_dist_sampler, pin_memory=True, drop_last=True)
    val_data_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=8, sampler=val_dist_sampler, pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, args.batch_size, num_workers=8, pin_memory=True)

    model, criterion = utils.get_model_criterion(num_ranks, args)
    model = parallel.DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    optimizer = optim.Adam(model.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_path = 'model.pkl'

    epochs = 30
    best_val_mae = 100
    for epoch_idx in range(epochs):
        train_dist_sampler.set_epoch(epoch_idx)

        epoch_loss, t_train = engine.train(model, criterion, optimizer, train_data_loader)
        dist.reduce(epoch_loss, dst=args.master_rank, op=dist.ReduceOp.SUM)

        _, val_mae, val_t = engine.val(model, val_data_loader)
        dist.reduce(val_mae, dst=args.master_rank, op=dist.ReduceOp.SUM)

        if dist.get_rank() == args.master_rank:
            epoch_loss = epoch_loss.item() / dist.get_world_size()
            val_mae = val_mae.item() / dist.get_world_size()
            print_str = f'Fold:{fold_idx};Epoch:{epoch_idx + 1}/{epochs};Train:[Loss={epoch_loss:.4f};Time={t_train:.2f}s];Val:[MAE={val_mae:.3f};Time={val_t:.2f}s];'
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.module.state_dict(), model_path)
                print_str += ' :-)'
            print(print_str)

    test_acc, test_mae = 0, 0
    if dist.get_rank() == args.master_rank:
        model.module.load_state_dict(torch.load(model_path))
        test_acc, test_mae, test_t = engine.val(model, test_data_loader)
        test_acc, test_mae = test_acc.item(), test_mae.item()
        print(f'Fold:{fold_idx};Test:[ACC:{test_acc:.3f},MAE:{test_mae:.3f}];Time:{test_t:.2f}s.')

    return test_acc, test_mae


def main():
    st = time.time()

    args = get_args_parser()
    dist.init_process_group(backend='nccl')
    set_random_seed(args.random_seed + dist.get_rank())
    torch.cuda.set_device(torch.device(f'cuda:{dist.get_rank()}'))

    num_folds = utils.get_num_folds(args)

    test_acc_list, test_mae_list = [], []
    for fold_idx in range(num_folds):
        test_acc, test_mae = run_fold(args, fold_idx)
        test_acc_list.append(test_acc)
        test_mae_list.append(test_mae)

    if dist.get_rank() == args.master_rank:
        test_acc_mean = np.array(test_acc_list).mean()
        test_acc_std = np.array(test_acc_list).std()

        test_mae_mean = np.array(test_mae_list).mean()
        test_mae_std = np.array(test_mae_list).std()

        print(f'FinalTest:[ACC:{test_acc_mean:.3f}±{test_acc_std:.3f};MAE:{test_mae_mean:.3f}±{test_mae_std:.3f}].Time={time.time() - st:.2f}s.')


if __name__ == '__main__':
    main()
