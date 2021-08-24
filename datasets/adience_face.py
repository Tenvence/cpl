import os

import datasets.vision_dataset as vd


def get_samples(root, fold_idx, name):
    samples = []
    with open(os.path.join(root, f'test_fold_is_{fold_idx}', name), 'r') as f:
        lines = f.readlines()
        for line in lines:
            path, label = line.strip().split()
            label = int(label)
            samples.append((os.path.join(root, 'aligned', path), label))
    return samples


def get_data_loader(root, fold_idx, train_batch_size, eval_batch_size):
    train_samples = get_samples(root, fold_idx, 'age_train.txt')
    train_data_loader, train_sampler = vd.get_data_loader(train_samples, train_batch_size, is_train=True, use_ddp=True)

    val_samples = get_samples(root, fold_idx, 'age_val.txt')
    val_data_loader, _ = vd.get_data_loader(val_samples, eval_batch_size, is_train=False, use_ddp=True)

    test_samples = get_samples(root, fold_idx, 'age_test.txt')
    test_data_loader, _ = vd.get_data_loader(test_samples, eval_batch_size, is_train=False, use_ddp=True)

    return train_data_loader, val_data_loader, test_data_loader, train_sampler
