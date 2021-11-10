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


def get_datasets(root, fold_idx):
    train_samples = get_samples(root, fold_idx, 'age_train.txt')
    val_samples = get_samples(root, fold_idx, 'age_val.txt')
    test_samples = get_samples(root, fold_idx, 'age_test.txt')

    train_dataset = vd.get_dataset(train_samples, is_train=True)
    val_dataset = vd.get_dataset(val_samples, is_train=False)
    test_dataset = vd.get_dataset(test_samples, is_train=False)

    return train_dataset, val_dataset, test_dataset
