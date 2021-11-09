import os
import random

import datasets.vision_dataset as vd


def get_samples(root):
    decades = ['1930s', '1940s', '1950s', '1960s', '1970s']
    train_samples, val_samples, test_samples = [], [], []
    for label, decade in enumerate(decades):
        img_names = os.listdir(os.path.join(root, decade))
        train_names = random.sample(img_names, k=210)
        val_test_names = set(img_names) - set(train_names)
        val_names = random.sample(val_test_names, k=5)
        test_names = list(val_test_names - set(val_names))

        for train_name in train_names:
            train_samples.append((os.path.join(root, decade, train_name), label))
        for val_name in val_names:
            val_samples.append((os.path.join(root, decade, val_name), label))
        for test_name in test_names:
            test_samples.append((os.path.join(root, decade, test_name), label))

    return train_samples, val_samples, test_samples


def get_data_loaders(root, batch_size):
    train_samples, val_samples, test_samples = get_samples(root)

    train_data_loader, _ = vd.get_data_loader(train_samples, batch_size, is_train=True, use_ddp=False)
    val_data_loader, _ = vd.get_data_loader(val_samples, batch_size, is_train=False, use_ddp=False)
    test_data_loader, _ = vd.get_data_loader(test_samples, batch_size, is_train=False, use_ddp=False)

    return train_data_loader, val_data_loader, test_data_loader
