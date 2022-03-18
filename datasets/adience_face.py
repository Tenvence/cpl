import os

import datasets.vision_dataset as vd


class AdienceFaceDatasets(vd.VisionDatasets):
    def __init__(self, root, fold_idx):
        super(AdienceFaceDatasets, self).__init__()
        self.root = root
        self.fold_idx = fold_idx
        self.num_ranks = 8

    def get_samples(self, name):
        samples = []
        with open(os.path.join(self.root, f'test_fold_is_{self.fold_idx}', name), 'r') as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.strip().split()
                label = int(label)
                samples.append((os.path.join(self.root, 'aligned', path), label))
        return samples

    def get_train_dataset(self):
        train_samples = self.get_samples(name='age_train.txt')
        train_dataset = vd.VisionDataset(train_samples, is_train=True)
        return train_dataset

    def get_val_dataset(self):
        val_samples = self.get_samples(name='age_val.txt')
        val_dataset = vd.VisionDataset(val_samples, is_train=False)
        return val_dataset

    def get_test_dataset(self):
        test_samples = self.get_samples(name='age_test.txt')
        test_dataset = vd.VisionDataset(test_samples, is_train=False)
        return test_dataset
