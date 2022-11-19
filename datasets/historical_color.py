import json
import os

import datasets.vision_dataset as vd


class HistoricalColorDatasets(vd.VisionDatasets):
    def __init__(self, root, fold_idx):
        super(HistoricalColorDatasets, self).__init__()
        self.train_samples, self.val_samples, self.test_samples = self.get_samples(root, fold_idx)
        self.num_ranks = 5

    @staticmethod
    def get_samples(root, fold_idx):
        with open(os.path.join(root, f'fold_{fold_idx}.txt'), 'r') as f:
            samples = json.loads(f.read())

        train_samples = [(os.path.join(root, decade, name), label) for (decade, name, label) in samples['train']]
        val_samples = [(os.path.join(root, decade, name), label) for (decade, name, label) in samples['val']]
        test_samples = [(os.path.join(root, decade, name), label) for (decade, name, label) in samples['test']]

        return train_samples, val_samples, test_samples

    def get_train_dataset(self):
        return vd.VisionDataset(self.train_samples, is_train=True)

    def get_val_dataset(self):
        return vd.VisionDataset(self.val_samples, is_train=False)

    def get_test_dataset(self):
        return vd.VisionDataset(self.test_samples, is_train=False)
