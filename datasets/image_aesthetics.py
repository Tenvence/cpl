import os
import json

import datasets.vision_dataset as vd


class ImageAestheticsDatasets(vd.VisionDatasets):
    def __init__(self, root, fold_idx):
        super(ImageAestheticsDatasets, self).__init__()
        root, category = root.split('#')
        self.train_samples, self.val_samples, self.test_samples = self.get_samples(root, category, fold_idx)
        self.num_ranks = 5

    @staticmethod
    def parse_samples(root, file_name):
        with open(os.path.join(root, file_name), 'r') as f:
            samples = [line.strip().split(',') for line in f.readlines()]
            samples = [(os.path.join(root, 'img_files', f'{i}.jpg'), int(l)) for i, l in samples]
        return samples

    @staticmethod
    def get_samples(root, category, fold_idx):
        with open(os.path.join(root, f'{category}_fold_{fold_idx}.txt'), 'r') as f:
            sample_dict = json.load(f)
        train_samples = [(os.path.join(root, 'img_files', f'{i}.jpg'), int(l)) for i, l in sample_dict['train']]
        val_samples = [(os.path.join(root, 'img_files', f'{i}.jpg'), int(l)) for i, l in sample_dict['val']]
        test_samples = [(os.path.join(root, 'img_files', f'{i}.jpg'), int(l)) for i, l in sample_dict['test']]
        return train_samples, val_samples, test_samples

    def get_train_dataset(self):
        return vd.VisionDataset(self.train_samples, is_train=True)

    def get_val_dataset(self):
        return vd.VisionDataset(self.val_samples, is_train=False)

    def get_test_dataset(self):
        return vd.VisionDataset(self.test_samples, is_train=False)
