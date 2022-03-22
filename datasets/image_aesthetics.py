import os

import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection

import datasets.vision_dataset as vd


class ImageAestheticsDatasets(vd.VisionDatasets):
    def __init__(self, root, fold_idx):
        super(ImageAestheticsDatasets, self).__init__()
        root, category = root.split('#')
        self.train_samples, self.val_samples, self.test_samples = self.get_samples(root, category, fold_idx)
        self.num_ranks = 5

    @staticmethod
    def get_samples(root, category, fold_idx):
        tsv_file = pd.read_csv(os.path.join(root, 'beauty-icwsm15-dataset.tsv'), sep='\t', header=0)
        img_paths, labels = [], []
        for photo_id, category, scores in zip(tsv_file['#flickr_photo_id'], tsv_file['category'], tsv_file['beauty_scores']):
            if category == category:
                img_paths.append(os.path.join(root, 'img_files', f'{photo_id}.jpg'))
                labels.append(int(np.median(np.median([int(s) for s in scores.split(',')]))) - 1)

        skf = model_selection.StratifiedKFold(n_splits=5)
        train_val_ids, test_ids = skf.split(img_paths, labels)[fold_idx]

        train_val_paths = [img_paths[i] for i in train_val_ids]
        train_val_labels = [labels[i] for i in train_val_ids]
        train_paths, val_paths, train_labels, val_labels = model_selection.train_test_split(train_val_paths, train_val_labels, test_size=0.067, stratify=train_val_labels)
        test_paths = [labels[i] for i in train_val_ids]
        test_labels = [labels[i] for i in test_ids]

        train_samples = [(p, l) for p, l in zip(train_paths, train_labels)]
        val_samples = [(p, l) for p, l in zip(val_paths, val_labels)]
        test_samples = [(p, l) for p, l in zip(test_paths, test_labels)]

        return train_samples, val_samples, test_samples

    def get_train_dataset(self):
        return vd.VisionDataset(self.train_samples, is_train=True)

    def get_val_dataset(self):
        return vd.VisionDataset(self.val_samples, is_train=False)

    def get_test_dataset(self):
        return vd.VisionDataset(self.test_samples, is_train=False)
