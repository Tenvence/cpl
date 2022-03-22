import os

import datasets.vision_dataset as vd


class HistoricalColorDatasets(vd.VisionDatasets):
    def __init__(self, root, fold_idx):
        super(HistoricalColorDatasets, self).__init__()
        self.train_samples, self.val_samples, self.test_samples = self.get_samples(root, fold_idx)
        self.num_ranks = 5

    @staticmethod
    def get_samples(root, fold_idx):
        decades = ['1930s', '1940s', '1950s', '1960s', '1970s']
        train_samples, val_samples, test_samples = [], [], []
        for label, decade in enumerate(decades):
            img_names = os.listdir(os.path.join(root, decade))
            val_names = img_names[:5]
            train_test_names = img_names[5:]
            test_names = train_test_names[fold_idx * 23:fold_idx * 23 + 50]
            train_names = list(set(train_test_names) - set(test_names))
            print(len(train_names), len(val_names), len(test_names))
            for train_name in train_names:
                train_samples.append((os.path.join(root, decade, train_name), label))
            for val_name in val_names:
                val_samples.append((os.path.join(root, decade, val_name), label))
            for test_name in test_names:
                test_samples.append((os.path.join(root, decade, test_name), label))
        return train_samples, val_samples, test_samples

    def get_train_dataset(self):
        return vd.VisionDataset(self.train_samples, is_train=True)

    def get_val_dataset(self):
        return vd.VisionDataset(self.val_samples, is_train=False)

    def get_test_dataset(self):
        return vd.VisionDataset(self.test_samples, is_train=False)
