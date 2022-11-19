import torch.utils.data as data
import PIL.Image as Image
import torchvision.transforms as transforms


class VisionDataset(data.Dataset):
    def __init__(self, samples, is_train=True):
        super(VisionDataset, self).__init__()

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.eval_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.train_transforms(img) if self.is_train else self.eval_transforms(img)
        return img, label


class VisionDatasets:
    def get_train_dataset(self):
        raise NotImplementedError

    def get_val_dataset(self):
        raise NotImplementedError

    def get_test_dataset(self):
        raise NotImplementedError
