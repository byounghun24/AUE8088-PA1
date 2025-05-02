# Python packages
from termcolor import colored
from tqdm import tqdm
import os
import tarfile
import wget
import torch
import numpy as np

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Custom packages
import src.config as cfg

def mixup_collate_fn(batch, alpha=1.0, num_classes=200):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # MixUp lambda
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0))

    mixed_images = lam * images + (1 - lam) * images[indices]

    # one-hot 라벨
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[indices]

    return mixed_images, mixed_labels

class TinyImageNetDatasetModule(LightningDataModule):
    __DATASET_NAME__ = 'tiny-imagenet-200'

    def __init__(self, batch_size: int = cfg.BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        if not os.path.exists(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__)):
            # download data
            print(colored("\nDownloading dataset...", color='green', attrs=('bold',)))
            filename = self.__DATASET_NAME__ + '.tar'
            wget.download(f'https://hyu-aue8088.s3.ap-northeast-2.amazonaws.com/{filename}')

            # extract data
            print(colored("\nExtract dataset...", color='green', attrs=('bold',)))
            with tarfile.open(name=filename) as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(path=cfg.DATASET_ROOT_PATH, member=member)
            os.remove(filename)

    def train_dataloader(self):
        tf_train = transforms.Compose([
            transforms.RandomRotation(cfg.IMAGE_ROTATION),
            transforms.RandomHorizontalFlip(cfg.IMAGE_FLIP_PROB),
            transforms.RandomCrop(cfg.IMAGE_NUM_CROPS, padding=cfg.IMAGE_PAD_CROPS),
            # transforms.RandomResizedCrop(cfg.IMAGE_NUM_CROPS, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
            # transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'train'), tf_train)
        msg = f"[Train]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
            collate_fn=lambda batch: mixup_collate_fn(batch, alpha=0.2, num_classes=cfg.NUM_CLASSES)
        )

    def val_dataloader(self):
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'val'), tf_val)
        msg = f"[Val]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            pin_memory=True,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'test'), tf_test)
        msg = f"[Test]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )
