"""Some parts adapted from the TD-VAE code by Xinqiang Ding <xqding@umich.edu>."""

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets

from pylego.reader import DatasetReader


class MovingMNISTDataset(datasets.MNIST):

    def __init__(self, data_path, train, seq_len, binarize):
        super().__init__(data_path, train=train, download=True)
        self.seq_len = seq_len
        self.binarize = binarize

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        image = np.array(image)

        if self.binarize:
            tmp = np.random.rand(28, 28) * 255
            image = tmp <= image
            image = image.astype(np.float32)
        else:
            image = image.astype(np.float32) / 255.0

        # randomly choose a direction and generate a sequence of images that move in the chosen direction
        direction = np.random.choice(2)
        image = np.roll(image, np.random.randint(28), 1)  # start with a random roll
        image_list = [image.reshape(-1)]
        for _ in range(1, self.seq_len):
            if direction:
                image = np.roll(image, -1, 1)
                image_list.append(image.reshape(-1))
            else:
                image = np.roll(image, 1, 1)
                image_list.append(image.reshape(-1))

        return np.array(image_list)


class MovingMNISTReader(DatasetReader):

    def __init__(self, data_path, seq_len=20, binarize=False):
        train_dataset = MovingMNISTDataset(data_path, True, seq_len, binarize)
        test_dataset = MovingMNISTDataset(data_path, False, seq_len, binarize)

        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        torch.manual_seed(0)
        train_dataset, val_dataset = data.random_split(train_dataset, [train_size, val_size])
        super().__init__({'train': train_dataset, 'val': val_dataset, 'test': test_dataset})
