import random
from abc import ABC, abstractmethod

import numpy as np
import torch.utils.data as th_data
import torchvision.transforms as tf


# Reffered to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/base_dataset.py
class BaseDataset(th_data.Dataset, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __getitem__(self, index):
        return NotImplementedError

    @abstractmethod
    def __len__(self):
        return NotImplementedError


# Reffered to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/unaligned_dataset.py
class UnalignedDataset(BaseDataset):
    def __init__(
        self,
        sim_data: np.ndarray,
        real_data: np.ndarray,
        sim_transform_list=[],
        real_transform_list=[],
        random: bool = True,
    ):
        super().__init__()

        self.sim_data = sim_data
        self.real_data = real_data

        self.sim_data_size = len(self.sim_data)
        self.real_data_size = len(self.real_data)

        self.sim_transform = tf.Compose(sim_transform_list)
        self.real_transform = tf.Compose(real_transform_list)

        self.random = random

    def __getitem__(self, index):
        sim_img = self.sim_data[index % self.sim_data_size]
        if self.random:
            real_img = self.real_data[random.randint(0, self.real_data_size - 1)]
        else:
            real_img = self.real_data[index % self.real_data_size]
        return {"A": self.sim_transform(sim_img), "B": self.real_transform(real_img)}

    def __len__(self):
        return max(self.sim_data_size, self.real_data_size)
