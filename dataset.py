import random
from abc import ABC, abstractmethod

import torch.utils.data as th_data


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
    def __init__(self, sim_data, real_data, random: bool = True):
        super().__init__()

        self.sim_data = sim_data
        self.real_data = real_data
        self.random = random

        self.sim_data_size = len(self.sim_data)
        self.real_data_size = len(self.real_data)

    def __getitem__(self, index):
        sim_img = self.sim_data[index % self.sim_data_size]
        if self.random:
            real_img = self.real_data[random.randint(0, self.real_data_size - 1)]
        else:
            real_img = self.real_data[index % self.real_data_size]
        return {"A": sim_img, "B": real_img}

    def __len__(self):
        return max(self.sim_data_size, self.real_data_size)
