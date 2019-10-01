from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import random
from skimage.color import rgb2lab, rgb2gray
import numpy as np
from PIL import Image
import torch

class BaseLoader(Dataset):
    def __init__(self, config, file_list, mode):
        """
        :param file_list(list): all file list
               mode(string): 'train' = Training mode, 'test' = Testing mode
        """
        self.name = "DataLoader"
        self.step = 0
        self.config = config
        self.mode = mode

        if type(file_list) is str:
            self.file_list = [filename[:-1] for filename in open(file_list, 'r').readlines()]
        else:
            self.file_list = file_list

        if config.DATASET_SIZE is not -1:
            self.reset()
            self.file_list = self.file_list[:config.DATASET_SIZE]

        print('[INFO] Dataloader Format Example: Mode {}'.format(mode))
        print(self.file_list[0])
        print('[INFO] must to check dataloader format')

    def reset(self):
        random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)