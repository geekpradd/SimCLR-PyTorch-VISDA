import os
import torch
import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

class VisdaDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = os.path.join(root_dir, "visda")
        self.transform = transform
        self.train = train

        # when train is true, this will return samples from the training set
        if self.train:
            self.root_dir = os.path.join(self.root_dir, "train")
        else:
            self.root_dir = os.path.join(self.root_dir, "test")
        self.categories = (os.listdir(self.root_dir))
        self.categories.remove("image_list.txt")

        self.counter = []
        current = 0 
        for f in self.categories:
            current += len(os.listdir(os.path.join(self.root_dir, f)))
            self.counter.append(current - 1)

        self.length = current

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        index = -1
        for c, val in enumerate(self.counter):
            if idx <= val:
                index = c
                break

        current_count = idx
        if index != 0:
            current_count -= self.counter[index-1] + 1

        folder_path = os.path.join(self.root_dir, self.categories[index])
        file_name = os.listdir(folder_path)[current_count]

        image_path = os.path.join(folder_path, file_name)

        image = io.imread(image_path)

        image = (Image.fromarray(image))
        # print (image[215, 384])
        if self.transform:
            image = self.transform(image)

        return image





# d = VisdaDataset(str(Path(os.getcwd()).parent.absolute()))
# print(len(d))
# d[142797]