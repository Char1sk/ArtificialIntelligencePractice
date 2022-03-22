# -*- coding: utf-8 -*-
# @Author  : Fraunhofer0126
# @Time    : 2022/03/22

import os
import numpy as np
import pickle
import torch
from PIL import Image
from typing import Any, Callable, Optional, Tuple


class MyCifar10:

    """
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:

        self.root = root
        self.transform = transform
        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class MyDataLoader:
    """
    Args:
        dataset (MyCifar10): a object from MyCifar10 class
        train (batch_size, optional)
    """
    def __init__(self, dataset, batch_size=1) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_iter = len(dataset)
        self.count = 0

    def __iter__(self) -> 'MyDataLoader':
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if(self.count < self.max_iter):
            dataTensor = None
            labelTensor = None
            for index in range(self.count, self.count + self.batch_size):     
                data, label = self.dataset[index]
                data = data.unsqueeze(0)
                if dataTensor is None:
                    dataTensor = data
                else:
                    dataTensor = torch.cat((dataTensor, data), 0)
                label = torch.from_numpy(np.array(label)).unsqueeze(0)
                if labelTensor is None:
                    labelTensor = label
                else:
                    labelTensor = torch.cat((labelTensor, label), 0)
            self.count += self.batch_size
            return [dataTensor, labelTensor]
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self.dataset)
