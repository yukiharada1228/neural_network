import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils import data


class ImageNet32(data.Dataset):
    def __init__(
        self,
        data_folder,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            data_list = sorted(
                data_folder.glob("Imagenet32_train/train_data_batch_*"),
                key=lambda x: int(x.stem.split("_")[-1]),
            )
        else:
            data_list = data_folder.glob("val_data")

        self.data: Any = []
        self.targets = []

        for file_path in data_list:
            with open(file_path, "rb") as f:
                entry = pickle.load(f)
                self.data.append(entry["data"])
                self.targets.extend([label - 1 for label in entry["labels"]])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

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

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
