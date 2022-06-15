import glob
import os

import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data_models.ImageListDataset import ImageListDataset

PATH_DATASETS = "data/afhq/train"
BATCH_SIZE = 32
NUM_WORKERS = int(os.cpu_count() / 2)


class AnimalFaceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        img_size: int = 128,
        data_size: int = 10000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, self.img_size, self.img_size)
        self.num_classes = 3
        assert data_size % 3 == 0
        self.data_size = data_size

    def setup(self, stage=None):
        img_path_dict = {}

        labels_dict = {'cat': 0, 'dog': 1, 'wild': 2}
        for label in labels_dict.keys():
            img_path_dict[label] = glob.glob(f"data/afhq/train/{label}/*.jpg")
            img_path_dict[label] = img_path_dict[label][:self.data_size//3]

        img_path_list = []

        for label in labels_dict.keys():
            img_path_list.extend([(path, labels_dict[label])
                                  for path in img_path_dict[label]])

        # tra lai label i la so thu tu, data[1] la label
        img_path_list = [(data[0], (i, data[1]))
                         for i, data in enumerate(sorted(img_path_list))]

        self.dataset = ImageListDataset(
            img_path_list, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
