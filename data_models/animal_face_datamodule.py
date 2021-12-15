from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from torchvision import datasets
import os

from .setup_dataloader_smallgan import setup_dataloader

PATH_DATASETS = "data/afhq"
BATCH_SIZE = 32
NUM_WORKERS = int(os.cpu_count() / 2)


class AnimalFaceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        img_size: int = 64,
        data_size: int = 10000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
        ])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (3, 64, 64)
        self.num_classes = 3
        self.data_size = data_size

    def train_dataloader(self):
        # return DataLoader(
        #     self.data_train,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=True,
        # )
        return setup_dataloader("animal", batch_size=self.batch_size, num_workers=self.num_workers, data_size=self.data_size//3)

    # def val_dataloader(self):
    #     return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #    return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
