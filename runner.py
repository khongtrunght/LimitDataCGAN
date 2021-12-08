from argparse import ArgumentParser
from data_models.animal_face_datamodule import AnimalFaceDataModule
from models import *

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml
from pytorch_lightning import loggers
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import datasets
from data_models.mnist_datamodule import MNISTDataModule
from utils.utils import model_init

parser = ArgumentParser()
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to configuration file',
                    default='configs/dcgan.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as f:
    try:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)


logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
)

torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])

if __name__ == "__main__":
    model = gan_models[config['model_params']
                       ['name']](**config['model_params'])
    # chk_path = "lightning_logs/GP-CGAN-best/version_3/checkpoints/epoch=63-step=3007.ckpt"
    # model = model.load_from_checkpoint(chk_path, **config['model_params'])
    dm = AnimalFaceDataModule(
        **config['data_model_params'],)
    trainer = pl.Trainer(logger=logger, **config['trainer_params'])
    print(f"======= Training {config['model_params']['name']} ======")
    # torch.save(model.state_dict(), "model.pth")

    preD = torch.load("pretrains/netD_epoch_199.pth")
    preG = torch.load("pretrains/netG_epoch_199.pth")

    model = model_init(model, preG, preD)
    trainer.fit(model, dm)
