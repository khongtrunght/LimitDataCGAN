from argparse import ArgumentParser
from mnist_data_module import MNISTDataModule
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
# from pl_bolts.datamodules import MNISTDataModule

parser = ArgumentParser()
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to configuration file',
                    default='configs/gp-cgan-best.yaml')

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
    dm = MNISTDataModule(
        **config['data_model_params'],)
    trainer = pl.Trainer(logger=logger, **config['trainer_params'])
    print(f"======= Training {config['model_params']['name']} ======")
    trainer.fit(model, dm)
