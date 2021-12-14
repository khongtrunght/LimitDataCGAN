from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import torch

from argparse import ArgumentParser

from data_models.animal_face_datamodule import AnimalFaceDataModule

from models import *
import models.BigGAN as biggan


def setup_model(name, dataset_size, config_model, resume=None, biggan_pretrain_path='./data/G_ema.pth'):
    if name == 'TransferBigGAN':
        generator = biggan.Generator(**bigagn128config)
        generator.load_state_dict(torch.load(
            biggan_pretrain_path, map_location=lambda storage, loc: storage))
        model = tranfer_models[config_model
                               ['name']](generator=generator, **config_model)
        return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        default='configs/animal_biggan.yaml')

    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='model weights to resume')

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

    model = setup_model(config['model_params']['name'],
                        config['data_model_params']['dataset_size'], config['model_params'], args.resume)
    dataset = AnimalFaceDataModule(
        **config['data_model_params'],)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='transfer-biggan-{epoch:03d}'
    )

    trainer = Trainer(callbacks=[checkpoint_callback],
                      logger=logger, **config['trainer_params'])

    trainer.fit(model, dataset)
