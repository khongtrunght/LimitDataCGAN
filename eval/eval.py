from argparse import ArgumentParser
from models.transferBigGAN import TransferBigGAN
from runner import setup_model
import visualizers
import yaml
from models import *
import models.BigGAN as biggan

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


ckpt_path = "D:\OneDrive - Hanoi University of Science and Technology\Downloads\\transfer-biggan2-epoch=699.ckpt"
biggan_pretrain_path = './data/G_ema.pth'
generator = biggan.Generator(**bigagn128config)
generator.load_state_dict(torch.load(
    biggan_pretrain_path, map_location=lambda storage, loc: storage))
model = TransferBigGAN.load_from_checkpoint(
    ckpt_path, generator=generator, data_size=config['data_model_params']['data_size'], **config['model_params'])


model.eval()


visualizers.random(model, "test/random.png")
visualizers.interpolate(model, "test/interpolate.png", 0, 10)
visualizers.reconstruct(model, "test/reconstruct.png",
                        [[0, 1, 2, 3], [0, 0, 0, 0]])
