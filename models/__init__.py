from .cGAN import *
from .gp_cgan_32 import GPCGANBEST


gan_models = {
    'CGAN': CGAN,
    # 'GP-CGAN': GPCGAN,
    # 'GP-CGAN28': GPCGAN28,
    'GP-CGAN-best': GPCGANBEST,
}
