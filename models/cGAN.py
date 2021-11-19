from torchsummary import summary
from typing import OrderedDict
import numpy as np
from pytorch_lightning import loggers
import torch
import matplotlib.pyplot as plt

from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger

from argparse import ArgumentParser

import yaml

EMBEDDING_SIZE = 64


class Discriminator(nn.Module):

    def __init__(self, img_shape, embedd_dim):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.embedding = nn.Embedding(EMBEDDING_SIZE, embedd_dim)
        self.embedding_layer = nn.Sequential(
            self.embedding,
            nn.Linear(embedd_dim, int(np.prod(img_shape[1:])))
        )

        def block(in_channel, out_channel, normalize=True):
            layers = [nn.Conv2d(in_channel, out_channel,
                                kernel_size=3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(img_shape[0] + 1, 64, normalize=False),
            *block(64, 64 * 2),
            nn.Flatten(),
            nn.Linear(int(128 * img_shape[1] * img_shape[1] / 16), 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        image = x.view(-1, *self.img_shape)
        embedding = self.embedding_layer(y).view(-1,1, *self.img_shape[1:])
        input = torch.cat((image, embedding), 1)
        output = self.model(input)
        return output  # probability of real


class Generator(nn.Module):

    def __init__(self, latent_dim, embedd_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        self.embedding = nn.Embedding(EMBEDDING_SIZE, embedd_dim)

        self.embedding_layer = nn.Sequential(
            self.embedding,
            nn.Linear(embedd_dim, 56*56),
        )

        def block(in_channel, out_channel, normalize=True):
            layers = [nn.ConvTranspose2d(in_channel, out_channel,
                                         kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(129, 64),
            *block(64, 32),
            nn.Conv2d(32, img_shape[0], kernel_size=7,
                      stride=1, padding='same'),
            nn.Tanh(),
        )

        self.to_shape = nn.Linear(latent_dim, int(np.prod(self.img_shape[1:])*128 / 16))

    def forward(self, z, y):
        y = y.long()
        latent = self.embedding_layer(y).view(-1, 1, int(self.img_shape[1]/4), int(self.img_shape[1]/4))
        z = self.to_shape(z).view(-1, 128, int(self.img_shape[1]/4), int(self.img_shape[1]/4))
        input = torch.cat((z, latent), 1)
        output = self.model(input).view(-1, *self.img_shape)
        return output


class CGAN(pl.LightningModule):
    def __init__(
        self,
        img_shape=(1, 28, 28),
        embedd_dim=50,
        learning_rate=0.0002,
        latent_dim=100,
        batch_size=144,
        num_classes=10,
        **kwargs
    ):
        super(CGAN, self).__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.embedd_dim = embedd_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.discriminator = Discriminator(
            img_shape=img_shape, embedd_dim=embedd_dim)
        self.generator = Generator(latent_dim=latent_dim,
                                   embedd_dim=embedd_dim,
                                   img_shape=img_shape)

        self.validation_z = torch.randn(
            8, self.latent_dim, requires_grad=False)

    def forward(self, z, y):
        return self.generator(z, y)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        # noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)  # dua z ve cung device
        # random label cho z
        z_labels = torch.randint(0, self.num_classes, (imgs.shape[0], 1)).to(self.device)

        # imgs = imgs*2 - 1

        # train discriminator
        if optimizer_idx == 1:
            valid = torch.ones(imgs.shape[0], 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(
                self.discriminator(imgs, labels), valid)

            fake = torch.zeros(imgs.shape[0], 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(
                    self(z, z_labels),
                    z_labels),
                fake)

            d_loss = (real_loss + fake_loss)/2

            tensorboard_logs = {'d_loss': d_loss}
            output = {"loss": d_loss, "progress_bar": tensorboard_logs,
                      "log": tensorboard_logs}

            # self.log("d_loss", d_loss, on_step=True, on_epoch=True)
            return output

        if optimizer_idx == 0:
            # train generator
            self.generated_imgs = self(z, z_labels)

            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image(
                "generated_images", grid, self.global_step)

            valid = torch.ones(self.generated_imgs.shape[0], 1) * 0.9
            valid = valid.type_as(imgs)

            g_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs, z_labels), valid)
            tensorboard_logs = {'g_loss': g_loss}

            output = {"loss": g_loss, "progress_bar": tensorboard_logs,
                      "log": tensorboard_logs}
            # self.log("g_loss", g_loss, on_step=True, on_epoch=True)
            return output

    def configure_optimizers(self):
        lr = self.learning_rate

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return opt_g, opt_d

    def on_train_epoch_end(self):
        z = self.validation_z.to(self.device)
        for i in range(self.num_classes):
            z_labels = torch.ones((z.shape[0], 1)) * i
            z_labels = z_labels.to(self.device)
            self.generated_imgs = self(z, z_labels)
            grid = torchvision.utils.make_grid(self.generated_imgs)
            self.logger.experiment.add_image(
                "generated_images_condition", grid, i)
        # plt.imshow(grid[0].detach().cpu().numpy().squeeze(), cmap='gray')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        d_outputs = outputs[0]
        g_outputs = outputs[1]
        avg_dloss = torch.stack([x['loss'] for x in d_outputs]).mean()
        avg_gloss = torch.stack([x['loss'] for x in g_outputs]).mean()
        self.logger.log_metrics({"d_loss": avg_dloss, "g_loss": avg_gloss},
                                self.current_epoch)
