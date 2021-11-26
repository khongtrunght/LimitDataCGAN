import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
import torch.optim as optim
import pytorch_lightning as pl
import torchvision
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, num_classes=3, channels_img=3, num_df=64, img_size=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # input is (channels_img + 1) x 64 x 64
            nn.Conv2d(channels_img + 1, num_df,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_df) x 32 x 32
            self._block(num_df, num_df * 2, 4, 2, 1),
            # state size. (num_df*2) x 16 x 16
            self._block(num_df * 2, num_df * 4, 4, 2, 1),
            # state size. (num_df*4) x 8 x 8
            self._block(num_df * 4, num_df * 8, 4, 2, 1),
            # state size. (num_df*8) x 4 x 4
            nn.Conv2d(num_df * 8, 1, kernel_size=4, stride=2, padding=0),
            # nn.Sigmoid()
        )

        self.embed = nn.Embedding(num_classes, img_size * img_size)
        self.image_size = img_size

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False,),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, labels):
        labels = labels.long()
        embedding = self.embed(
            labels).view(-1, 1, self.image_size, self.image_size)
        input_concat = torch.cat((x, embedding), 1)  # concat along channel
        return self.net(input_concat)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, num_gf=64, img_size=64, embedding_size=128):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.embedding_size = embedding_size
        self.channels_img = channels_img
        self.net = nn.Sequential(
            # input is Z, going into a convolution (channels_noise + embedding_size) x 1 x 1
            self._block(channels_noise + embedding_size, num_gf * 8, 4, 1, 0),
            # state size. (num_gf*8) x 4 x 4
            self._block(num_gf * 8, num_gf * 4, 4, 2, 1),
            # state size. (num_gf*4) x 8 x 8
            self._block(num_gf * 4, num_gf * 2, 4, 2, 1),
            # state size. (num_gf*2) x 16 x 16
            self._block(num_gf * 2, num_gf, 4, 2, 1),
            # state size. (num_gf) x 32 x 32
            nn.ConvTranspose2d(num_gf, channels_img, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output size. (channels_img) x 64 x 64
        )

        self.embed = nn.Embedding(embedding_size, embedding_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding, bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)

    def forward(self, z, labels):
        z = z.view(z.size(0), z.size(1), 1, 1)
        labels = labels.long()
        embedding = self.embed(labels).view(-1, self.embedding_size, 1, 1)
        input_concat = torch.cat((z, embedding), 1)
        return self.net(input_concat).view(-1, self.channels_img, self.img_size, self.img_size)


class GPCGANBEST(pl.LightningModule):
    def __init__(self, channels_noise, channels_img, num_df, num_gf, img_size, num_classes, embedding_size, **kwargs):
        super(GPCGANBEST, self).__init__()
        self.channels_noise = channels_noise
        self.channels_img = channels_img
        self.num_classes = num_classes
        self.img_size = img_size

        self.generator = Generator(
            channels_noise, channels_img, num_gf, img_size, embedding_size)
        self.discriminator = Discriminator(
            num_classes, channels_img, num_df, img_size)

        # weight initialization
        self.generator.apply(self.initialize_weights)
        self.discriminator.apply(self.initialize_weights)

        # hyperparameters
        self.n_critic = kwargs.get('n_critic', 5)
        self.labda_gp = kwargs.get('labda_gp', 10)
        self.learning_rate = kwargs.get('learning_rate', 0.0002)
        self.embedding_size = embedding_size

        # validation_z
        self.validation_z = torch.randn(
            8, self.channels_noise, requires_grad=False)

    def gradient_penalty(self, reals, fakes, labels):
        alpha = torch.rand(reals.shape[0], 1, 1, 1).type_as(reals)
        interpolates = (alpha * reals + (1 - alpha)
                        * fakes).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, labels)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, z, labels):
        return self.generator(z, labels)

    def discriminator_step(self, reals, labels, batch_idx):
        z = torch.randn(reals.shape[0], self.channels_noise,
                        requires_grad=False).type_as(reals)
        fakes = self.generator(z, labels)

        critic_real = self.discriminator(reals, labels).reshape(-1)
        critic_fake = self.discriminator(fakes.detach(), labels).reshape(-1)
        gp = self.gradient_penalty(reals, fakes, labels)
        critic_loss = (
            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.labda_gp * gp)

        return {"loss": critic_loss}

    def generator_step(self, reals, labels, batch_idx):
        z = torch.randn(reals.shape[0], self.channels_noise,
                        requires_grad=False).type_as(reals)
        fakes = self.generator(z, labels)
        critic_fake = self.discriminator(fakes, labels).reshape(-1)

        sample_images = fakes[:6]
        grid = torchvision.utils.make_grid(sample_images, normalize=True)
        self.logger.experiment.add_image("fake_images", grid, self.global_step)
        loss = -critic_fake.mean()

        return {"loss": loss}

    def training_step(self, batch, batch_idx, optimizer_idx):
        reals, labels = batch

        if optimizer_idx == 0:
            return self.discriminator_step(reals, labels, batch_idx)
        elif optimizer_idx == 1:
            if batch_idx % self.n_critic == 0:
                return self.generator_step(reals, labels, batch_idx)
            else:
                return None

    def configure_optimizers(self):
        lr = self.learning_rate

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

        return opt_d, opt_g

    def training_epoch_end(self, outputs):
        # log condition images
        z = self.validation_z.to(self.device)
        for i in range(self.num_classes):
            z_labels = torch.ones((z.shape[0], 1)) * i
            z_labels = z_labels.to(self.device)
            self.generated_imgs = self(z, z_labels)
            grid = torchvision.utils.make_grid(
                self.generated_imgs, normalize=True)
            self.logger.experiment.add_image(
                "generated_images_condition", grid, i)

        # log loss
        d_outputs = outputs[0]
        g_outputs = outputs[1]
        avg_dloss = torch.stack([x['loss'] for x in d_outputs]).mean()
        avg_gloss = torch.stack([x['loss'] for x in g_outputs]).mean()
        self.logger.log_metrics({"d_loss": avg_dloss, "g_loss": avg_gloss},
                                self.current_epoch)
