import torch
import pytorch_lightning as pl
import torch.nn as nn
from loss.transferBigGANLoss import TransferBigGANLoss
from visualizers import random
import torch.optim as optim
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import BaseFinetuning


class GeneratorFreeze(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.generator, train_bn=True)


class TransferBigGAN(pl.LightningModule):
    def __init__(self, generator, data_size, n_classes=3, embedding_size=120, shared_embedding_size=128, cond_embedding_size=20, embedding_init="zero", **kwargs):
        '''
        generator: pretrained generator
        data_size: number of training images, tac gia de nghi nen duoi 100
        shared_embedding_size: class shared embedding
        cond_embed_dim: class conditional embedding
        '''
        super(TransferBigGAN, self).__init__()
        self.generator = generator
        self.data_size = data_size
        self.embedding_size = embedding_size
        self.shared_embedding_size = shared_embedding_size
        self.cond_embedding_size = cond_embedding_size
        self.embedding_init = embedding_init

        self.embeddings = nn.Embedding(data_size, embedding_size)
        if embedding_init == "zero":
            self.embeddings.from_pretrained(torch.zeros(
                data_size, embedding_size), freeze=False)

        in_channels = self.generator.blocks[0][0].conv1.in_channels

        self.scale = nn.Parameter(torch.ones(in_channels,))
        self.shift = nn.Parameter(torch.zeros(in_channels,))

        init_weight = generator.shared.weight.mean(
            dim=0, keepdim=True).transpose(1, 0)

        self.class_embeddings = nn.Embedding(
            n_classes, self.shared_embedding_size)
        del generator.shared

        # self.losses de luu lai loss trong qua trinh tinh toan
        # self.losses = ...

        # to_do: set training params
        self.set_training_params()

        self.criterion = TransferBigGANLoss(
            **kwargs.get("loss")
        )

        self.lr_args = kwargs.get("lr")
    # y là vector da di qua embeding

    def forward(self, z, y):  # y
        '''
        z: shape (batch_size, chuabiet)
        y: shape (batch_size, shared_embedding_size)
        '''

        # tach z thanh nhieu phan va dung z khac nhau moi layer
        if self.generator.hier:
            zs = torch.split(z, self.generator.z_chunk_size, dim=1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            raise NotImplementedError("Chua lam")

        # layer dau tien
        h = self.generator.linear(z)
        h = h.view(h.size(0), -1, self.generator.bottom_width,
                   self.generator.bottom_width)

        # ap dung scale va shift
        h = h * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)

        # di qua cac block cua generator :
        for id, blocklist in enumerate(self.generator.blocks):
            for block in blocklist:
                h = block(h, ys[id])

        # dung tank owr output
        return torch.tanh(self.generator.output_layer(h))

    def set_training_params(self):
        '''
        chi de requires_grad = True cho shift and scale + fc cua linear , con lai requires_grad = False
        '''

        for param in self.parameters():
            param.requires_grad = False

        params_requires_grad = {}
        # scale va shift sau linear dau
        params_requires_grad.update(self.after_first_linear_params())
        # weight fully connected o generator duoc finetune voi learning rate rat nho
        params_requires_grad.update(self.linear_generator_params())
        # embedding cua class conditional
        params_requires_grad.update(self.class_conditional_embeddings_params())
        # embeding
        params_requires_grad.update(self.embeddings_params())
        # batch stat
        params_requires_grad.update(self.batch_stat_generator_params())

        for name, param in params_requires_grad.items():
            param.requires_grad = True

    def after_first_linear_params(self):
        return {"scale": self.scale, "shift": self.shift}

    def linear_generator_params(self):
        return {"generator.linear.weight": self.generator.linear.weight,
                "generator.linear.bias": self.generator.linear.bias}

    def class_conditional_embeddings_params(self):
        return {"class_embeddings.weight": self.class_embeddings.weight}

    def embeddings_params(self):
        return {"embeddings.weight": self.embeddings.weight}

    def batch_stat_generator_params(self):
        named_params = {}
        for name, module in self.named_modules():
            if name.split(".")[-1] in ["gain", "bias"]:
                for name2, param in module.named_parameters():
                    name = name + "." + name2
                    params = param
                    named_params[name] = params
        return named_params

    def configure_optimizers(self):

        def setup_optimizer(model, lr_g_batch_stat, linear_gen, scale_shift, embed, class_conditional_embed, step, step_factor=0.1):
            # group parameters by lr
            params = []
            params.append(
                {"params": list(model.batch_stat_generator_params().values()), "lr": lr_g_batch_stat})
            params.append(
                {"params": list(model.linear_generator_params().values()), "lr": linear_gen})
            params.append(
                {"params": list(model.after_first_linear_params().values()), "lr": scale_shift})
            params.append(
                {"params": list(model.embeddings_params().values()), "lr": embed})
            params.append({"params": list(
                model.class_conditional_embeddings_params().values()), "lr": class_conditional_embed})

            # setup optimizer
            # 0 is okay because sepcific lr is set by `params`
            optimizer = optim.Adam(params, lr=0)
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=step, gamma=step_factor)
            return optimizer, scheduler

        optimizer, scheduler = setup_optimizer(self,
                                               linear_gen=self.lr_args.get(
                                                   'linear_gen'),
                                               lr_g_batch_stat=self.lr_args.get(
                                                   'linear_batch_stat'),
                                               scale_shift=self.lr_args.get(
                                                   'scale_shift'),
                                               embed=self.lr_args.get('embed'),
                                               class_conditional_embed=self.lr_args.get(
                                                   'class_conditional_embed'),
                                               step=self.lr_args.get('step'),
                                               step_factor=self.lr_args.get(
                                                   'step_factor'),
                                               )

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            # "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": False,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return [optimizer], [lr_scheduler_config]

    def training_step(self, train_batch, batch_idx):

        self.eval()
        # phai de eval) vi muon giu nguyen batchnorm mean va var
        # van tune batchnorm scale and shift

        # dataset se co 2 label : indices va labels
        img, labels = train_batch
        indices = labels[0]
        real_label = labels[1]

        # to do scheduler step

        embeddings = self.embeddings(indices)
        embeddings_eps = torch.randn(
            embeddings.size(), device=self.device) * 0.01
        embeddings = embeddings + embeddings_eps

        real_label_embeddings = self.class_embeddings(real_label)

        img_gen = self(embeddings, real_label_embeddings)
        loss = self.criterion(img_gen, img, embeddings,
                              self.class_embeddings.weight)

        if self.global_step % 50 == 0:
            random(self, f'samples_{self.global_step}.jpg', truncate=True)

        # todo : self.losses.update(loss.item(), img.size(0))

        return {"loss": loss}

    # def training_epoch_end(self, outputs):
    #     if self.global_step % 500 == 0:
    #         super().training_epoch_end(outputs)
    #         random(self, f'samples_{self.global_step}.jpg', truncate=True)
