import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchvision
from loss.transferBigGANLoss import TransferBigGANLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import BaseFinetuning
from scipy.stats import truncnorm


class GeneratorFreeze(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.generator, train_bn=True)


class TransferBigGAN(pl.LightningModule):
    def __init__(self, generator, data_size, n_classes=3, embedding_size=120, shared_embedding_size=128, cond_embedding_size=20, embedding_init="zero", conditional_init="pretrain", **kwargs):
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

        self.class_embeddings = nn.Embedding(
            n_classes, self.shared_embedding_size)

        # torch.nn.init.kaiming_normal_(self.class_embeddings.weight)
        # init weight từ shared embedding của gen
        if conditional_init == 'pretrain':
            cat = torch.LongTensor([281, 285, 282])
            dog = torch.LongTensor([227, 248, 273])
            lion = torch.LongTensor([292, 282, 291])

            #cat = torch.LongTensor([283, 281, 282])
            # dog = torch.LongTensor([198, 253, 232])
            # lion = torch.LongTensor([292, 282, 291])

            cat_embeds = self.generator.shared.weight.index_select(0, cat)
            dog_embeds = self.generator.shared.weight.index_select(0, dog)
            lion_embeds = self.generator.shared.weight.index_select(0, lion)

            cat_embeds = cat_embeds.mean(dim=0, keepdim=True)
            dog_embeds = dog_embeds.mean(dim=0, keepdim=True)
            lion_embeds = lion_embeds.mean(dim=0, keepdim=True)

            init_weight = torch.cat(
                [cat_embeds, dog_embeds, lion_embeds], dim=0)
        elif conditional_init == 'mean':
            mean_embeds = self.generator.shared.weight.mean(
                dim=0, keepdim=True)
            init_weight = mean_embeds.repeat(n_classes, 1)

        assert init_weight.shape == self.class_embeddings.weight.shape
        self.class_embeddings.weight.data = init_weight

        del generator.shared

        # self.losses de luu lai loss trong qua trinh tinh toan
        # self.losses = ...

        # to_do: set training params
        self.set_training_params()

        self.criterion = TransferBigGANLoss(
            **kwargs.get("loss")
        )

        self.lr_args = kwargs.get("lr")
        self.log_freq = kwargs.get("log_freq")

    # y là vector da di qua embeding

    def forward(self, z, y):  # y
        '''
        z: shape (batch_size, embedding_size)
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

        params = []
        params.append(
            {"params": list(self.batch_stat_generator_params().values()), "lr": self.lr_args.get('linear_batch_stat')})
        params.append(
            {"params": list(self.linear_generator_params().values()), "lr": self.lr_args.get('linear_gen')})
        params.append(
            {"params": list(self.after_first_linear_params().values()), "lr": self.lr_args.get('scale_shift')})
        params.append(
            {"params": list(self.embeddings_params().values()), "lr": self.lr_args.get('embed')})
        params.append({"params": list(
            self.class_conditional_embeddings_params().values()), "lr": self.lr_args.get('class_conditional_embed')})

        optimizer = optim.Adam(params, lr=0)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.lr_args.get('step'), gamma=self.lr_args.get('step_factor'))

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

        embeddings = self.embeddings(indices)
        embeddings_eps = torch.randn(
            embeddings.size(), device=self.device) * 0.01
        embeddings = embeddings + embeddings_eps

        real_label_embeddings = self.class_embeddings(real_label)

        img_gen = self(embeddings, real_label_embeddings)
        loss = self.criterion(img_gen, img, embeddings,
                              self.class_embeddings.weight)

        if self.global_step % self.log_freq == 0:
            self.random(f'samples_{self.global_step}.jpg', truncate=True)
            self.interpolate(
                f'interpolate_{self.global_step}.jpg', source=1, dist=10)
            self.reconstruct(f'reconstruct_{self.global_step}.jpg', indices_labels=(
                indices, real_label))
            self.logger.experiment.add_scalar("loss", loss, self.global_step)

        return {"loss": loss}

    def interpolate(self, out_path, source, dist, trncate=0.4, num=5):
        with torch.no_grad():
            self.eval()
            device = next(self.parameters()).device
            dataset_size = self.embeddings.weight.size()[0]
            indices = torch.tensor([source, dist], device=device)
            indices = indices.to(device)
            embeddings = self.embeddings(indices)
            embeddings = embeddings[[0]] * torch.linspace(1, 0, num, device=device)[
                :, None] + embeddings[[1]] * torch.linspace(0, 1, num, device=device)[:, None]

            batch_size = embeddings.size()[0]

            labels = [0, ] * batch_size
            labels = torch.tensor(labels, device=device)
            labels_embeddings = self.class_embeddings(labels)

            image_tensors = self(embeddings, labels_embeddings)
            grid = torchvision.utils.make_grid(image_tensors, normalize=True)
            self.logger.experiment.add_image(
                "interpolate", grid, self.global_step)

    def random(self, out_path, tmp=0.4, n=9, truncate=False):
        with torch.no_grad():
            self.eval()
            device = next(self.parameters()).device
            dataset_size = self.embeddings.weight.size()[0]
            dim_z = self.embeddings.weight.size(1)
            if truncate:
                embeddings = truncnorm(-tmp, tmp).rvs(n *
                                                      dim_z).astype("float32").reshape(n, dim_z)
            else:
                embeddings = np.random.normal(
                    0, tmp, size=(n, dim_z)).astype("float32")
            embeddings = torch.tensor(embeddings, device=device)
            batch_size = embeddings.size()[0]

            labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
            labels = torch.tensor(labels, device=device)
            label_embeddings = self.class_embeddings(labels)

            image_tensors = self(embeddings, label_embeddings)
            grid = torchvision.utils.make_grid(image_tensors, normalize=True)
            self.logger.experiment.add_image(
                "samples", grid, self.global_step)

    def reconstruct(self, out_path, indices_labels, add_small_noise=False):
        with torch.no_grad():
            self.eval()
            device = next(self.parameters()).device
            dataset_size = self.embeddings.weight.size()[0]
            indices, labels = indices_labels
            assert type(indices) == torch.Tensor
            indices = indices.to(device)
            embeddings = self.embeddings(indices)
            batch_size = embeddings.size()[0]

            # labels = [0, ] * batch_size
            # labels = torch.tensor(labels, device=device)
            labels = labels.to(device)
            labels_embeddings = self.class_embeddings(labels)

            if add_small_noise:
                embeddings += torch.randn(embeddings.size(),
                                          device=device)*0.01
            image_tensors = self(embeddings, labels_embeddings)
            grid = torchvision.utils.make_grid(image_tensors, normalize=True)
            self.logger.experiment.add_image(
                "reconstruct", grid, self.global_step)

    def bs_reg(self):
        bn1s = [self.generator.block[i]
                [0].bn1 for i in range(len(self.generator.block))]


if __name__ == '__main__':
    def setup_model(name, data_size, config_model, resume=None, biggan_pretrain_path='../data/G_ema.pth'):
        if name == 'TransferBigGAN':
            generator = biggan.Generator(**bigagn128config)
            generator.load_state_dict(torch.load(
                biggan_pretrain_path, map_location=lambda storage, loc: storage))
            model = tranfer_models[config_model
                                   ['name']](generator=generator, data_size=data_size, **config_model)
            return model

    config_model = {
        'name': 'TransferBigGAN',
        # learning rate
        'lr':
        {'linear_gen': 0.0000001,
         'linear_batch_stat': 0.0005,
         'embed': 0.05,
         'scale_shift': 0.0005,
         'class_conditional_embed': 0.001,
         'step': 3000,  # giam learning rate sau moi <step> iter,
         'step_factor': 0.1},
        # loss
        'loss':
        {'perceptural': 0.1,
         'earth_mover': 0.1,
         'regulization': 0.02,
         'norm_img': True,
         'norm_perceptural': True,
         'dis_perceptural': "l2"},


        'iteration':
        {'num_iter': 10000,
         'num_workers': 4},

        'n_classes': 3
    }

    setup_model('TransferBigGAN', 45, config_model)
