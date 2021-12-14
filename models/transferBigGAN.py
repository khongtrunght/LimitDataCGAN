import torch
import pytorch_lightning as pl
import torch.nn as nn
from loss.transferBigGANLoss import TransferBigGANLoss


class TransferBigGAN(pl.LightningModule):
    def __init__(self, generator, data_size, n_classes, embedding_size=120, shared_embedding_size=128, cond_embedding_size=20, embedding_init="zero"):
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

        self.embedding = nn.Embedding(data_size, embedding_size)
        if embedding_init == "zero":
            self.embedding.weight.data.zero_()

        in_channels = self.generator.blocks[0][0].conv1.in_channels

        self.scale = nn.Parameter(torch.ones(in_channels,))
        self.shift = nn.Parameter(torch.zeros(in_channels,))

        self.linear = nn.Linear(1, shared_embedding_size, bias=False)
        init_weight = generator.shared.weight.mean(
            dim=0, keepdim=True).transpose(1, 0)
        self.linear.weight.data = init_weight

        self.shared = nn.Embedding(n_classes, self.shared_embedding_size)

        # self.losses de luu lai loss trong qua trinh tinh toan
        #self.losses = ...

        # to_do: set training params

        self.criterion = TransferBigGANLoss(
            # to do
        )

    def forward(self, z, y):
        '''
        z: shape (batch_size, chuabiet)
        y: shape (batch_size, 1)
        '''

        y = self.linear(y)

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

    def after_first_linear_params(self):
        return {"scale": self.scale, "shift": self.shift}

    def linear_generator_params(self):
        return {"generator.linear.weight": self.generator.linear.weight,
                "generator.linear.bias": self.generator.linear.bias}

    def class_conditional_embeddings_params(self):
        return {"embeddings.weight": self.linear.weight}

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
        pass

    def training_step(self, train_batch, batch_idx):
        self.eval()
        # phai de eval) vi muon giu nguyen batchnorm mean va var
        # van tune batchnorm scale and shift
        img, indices = train_batch

        # to do scheduler step

        embeddings = self.embeddings(indices)
        embeddings_eps = torch.randn(embeddings.size(), device=self.device)

        embeddings = embeddings + embeddings_eps

        img_gen = self(embeddings)
        loss = self.criterion(img_gen, img, embeddings, self.linear.weight)

        #todo : self.losses.update(loss.item(), img.size(0))

        return {"loss": loss}
