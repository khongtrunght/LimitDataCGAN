import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .Vgg16PerceptualLoss import PerceptualLoss


class TransferBigGANLoss(nn.Module):
    def __init__(self,
                 perceptural=0.001,
                 earth_mover=0.1,
                 regulization=0.02,
                 norm_img=True,
                 norm_perceptural=True,
                 dis_perceptural="l2",
                 ):
        super(TransferBigGANLoss, self).__init__()

        self.scale_per = perceptural
        self.scale_emd = earth_mover
        self.scale_reg = regulization
        self.norm_img = norm_img
        self.norm_perceptural = norm_perceptural
        self.dis_perceptual = dis_perceptural
        self.PerceptualLoss = PerceptualLoss(loss_func=self.dis_perceptual)

    def pixel_level_loss(self, x, y):  # term 1
        '''
        x:generated image. shape = (batch,channel,h,w)
        y:target image. shape = (batch,channel,h,w)
        Vì sau khi cho ảnh y qua transforms.ToTensor() sẽ scale pixel từ 0->255 thành 0->1
        Ảnh generate được đi qua hàm tanh nên pixel có giá trị từ -1 -> 1
        Vì vậy phải chuyển y từ 0->1 thành -1->1
        '''
        return F.l1_loss(x, 2.0 * (y - 0.5))

    def semantic_level_loss(self, x, y):  # term 2

        return self.PerceptualLoss.forward(x, y)

    def EM_loss(self, z):  # term 3
        """
        EM distance between z and N(0,1)
        """
        dim_z = z.shape[1]
        N = z.shape[0]  # Batch size
        # r ~ N(0,1), lấy k = N*10
        r = torch.randn((N * 10, dim_z), device=z.device)
        z_mul_r = torch.matmul(z, r.permute(1, 0))  # z*r.T
        dist = torch.sum(z ** 2, dim=1, keepdim=True) - 2 * z_mul_r + torch.sum(r ** 2,
                                                                                dim=1)  # ||z_i - r_j|| (norm 2)
        '''
           [[||z_1 - r_1||, ||z_1 - r_2||, ..., ||z_1 - r_k|| ]
    dist =  [||z_2 - r_1||, ||z_2 - r_2||, ..., ||z_2 - r_k|| ]
            [     ...           ...        ...      ...       ]
            [||z_N - r_1||, ||z_N - r_2||, ..., ||z_N - r_k|| ]]
        '''
        return torch.mean(dist.min(dim=0)[0]) + torch.mean(dist.min(dim=1)[0])

    def regulization_loss(self, W):  # term 4
        return torch.mean(W ** 2)

    def forward(self, x, y, z, W):
        '''
        x:generated image. shape = (batch,channel,h,w)
        y:target image. shape = (batch,channel,h,w)
        z: seed image embeddings (BEFORE adding the noise of eps). shape = (batch,embedding_dim)
        W: model.linear.weight
        '''
        loss = 0
        term1 = self.pixel_level_loss(x, y)
        if self.norm_img:
            loss += term1/term1.item()
        else:
            loss += term1
        # print('term1:',self.pixel_level_loss(x, y))

        term2 = self.semantic_level_loss(x, y)
        if self.norm_perceptural:
            loss += self.scale_per * term2/term2.item()
        else:
            loss += self.scale_per * term2
        # print('term2:', self.semantic_level_loss(x, y))
        loss += self.scale_emd * self.EM_loss(z)
        # print('term3:',self.EM_loss(z))
        loss += self.scale_reg * self.regulization_loss(W)
        # print('term4:',self.regulization_loss(W))
        return loss
