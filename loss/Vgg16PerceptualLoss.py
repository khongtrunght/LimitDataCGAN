import torch
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms

'''
Perceptual loss functions are used when comparing two different images that look similar
, like the same photo but shifted by one pixel. The function is used to compare high level 
differences, like content and style discrepancies, between images.
'''


class PerceptualLoss(torch.nn.Module):  # sử dụng VGG16 model
    def __init__(self, perceptual_layers=[1, 3, 6, 8, 11, 13, 15, 18, 20, 22], loss_func="l2", requires_grad=False):
        '''
        Kiến trúc của mạng VGG16
        Sử dụng percepture loss ở sau layer ReLU: 1, 3, 6, 8, 11, 13, 15, 18, 20, 22
        (0): Conv2d
        (1): ReLU
        (2): Conv2d
        (3): ReLU
        (4): MaxPool2d
        (5): Conv2d
        (6): ReLU
        (7): Conv2d
        (8): ReLU
        (9): MaxPool2d
        (10): Conv2d
        (11): ReLU
        (12): Conv2d
        (13): ReLU
        (14): Conv2d
        (15): ReLU
        (16): MaxPool2d
        (17): Conv2d
        (18): ReLU
        (19): Conv2d
        (20): ReLU
        (21): Conv2d
        (22): ReLU
        (23): MaxPool2d
        (24): Conv2d
        (25): ReLU
        (26): Conv2d
        (27): ReLU
        (28): Conv2d
        (29): ReLU
        (30): MaxPool2d
        ... Fully Connected
        '''
        super(PerceptualLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features.eval()
        self.perceptual_layers = perceptual_layers
        self.vgg_partial = torch.nn.Sequential(
            *list(vgg_pretrained_features))[0:22+1]
        if loss_func == 'l1':
            self.loss_func = F.l1_loss
        elif loss_func == 'l2':
            self.loss_func = F.mse_loss
        else:
            self.loss_func = 0

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, batch):
        # Trước khi đưa vào VGG thì dùng std normalize
        # normalize using imagenet mean and std
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalize(batch)

    def forward_img(self, image):
        image = self.normalize(image)
        images_after_perceptual_layers = []
        for index, layer in enumerate(self.vgg_partial):
            image = layer(image)
            if index in self.perceptual_layers:
                images_after_perceptual_layers.append(image)
        return images_after_perceptual_layers

    def forward(self, x, y):
        '''
        x:generated image. shape = (batch,channel,h,w)
        y:target image. shape = (batch,channel,h,w)
        x đang ở scale [-1,1] nên phải chuyển về [0,1]
        '''
        x = (x + 1) / 2

        losses = []
        for x_i, y_i in zip(self.forward_img(x), self.forward_img(y)):
            loss_i = self.loss_func(x_i, y_i, reduction='mean')
            losses.append(loss_i)

        return sum(losses)
