import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def model_init(m, state_dictG, state_dictD):
    with torch.no_grad():
        m.generator.net[3][0].weight.copy_(state_dictG['main.9.weight'])
        # m.generator.net.0.1.bias.copy_(state_dictG['main.0.bias'])
        m.generator.net[1][0].weight.copy_(state_dictG['main.3.weight'])
        m.generator.net[2][0].weight.copy_(state_dictG['main.6.weight'])
        m.generator.net[4].weight.copy_(state_dictG['main.12.weight'])

        m.discriminator.net[2][0].weight.copy_(state_dictD['main.2.weight'])
        m.discriminator.net[3][0].weight.copy_(state_dictD['main.5.weight'])
        m.discriminator.net[4][0].weight.copy_(state_dictD['main.8.weight'])
        # m.discriminator.net[5].weight.copy_(state_dictD['main.11.weight'])

    return m
