# Source: https://arxiv.org/abs/1703.06953 -->
# --> https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer -->
# --> https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer/blob/master/msgnet.ipynb


import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from PIL import Image

from styletransfer.styletransfer import \
    StyleTransfer, StyleTransferType, StyleTransferConfig, StyleTransferInference

# TODO implement cuda - https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer/blob/75d256049a6af7ceccae01bd5c472556478048ea/experiments/main.py#L252
# from styletransfer.torchdevice import device


class MSGNetConfig(StyleTransferConfig):

    def __init__(self,
                 image_size: int = 512,
                 keep_content_aspect_ratio: bool = True,
                 model_path: str = './styletransfer/msgnet/21styles.model',
                 ):

        super().__init__(image_size, keep_content_aspect_ratio)
        self.model_path = model_path


# define Gram Matrix
class GramMatrix(nn.Module):
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


# proposed Inspiration(CoMatch) Layer
class Inspiration(nn.Module):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, C, B=1):
        super().__init__()
        # B is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
        # non-parameter buffer
        self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        # input X is a 3D feature map
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
                         X.view(X.size(0), X.size(1), -1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x ' + str(self.C) + ')'


# some basic layers, with reflectance padding
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 upsample=None):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        self.reflection_padding = kernel_size // 2
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class Bottleneck(nn.Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes*self.expansion,
                                            kernel_size=1, stride=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       ConvLayer(planes, planes, kernel_size=3, stride=stride)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes*self.expansion, kernel_size=1,
                                 stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)


class UpBottleneck(nn.Module):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes*self.expansion,
                                                kernel_size=1, stride=1,
                                                upsample=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       UpsampleConvLayer(planes, planes, kernel_size=3,
                                         stride=1, upsample=stride)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes*self.expansion, kernel_size=1,
                                 stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


# the MSG-Net
class Net(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64,
                 norm_layer=nn.InstanceNorm2d, n_blocks=6, gpu_ids=[]):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.gram = GramMatrix()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        model1 = []
        model1 += [ConvLayer(input_nc, 64, kernel_size=7, stride=1),
                   norm_layer(64),
                   nn.ReLU(inplace=True),
                   block(64, 32, 2, 1, norm_layer),
                   block(32*expansion, ngf, 2, 1, norm_layer)]
        self.model1 = nn.Sequential(*model1)

        model = []
        self.ins = Inspiration(ngf*expansion)
        model += [self.model1]
        model += [self.ins]

        for i in range(n_blocks):
            model += [block(ngf*expansion, ngf, 1, None, norm_layer)]

        model += [upblock(ngf*expansion, 32, 2, norm_layer),
                  upblock(32*expansion, 16, 2, norm_layer),
                  norm_layer(16*expansion),
                  nn.ReLU(inplace=True),
                  ConvLayer(16*expansion, output_nc, kernel_size=7, stride=1)]

        self.model = nn.Sequential(*model)

    def setTarget(self, Xs):
        f = self.model1(Xs)
        G = self.gram(f)
        self.ins.setTarget(G)

    def forward(self, input):
        return self.model(input)


class MSGNetInference(StyleTransferInference):

    @staticmethod
    def image_to_tensor(img, size=None, scale=None, keep_asp=False):
        if size is not None:
            if keep_asp:
                size2 = int(size * 1.0 / img.size[0] * img.size[1])
                img = img.resize((size, size2), Image.ANTIALIAS)
            else:
                img = img.resize((size, size), Image.ANTIALIAS)

        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                             Image.ANTIALIAS)
        img = np.array(img).transpose(2, 0, 1)
        img = torch.from_numpy(img).float().unsqueeze(0)
        return img

    @staticmethod
    def preprocess_batch(batch):
        batch = batch.transpose(0, 1)
        (r, g, b) = torch.chunk(batch, 3)
        batch = torch.cat((b, g, r))
        batch = batch.transpose(0, 1)
        return batch

    @staticmethod
    def tensor_to_rgbimage(tensor, cuda=False):
        if cuda:
            img = tensor.clone().cpu().clamp(0, 255).numpy()
        else:
            img = tensor.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype('uint8')
        img = Image.fromarray(img)
        return img

    @staticmethod
    def tensor_to_image(tensor, cuda=False):
        (b, g, r) = torch.chunk(tensor, 3)
        tensor = torch.cat((r, g, b))
        img = MSGNetInference.tensor_to_rgbimage(tensor, cuda)
        return img

    def __init__(self, config: MSGNetConfig, content_image: Image, style_image: Image, model_dict):
        super().__init__(config, content_image, style_image)
        self._model_dict = model_dict

    def _get_style_model(self, style: Image):
        style_model = Net(ngf=128)
        model_dict = self._model_dict.copy()

        for key, value in self._model_dict.items():
            if key.endswith(('running_mean', 'running_var')):
                del model_dict[key]

        style_model.load_state_dict(model_dict, False)

        style_v = Variable(style)
        style_model.setTarget(style_v)
        return style_model

    def _inference(self) -> Image:
        config = self._config
        content = MSGNetInference.image_to_tensor(
            self._content_image, size=config.image_size, keep_asp=config.keep_content_aspect_ratio)
        style = MSGNetInference.image_to_tensor(self._style_image, size=config.image_size)
        style = MSGNetInference.preprocess_batch(style)

        style_model = self._get_style_model(style)

        content_image = Variable(MSGNetInference.preprocess_batch(content))
        output = style_model(content_image)
        result_image = MSGNetInference.tensor_to_image(output.data[0], False)
        return result_image


class MSGNet(StyleTransfer):

    def __init__(self, config: MSGNetConfig = None):
        super().__init__(config)

        # TODO Support cuda
        self._model_dict = torch.load(self._config.model_path, map_location=torch.device('cpu'))

    def get_default_config(self):
        return MSGNetConfig()

    def get_inference(self, content_image: Image, style_image: Image, config: MSGNetConfig = None):
        return MSGNetInference(config or self._config, content_image, style_image, self._model_dict)

    def type(self):
        return StyleTransferType.MSGNet
